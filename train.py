"""One-stage JEPA + teacher distillation: random student, frozen teacher."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import ImageFolder

from data import TrainImages, CalibrationImages
from models import Student, build_teacher, unpack, deterministic, load_checkpoint


def move(value, device):
    if isinstance(value, dict):
        return {k:move(v,device) for k,v in value.items()}
    return value.to(device,non_blocking=True) if torch.is_tensor(value) else value


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def setup(args):
    world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0))
    local=int(os.environ.get('LOCAL_RANK',0))
    device=torch.device('cuda',local) if args.device=='cuda' else torch.device('cpu')
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    return rank,world,device


def collective_finite(value,device):
    failed=torch.tensor(int(not torch.isfinite(value).all()),device=device)
    if dist.is_initialized():dist.all_reduce(failed,op=dist.ReduceOp.MAX)
    if failed.item():raise FloatingPointError('Non-finite loss or gradient; no optimizer update performed')


def teacher_tokens(teacher,batch,args):
    image=torch.cat([batch[f'global_{i}']['image'] for i in range(2)])
    with torch.no_grad():
        return unpack(teacher.forward_features(image),teacher.num_prefix_tokens)


def autocast(device,args):
    return torch.autocast(device_type=device.type,dtype=torch.bfloat16,
                           enabled=args.precision=='bf16')


@torch.no_grad()
def initialize_coordinates(student,teacher,base,args,rank,device):
    report=None
    if rank==0:
        indices=torch.randperm(len(base),generator=torch.Generator().manual_seed(args.seed+17))
        indices=indices[:min(args.calibration_images,len(base))]
        dataset=CalibrationImages(base,indices,args.global_size)
        loader=DataLoader(dataset,batch_size=args.batch_size,num_workers=args.num_workers)
        s,t=[],[]
        with deterministic(student.backbone):
            for image in loader:
                image=image.to(device)
                with autocast(device,args):
                    sc,_=unpack(student.backbone.forward_features(image),student.backbone.num_prefix_tokens)
                    tc,_=unpack(teacher.forward_features(image),teacher.num_prefix_tokens)
                s.append(sc.float().cpu());t.append(tc.float().cpu())
        report=student.tereg.initialize(torch.cat(s),torch.cat(t))
    if dist.is_initialized():
        for buffer in student.tereg.buffers():dist.broadcast(buffer,src=0)
    return report


def calibrate_weights(student,teacher,loader,args,device,world):
    """No optimizer updates; fit loss scales at this random initialization."""
    buffers={k:v.detach().clone() for k,v in student.named_buffers()}
    parameters=list(student.backbone.parameters())
    names=['cls','rkd','patch','gram','patch_reference','gram_reference','tereg']
    records=[];vectors=[];iterator=iter(loader)
    try:
        for batch_id in range(min(args.calibration_batches,len(loader))):
            batch=move(next(iterator),device)
            with autocast(device,args):
                terms=student(batch,teacher_tokens(teacher,batch,args),include_base=False)
            values={}
            for name in names:
                gradients=torch.autograd.grad(terms[name],parameters,retain_graph=True,allow_unused=True)
                vector=torch.cat([(torch.zeros_like(p) if g is None else g.detach()).flatten().float()
                                   for p,g in zip(parameters,gradients)])
                if dist.is_initialized():dist.all_reduce(vector);vector/=world
                collective_finite(vector,device)
                values[name]=vector
            vectors.append(values)
            records.append(dict(batch=batch_id,gradient_norms={k:float(v.norm()) for k,v in values.items()}))
        patch=min(.5*r['gradient_norms']['patch_reference']/max(r['gradient_norms']['patch'],1e-20) for r in records)
        gram=min(.3*r['gradient_norms']['gram_reference']/max(r['gradient_norms']['gram'],1e-20) for r in records)
        for record,v in zip(records,vectors):
            kd=v['cls']+v['rkd']+patch*v['patch']+gram*v['gram']
            record['kd_norm']=float(kd.norm())
        reg=min(args.reg_gradient_budget*r['kd_norm']/max(r['gradient_norms']['tereg'],1e-20) for r in records)
        if not all(math.isfinite(x) and 0<x for x in [patch,gram,reg]):
            raise ValueError('Degenerate calibration gradients; inspect data and teacher')
    finally:
        del iterator
        for k,v in student.named_buffers():v.copy_(buffers[k])
    student.patch_weight.fill_(patch);student.gram_weight.fill_(gram);student.tereg_weight.fill_(reg)
    return dict(patch_weight=patch,gram_weight=gram,tereg_weight=reg,records=records,
                weighted_tereg_to_kd=[reg*r['gradient_norms']['tereg']/r['kd_norm'] for r in records],
                rule='Match patch/Gram gradient scales to uniform-token/elementwise-Gram references; initial TeReg budget relative to KD only',
                encoder_updates=0)


def learning_rate(step,total,warmup,peak):
    if warmup and step<warmup:return peak*(.01+.99*step/warmup)
    fraction=min(1.,max(0.,(step-warmup)/max(1,total-warmup)))
    return peak*(.001+.999*.5*(1+math.cos(math.pi*fraction)))


def rng_state(device):
    return dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),
                cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)


def restore_rng(value,device):
    torch.set_rng_state(value['torch']);np.random.set_state(value['numpy']);random.setstate(value['python'])
    if value['cuda'] is not None and device.type=='cuda':torch.cuda.set_rng_state(value['cuda'],device)


def save_checkpoint(student,optimizer,epoch,step,recipe,args,rank,world):
    local=rng_state(next(student.parameters()).device);states=[None]*world
    if world>1:dist.all_gather_object(states,local)
    else:states=[local]
    if rank==0:
        path=Path(args.output)/'last.ckpt';temporary=path.with_suffix('.tmp')
        payload=dict(model=student.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                     global_step=step,recipe=recipe,rng=states)
        try:
            with temporary.open('wb') as f:
                torch.save(payload,f);f.flush();os.fsync(f.fileno())
            os.replace(temporary,path)
        finally:temporary.unlink(missing_ok=True)
    if world>1:dist.barrier()


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir',required=True)
    p.add_argument('--teacher_weights',required=True)
    p.add_argument('--output',default='outputs/tereg')
    p.add_argument('--resume',default='',help='Resume this same one-stage training protocol')
    p.add_argument('--student',default='vit_small_patch16_224')
    p.add_argument('--teacher',default='vit_small_patch16_dinov3.lvd1689m')
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--batch_size',type=int,default=64)
    p.add_argument('--num_workers',type=int,default=8)
    p.add_argument('--global_size',type=int,default=224)
    p.add_argument('--local_size',type=int,default=96)
    p.add_argument('--local_views',type=int,default=6)
    p.add_argument('--lr',type=float,default=.0005)
    p.add_argument('--weight_decay',type=float,default=.05)
    p.add_argument('--warmup_epochs',type=float,default=5.)
    p.add_argument('--drop_path',type=float,default=.1)
    p.add_argument('--prediction_weight',type=float,default=.95)
    p.add_argument('--sigreg_weight',type=float,default=.05)
    p.add_argument('--slices',type=int,default=1000)
    p.add_argument('--sketches',type=int,default=1024)
    p.add_argument('--rkd_tau',type=float,default=.1)
    p.add_argument('--patch_floor',type=float,default=.1)
    p.add_argument('--patch_temperature',type=float,default=.2)
    p.add_argument('--gram_delta',type=float,default=.01)
    p.add_argument('--calibration_images',type=int,default=2000)
    p.add_argument('--calibration_batches',type=int,default=2)
    p.add_argument('--reg_gradient_budget',type=float,default=.1)
    p.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--log_every',type=int,default=20)
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    if min(args.epochs,args.batch_size,args.slices,args.sketches,args.calibration_images,args.calibration_batches,args.log_every)<=0:
        p.error('Epochs, batch sizes, calibration and sketch counts must be positive')
    if args.batch_size<2 or args.calibration_images<2 or args.num_workers<0 or args.local_views<1:
        p.error('Need batch/calibration size >=2 and at least one local view')
    if not 0<=args.warmup_epochs<args.epochs or not 0<=args.drop_path<1:
        p.error('Invalid warmup or drop-path setting')
    if min(args.lr,args.prediction_weight,args.sigreg_weight,args.rkd_tau,args.patch_temperature,args.gram_delta)<=0:
        p.error('Learning rate and objective scales must be positive')
    if args.weight_decay<0 or not 0<args.patch_floor<=1 or not 0<args.reg_gradient_budget<1:
        p.error('Invalid loss budget, reliability floor, or weight decay')
    return args


def main():
    args=parse_args();torch.set_num_threads(4)
    rank,world,device=setup(args)
    try:
        seed(args.seed)
        base=ImageFolder(Path(args.data_dir)/'train')
        dataset=TrainImages(base,args)
        sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,
                                   seed=args.seed,drop_last=True) if world>1 else None
        loader=DataLoader(dataset,batch_size=args.batch_size,sampler=sampler,shuffle=sampler is None,
            drop_last=True,num_workers=args.num_workers,pin_memory=device.type=='cuda')
        if not len(loader):raise ValueError('Dataset too small for one complete batch per rank')
        teacher=build_teacher(args).to(device)
        student=Student(args,teacher.num_features).to(device).train()
        recipe={k:v for k,v in vars(args).items() if k not in
                {'output','resume','log_every','device','teacher_weights'}}
        recipe.update(world_size=world,dataset_size=len(dataset),classes=base.classes,
                      teacher_sha256=sha256(args.teacher_weights),steps_per_epoch=len(loader),
                      initialization='random',global_views=2)
        output=Path(args.output)
        if rank==0:
            output.mkdir(parents=True,exist_ok=True)
        if world>1:dist.barrier()
        if not args.resume and (output/'last.ckpt').exists():
            raise FileExistsError('Output already contains a checkpoint; use resume or a new directory')
        optimizer=torch.optim.AdamW(student.parameters(),lr=args.lr,weight_decay=args.weight_decay,betas=(.9,.999))
        start_epoch=step=0
        if args.resume:
            saved=load_checkpoint(args.resume)
            if saved['recipe']!=recipe:raise ValueError('Resume requires the same model, data and training protocol')
            student.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
            start_epoch=saved['epoch']+1;step=saved['global_step']
            if step!=start_epoch*len(loader) or start_epoch>=args.epochs:
                raise ValueError('Resume requires an unfinished, complete-epoch checkpoint')
            restore_rng(saved['rng'][rank],device);del saved
        else:
            coordinate_report=initialize_coordinates(student,teacher,base,args,rank,device)
            weight_report=calibrate_weights(student,teacher,loader,args,device,world)
            seed(args.seed+rank)
            if rank==0:
                (output/'calibration.json').write_text(json.dumps(dict(coordinates=coordinate_report,weights=weight_report),indent=2)+'\n')
        if rank==0:(output/'protocol.json').write_text(json.dumps(recipe,indent=2)+'\n')
        wrapped=DistributedDataParallel(student,device_ids=[device.index] if device.type=='cuda' else None) if world>1 else student
        total=args.epochs*len(loader);warmup=round(args.warmup_epochs*len(loader))
        for epoch in range(start_epoch,args.epochs):
            if sampler is not None:sampler.set_epoch(epoch)
            wrapped.train();started=time.monotonic()
            for batch in loader:
                batch=move(batch,device)
                for group in optimizer.param_groups:group['lr']=learning_rate(step,total,warmup,args.lr)
                with autocast(device,args):
                    terms=wrapped(batch,teacher_tokens(teacher,batch,args))
                collective_finite(terms['loss'],device)
                optimizer.zero_grad(set_to_none=True);terms['loss'].backward()
                norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.)
                collective_finite(norm,device);optimizer.step();step+=1
                if step%args.log_every==0 or step==total:
                    keys=list(terms);values=torch.stack([terms[k].detach().float() for k in keys])
                    if world>1:dist.all_reduce(values);values/=world
                    record=dict(epoch=epoch+1,step=step,lr=optimizer.param_groups[0]['lr'],
                                **dict(zip(keys,values.tolist())))
                    if rank==0:
                        with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                        print(json.dumps(record,allow_nan=False),flush=True)
            save_checkpoint(student,optimizer,epoch,step,recipe,args,rank,world)
            if rank==0:print(f'Epoch {epoch+1}/{args.epochs}: {time.monotonic()-started:.1f}s; saved last.ckpt',flush=True)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
