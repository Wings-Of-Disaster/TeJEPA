"""CPU tests: full base loss, fresh calibration, resume, and encoder exclusion."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import timm
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import train
from models import Student,unpack


def config():
    with patch.object(sys,'argv',['train.py','--data_dir','unused','--teacher_weights','unused',
         '--student','vit_tiny_patch16_224','--teacher','vit_tiny_patch16_224',
         '--global_size','32','--local_size','16','--batch_size','2','--num_workers','0',
         '--precision','fp32','--device','cpu','--slices','16','--sketches','16']):
        return train.parse_args()


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(47)
        self.args=config()
        self.teacher=timm.create_model(self.args.teacher,pretrained=False,num_classes=0,img_size=32).eval().requires_grad_(False)
        self.student=Student(self.args,self.teacher.num_features).train()
        self.batch={f'global_{i}':dict(image=torch.randn(2,3,32,32),
            box=torch.tensor([[0.,0.,1.,1.]]).repeat(2,1),flip=torch.tensor([False,False])) for i in range(2)}
        self.batch.update({f'local_{i}':dict(image=torch.randn(2,3,16,16)) for i in range(6)})

    def test_base_loss_keeps_local_views_sigreg_and_projector_gradients(self):
        with torch.no_grad():tokens=train.teacher_tokens(self.teacher,self.batch,self.args)
        self.student.tereg_weight.fill_(.05)
        terms=self.student(self.batch,tokens)
        terms['loss'].backward()
        self.assertGreater(float(terms['prediction'].detach()),0)
        self.assertGreater(float(terms['sigreg'].detach()),0)
        self.assertTrue(any(p.grad is not None and p.grad.norm()>0 for p in self.student.model.projector.parameters()))
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))
        self.assertTrue(self.student.backbone.training)
        different=copy.deepcopy(self.batch)
        for i in range(6):different[f'local_{i}']['image']+=5
        other=self.student(different,tokens)
        self.assertNotAlmostEqual(float(terms['prediction'].detach()),float(other['prediction'].detach()),places=5)
        # The deterministic teacher branch uses the unchanged two global crops.
        torch.testing.assert_close(terms['kd'],other['kd'])
        expected=.95*terms['prediction']+.05*terms['sigreg']+terms['kd']+.05*terms['tereg']
        torch.testing.assert_close(terms['loss'],expected)

    def test_calibration_changes_only_buffers_never_parameters(self):
        parameters={n:p.detach().clone() for n,p in self.student.named_parameters()}
        generator=torch.Generator().manual_seed(18)
        self.student.tereg.initialize(torch.randn(12,192,generator=generator),torch.randn(12,192,generator=generator))
        before_buffers={n:v.clone() for n,v in self.student.named_buffers()}
        report=train.calibrate_weights(self.student,self.teacher,[self.batch],self.args,torch.device('cpu'),1)
        for n,p in self.student.named_parameters():torch.testing.assert_close(p,parameters[n],atol=0,rtol=0)
        for n,v in self.student.named_buffers():
            if n not in ('patch_weight','gram_weight','tereg_weight'):
                torch.testing.assert_close(v,before_buffers[n],atol=0,rtol=0)
        self.assertEqual(report['encoder_updates'],0)
        self.assertLessEqual(max(report['weighted_tereg_to_kd']),.100001)
        self.assertEqual(int(self.student.tereg.frequency_step),0)
        self.assertFalse(any(k.startswith('teacher.') for k in self.student.state_dict()))


class EntryPointTests(unittest.TestCase):
    def test_two_rank_full_training_calibration_and_checkpoint(self):
        torch.set_num_threads(1)
        temporary_root=ROOT/'.test-tmp';temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            root=Path(directory);rng=np.random.default_rng(17)
            for cls in ['a','b']:
                folder=root/'images/train'/cls;folder.mkdir(parents=True)
                for i in range(4):
                    Image.fromarray(rng.integers(0,256,(48,48,3),dtype=np.uint8)).save(folder/f'{i}.png')
            teacher=timm.create_model('vit_tiny_patch16_224',pretrained=False,num_classes=0,img_size=32)
            weights=root/'teacher.pt';torch.save(teacher.state_dict(),weights);del teacher
            arguments=['--data_dir',str(root/'images'),'--teacher_weights',str(weights),
                '--output',str(root/'run'),'--student','vit_tiny_patch16_224','--teacher','vit_tiny_patch16_224',
                '--global_size','32','--local_size','16','--epochs','1','--warmup_epochs','0',
                '--batch_size','2','--num_workers','0','--precision','fp32','--device','cpu',
                '--slices','16','--sketches','16','--calibration_images','4','--calibration_batches','1','--log_every','1']
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
            result=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone',
                '--nproc_per_node','2','train.py',*arguments],cwd=ROOT,env=env,text=True,capture_output=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            saved=torch.load(root/'run/last.ckpt',map_location='cpu',weights_only=False)
            self.assertEqual(saved['recipe']['world_size'],2)
            self.assertEqual(saved['global_step'],2)
            self.assertEqual(len(saved['rng']),2)
            self.assertEqual(int(saved['model']['model.sigreg.global_step']),2)
            report=json.loads((root/'run/calibration.json').read_text())
            self.assertLessEqual(max(report['weights']['weighted_tereg_to_kd']),.100001)

    def test_cpu_fresh_training_and_full_state_resume(self):
        torch.set_num_threads(1)
        temporary_root=ROOT/'.test-tmp';temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            root=Path(directory);rng=np.random.default_rng(15)
            for split,count in [('train',2),('val',1)]:
                for cls in ['a','b']:
                    folder=root/'images'/split/cls;folder.mkdir(parents=True)
                    for i in range(count):Image.fromarray(rng.integers(0,256,(48,48,3),dtype=np.uint8)).save(folder/f'{i}.png')
            teacher=timm.create_model('vit_tiny_patch16_224',pretrained=False,num_classes=0,img_size=32)
            weights=root/'teacher.pt';torch.save(teacher.state_dict(),weights);del teacher
            output=root/'run'
            arguments=['--data_dir',str(root/'images'),'--teacher_weights',str(weights),'--output',str(output),
                '--student','vit_tiny_patch16_224','--teacher','vit_tiny_patch16_224',
                '--global_size','32','--local_size','16','--epochs','2','--warmup_epochs','0',
                '--batch_size','2','--num_workers','0','--precision','fp32','--device','cpu',
                '--slices','16','--sketches','16','--calibration_images','4','--calibration_batches','1','--log_every','1']
            # Interrupt only after a complete, atomically saved epoch. This tests
            # actual optimizer, frequency, and RNG restoration without another seed.
            script="import train; original=train.save_checkpoint\ndef stop(*a,**kw):\n original(*a,**kw)\n raise SystemExit(0)\ntrain.save_checkpoint=stop\ntrain.main()"
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
            first=subprocess.run([sys.executable,'-c',script,*arguments],cwd=ROOT,env=env,text=True,capture_output=True,timeout=180)
            self.assertEqual(first.returncode,0,first.stdout+first.stderr)
            saved=torch.load(output/'last.ckpt',map_location='cpu',weights_only=False)
            self.assertEqual(saved['epoch'],0);self.assertEqual(saved['global_step'],2)
            self.assertEqual(saved['recipe']['initialization'],'random')
            self.assertEqual(saved['recipe']['global_views'],2);self.assertEqual(saved['recipe']['local_views'],6)
            calibration=json.loads((output/'calibration.json').read_text())
            self.assertEqual(calibration['coordinates']['images'],4)
            self.assertEqual(calibration['weights']['encoder_updates'],0)
            del saved
            resumed=subprocess.run([sys.executable,'train.py',*arguments,'--resume',str(output/'last.ckpt')],
                cwd=ROOT,env=env,text=True,capture_output=True,timeout=180)
            self.assertEqual(resumed.returncode,0,resumed.stdout+resumed.stderr)
            saved=torch.load(output/'last.ckpt',map_location='cpu',weights_only=False)
            self.assertEqual(saved['epoch'],1);self.assertEqual(saved['global_step'],4)
            self.assertEqual(int(saved['model']['tereg.frequency_step']),4)
            self.assertEqual(int(saved['model']['model.sigreg.global_step']),4)
            self.assertFalse(any(k.startswith('teacher.') for k in saved['model']))
            self.assertTrue(all(torch.isfinite(v).all() for v in saved['model'].values() if torch.is_floating_point(v)))
            # Frozen evaluation loads the random-initialized-training checkpoint
            # through exactly the same public encoder loader as a real run.
            evaluation=root/'evaluation'
            result=subprocess.run([sys.executable,'eval_imagenet.py','--data_dir',str(root/'images'),
                '--checkpoint',str(output/'last.ckpt'),'--encoder','vit_tiny_patch16_224',
                '--output',str(evaluation),'--device','cpu','--epochs','1','--num_workers','0',
                '--batch_size','2','--feature_batch_size','2'],cwd=ROOT,env=env,text=True,capture_output=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(json.loads((evaluation/'metrics.json').read_text())['validation']['samples'],2)


if __name__=='__main__':unittest.main()
