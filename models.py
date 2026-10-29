"""Randomly initialized LeJEPA student with frozen-teacher supervision."""
from contextlib import contextmanager
import math

import timm
import torch
from torch import nn
import torch.nn.functional as F
from stable_pretraining.methods.lejepa import LeJEPA

from losses import reliable_patch_loss, compressed_gram_loss
from tereg import TeReg


def load_checkpoint(path):
    if str(path).endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(str(path))
    return torch.load(path, map_location='cpu', weights_only=False)


def state_dict(checkpoint):
    for key in ('state_dict', 'model'):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def unpack(features, prefix=1):
    if isinstance(features, dict):
        if 'x_norm_clstoken' in features:
            return features['x_norm_clstoken'], features['x_norm_patchtokens']
        features = features['x']
    if isinstance(features, (tuple, list)):
        features = features[0]
    return features[:, 0], features[:, prefix:]


@contextmanager
def deterministic(backbone):
    flags = [(layer, layer.training) for layer in backbone.modules()]
    try:
        backbone.eval()
        yield
    finally:
        for layer, training in flags:
            layer.training = training


def build_teacher(args):
    teacher = timm.create_model(args.teacher, pretrained=False, num_classes=0,
                                img_size=args.global_size)
    teacher.load_state_dict(state_dict(load_checkpoint(args.teacher_weights)), strict=True)
    return teacher.eval().requires_grad_(False)


class Student(nn.Module):
    def __init__(self, args, teacher_dim):
        super().__init__()
        # No initialization checkpoint, pretrained student, or preliminary stage.
        self.model = LeJEPA(encoder_name=args.student, pretrained=False,
            drop_path_rate=args.drop_path, lamb=args.sigreg_weight, n_slices=args.slices)
        dim = self.model.embed_dim
        if dim != teacher_dim:
            raise ValueError('Full-dimensional TeReg requires equal student/teacher dimensions')
        # Preserve the original per-rank pooled-view SIGReg estimator. TeReg is global.
        self.model.sigreg._is_ddp = False
        self.model.sigreg.ep._is_ddp = False
        self.model.sigreg.ep.world_size = 1
        self.align = nn.Sequential(nn.Linear(dim, 2048), nn.BatchNorm1d(2048),
                                   nn.ReLU(), nn.Linear(2048, teacher_dim))
        self.patch_align = nn.Linear(dim, teacher_dim)
        self.tereg = TeReg(dim, args.sketches)
        self.register_buffer('patch_weight', torch.tensor(.5))
        self.register_buffer('gram_weight', torch.tensor(.3))
        self.register_buffer('tereg_weight', torch.tensor(0.))
        self.args = args

    @property
    def backbone(self):
        return self.model.backbone

    def forward(self, batch, teacher_tokens, include_base=True):
        args = self.args
        globals_ = [batch[f'global_{i}']['image'] for i in range(2)]
        image = torch.cat(globals_)
        n = len(globals_[0])
        with deterministic(self.backbone):
            student_tokens = self.backbone.forward_features(image)
        sc, sp = unpack(student_tokens, self.backbone.num_prefix_tokens)
        tc, tp = teacher_tokens
        with torch.autocast(device_type=image.device.type, enabled=False):
            sc, sp, tc, tp = sc.float(), sp.float(), tc.float(), tp.float()
            aligned = self.align(sc[:n])
            cls = F.mse_loss(aligned, tc[:n])
            sn, tn = F.normalize(aligned, dim=-1), F.normalize(tc[:n], dim=-1)
            rkd = F.kl_div(F.log_softmax(sn @ sn.T / args.rkd_tau, dim=-1),
                           F.softmax(tn @ tn.T / args.rkd_tau, dim=-1), reduction='batchmean')
            sp = self.patch_align(sp)
            hs, ht = math.isqrt(sp.size(1)), math.isqrt(tp.size(1))
            if hs*hs != sp.size(1) or ht*ht != tp.size(1):
                raise ValueError('Patch tokens must form square grids')
            sp = F.interpolate(sp.transpose(1,2).reshape(2*n,-1,hs,hs), size=(ht,ht),
                                mode='bilinear', align_corners=False).flatten(2).transpose(1,2)
            sp, tp = F.normalize(sp,dim=-1), F.normalize(tp,dim=-1)
            boxes = torch.stack([batch[f'global_{i}']['box'] for i in range(2)])
            flips = torch.stack([batch[f'global_{i}']['flip'] for i in range(2)])
            patch, extra = reliable_patch_loss(sp.reshape(2,n,ht*ht,-1),
                tp.reshape(2,n,ht*ht,-1), boxes, flips, args.patch_floor, args.patch_temperature)
            gram = compressed_gram_loss(sp,tp,delta=args.gram_delta)
            reg, diagnostics = self.tereg(sc[:n],tc[:n])
            kd = cls + rkd + self.patch_weight*patch + self.gram_weight*gram
        prediction, sigreg = kd*0, kd*0
        if include_base:
            # Full original two-global/six-local LeJEPA prediction + SIGReg branch.
            features = [self.backbone(image)]
            locals_ = [batch[f'local_{i}']['image'] for i in range(args.local_views)]
            if locals_:
                features.append(self.backbone(torch.cat(locals_)))
            projected = self.model.projector(torch.cat(features))
            projected = projected.reshape(2+args.local_views,n,-1)
            prediction = (projected-projected[:2].mean(0)[None]).square().mean()
            with torch.autocast(device_type=image.device.type,enabled=False):
                sigreg = self.model.sigreg(projected.flatten(0,1).float())
        loss = args.prediction_weight*prediction + args.sigreg_weight*sigreg + kd + self.tereg_weight*reg
        terms = dict(loss=loss,kd=kd,cls=cls,patch=patch,gram=gram,rkd=rkd,
                     prediction=prediction,sigreg=sigreg,tereg=reg)
        if not include_base:
            # Scale references only: no alternative training rows are implemented.
            terms['patch_reference'] = (1-(sp[:n]*tp[:n]).sum(-1)).mean()
            terms['gram_reference'] = F.mse_loss(sp[:n]@sp[:n].transpose(1,2),tp[:n]@tp[:n].transpose(1,2))
        terms.update({f'patch_{k}':v.detach() for k,v in extra.items()})
        terms.update({f'tereg_{k}':v.detach() for k,v in diagnostics.items()})
        return terms


def load_backbone(encoder, path):
    backbone = timm.create_model(encoder, pretrained=False, num_classes=0)
    state = state_dict(load_checkpoint(path))
    prefix = 'model.backbone.'
    filtered = {k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    if not filtered:
        raise ValueError('Checkpoint has no student backbone')
    backbone.load_state_dict(filtered, strict=True)
    return backbone.eval().requires_grad_(False)
