"""Stage-2 TeJEPA training with a frozen visual teacher."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import lightning as pl
import stable_pretraining as spt
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_pretraining.data import transforms
from stable_pretraining.methods.lejepa import LeJEPA
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from models import load_checkpoint, load_student_weights, state_dict


def configure_runtime():
    # Keep artifact handling explicit; optional library callbacks can collect
    # environment metadata or add extra checkpoint outputs.
    spt.set(default_callbacks={name: False for name in (
        "progress_bar", "registry", "logging", "env_dump", "trainer_info",
        "sklearn_checkpoint", "wandb_checkpoint", "trackio_checkpoint",
        "swanlab_checkpoint", "module_summary", "slurm_info", "unused_params", "hf_checkpoint",
    )})


class AlignmentHead(nn.Module):
    def __init__(self, student_dim: int, teacher_dim: int, hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(student_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, teacher_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchAlignmentHead(nn.Module):
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _photometric_transforms():
    return [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=0.5),
        transforms.RandomSolarize(threshold=128, p=0.2),
    ]


def _view_transform(size: int, scale: tuple[float, float], patch_size: int, mask_ratio: float):
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((size, size), scale=scale),
        *_photometric_transforms(),
        transforms.ToImage(**spt.data.static.ImageNet),
        *([transforms.PatchMasking(patch_size=patch_size, drop_ratio=mask_ratio)] if mask_ratio else []),
    )


def unpack_tokens(features, num_prefix_tokens: int = 1):
    if isinstance(features, dict):
        if "x_norm_clstoken" in features and "x_norm_patchtokens" in features:
            return features["x_norm_clstoken"], features["x_norm_patchtokens"]
        if "x" in features:
            features = features["x"]
        else:
            raise KeyError(f"Cannot unpack token dict keys={list(features.keys())}")
    if isinstance(features, (tuple, list)):
        features = features[0]
    cls = features[:, 0]
    patches = features[:, num_prefix_tokens:]
    return cls, patches


def teacher_distill_loss(module, model: LeJEPA, teacher, distill_input: torch.Tensor):
    bs = distill_input.shape[0]
    first_param = next(teacher.parameters())
    if first_param.device != distill_input.device:
        teacher.to(distill_input.device)
    teacher.eval()

    with torch.no_grad():
        t_out = teacher.forward_features(distill_input)
        t_cls, t_patches = unpack_tokens(t_out, getattr(teacher, "num_prefix_tokens", 1))

    s_out = model.backbone.forward_features(distill_input)
    s_cls, s_patches = unpack_tokens(s_out, getattr(model.backbone, "num_prefix_tokens", 1))

    aligned_cls = module.distill_align_head(s_cls.float())
    aligned_patches = module.distill_patch_align(s_patches.float())

    ps, pt = s_patches.shape[1], t_patches.shape[1]
    s_h, t_h = int(math.sqrt(ps)), int(math.sqrt(pt))
    if s_h * s_h != ps or t_h * t_h != pt:
        raise ValueError(f"Patch tokens must form square grids, got student={ps}, teacher={pt}")
    ap_2d = aligned_patches.reshape(bs, s_h, s_h, -1).permute(0, 3, 1, 2)
    ap_interp = F.interpolate(ap_2d, size=(t_h, t_h), mode="bilinear", align_corners=False)
    ap_interp = ap_interp.permute(0, 2, 3, 1).reshape(bs, pt, -1)

    cls_loss = F.mse_loss(aligned_cls, t_cls.float())

    s_p = F.normalize(ap_interp, dim=-1)
    t_p = F.normalize(t_patches.float(), dim=-1)
    patch_loss = 1.0 - (s_p * t_p).sum(dim=-1).mean()

    s_norm = F.normalize(aligned_cls, dim=-1)
    t_norm = F.normalize(t_cls.float(), dim=-1)
    rkd_loss = F.kl_div(
        F.log_softmax(s_norm @ s_norm.T / module.rkd_tau, dim=-1),
        F.softmax(t_norm @ t_norm.T / module.rkd_tau, dim=-1),
        reduction="batchmean",
    )

    s_gram = torch.bmm(s_p, s_p.transpose(1, 2))
    t_gram = torch.bmm(t_p, t_p.transpose(1, 2))
    gram_loss = F.mse_loss(s_gram, t_gram)

    w = module.distill_weights
    loss = w["cls"] * cls_loss + w["patch"] * patch_loss + w["rkd"] * rkd_loss + w["gram"] * gram_loss
    return loss, {
        "cls_loss": cls_loss,
        "patch_loss": patch_loss,
        "rkd_loss": rkd_loss,
        "gram_loss": gram_loss,
    }


def build_loader(args):
    views = {
        **{f"global_{i}": _view_transform(args.global_size, (0.3, 1.0), args.patch_size, 0.0)
           for i in range(args.global_views)},
        **{f"local_{i}": _view_transform(args.local_size, (0.05, 0.3), args.patch_size, 0.0)
           for i in range(args.local_views)},
    }
    dataset = spt.data.datasets.FromTorchDataset(
        dataset=ImageFolder(Path(args.data_dir) / "train"),
        names=["image", "label"],
        transform=transforms.MultiViewTransform(views),
    )
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=args.accelerator == "gpu",
        persistent_workers=args.num_workers > 0,
    )


def build_teacher(args):
    teacher = timm.create_model(
        args.teacher_timm_model, pretrained=not args.teacher_weights,
        num_classes=0, img_size=args.global_size,
    )
    if args.teacher_weights:
        teacher.load_state_dict(state_dict(load_checkpoint(args.teacher_weights)), strict=True)
    return teacher.eval().requires_grad_(False)


class TeJEPA(pl.LightningModule):
    def __init__(self, args, model, teacher=None):
        super().__init__()
        self.args = args
        self.model = model
        self.save_hyperparameters(vars(args))
        # Exclude the frozen teacher from student checkpoints and the optimizer.
        self.__dict__["distill_teacher"] = teacher
        if teacher is not None:
            self.distill_align_head = AlignmentHead(model.embed_dim, teacher.num_features)
            self.distill_patch_align = PatchAlignmentHead(model.embed_dim, teacher.num_features)
        self.distill_weights = {"cls": 1.0, "patch": 0.5, "rkd": 1.0, "gram": 0.3}
        self.rkd_tau = 0.1

    def training_step(self, batch, batch_idx):
        globals_ = [batch[f"global_{i}"]["image"] for i in range(self.args.global_views)]
        locals_ = [batch[f"local_{i}"]["image"] for i in range(self.args.local_views)]
        global_features = self.model.backbone(torch.cat(globals_))
        local_features = self.model.backbone(torch.cat(locals_)) if locals_ else None
        features = torch.cat([global_features, local_features]) if locals_ else global_features
        projected = self.model.projector(features)
        projected = projected.view(len(globals_) + len(locals_), globals_[0].size(0), -1)
        centers = projected[:len(globals_)].mean(0)
        prediction = (centers.unsqueeze(0) - projected).square().mean()
        sigreg = self.model.sigreg(projected.flatten(0, 1))
        loss = self.args.inv_weight * prediction + self.args.lamb_sigreg * sigreg
        terms = {"prediction": prediction, "sigreg": sigreg}
        teacher = self.__dict__["distill_teacher"]
        if teacher is not None:
            distillation, components = teacher_distill_loss(self, self.model, teacher, globals_[0])
            loss = loss + self.args.beta * distillation
            terms.update(distillation=distillation, **components)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        terms["loss"] = loss
        self.log_dict({f"train/{k}": v for k, v in terms.items()},
                      on_step=True, on_epoch=True, sync_dist=True,
                      batch_size=globals_[0].size(0))
        return loss

    def configure_optimizers(self):
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = round(total_steps * self.args.warmup_epochs / self.args.epochs)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr,
                                      weight_decay=self.args.weight_decay, betas=(0.9, 0.999))

        def multiplier(step):
            if warmup_steps and step < warmup_steps:
                return 0.01 + 0.99 * step / warmup_steps
            fraction = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
            return 0.001 + 0.999 * 0.5 * (1 + math.cos(math.pi * fraction))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def parse_args(stage):
    parser = argparse.ArgumentParser(description=f"TeJEPA {stage} training")
    parser.add_argument("--data_dir", required=True, help="ImageNet root containing train/<class>/")
    parser.add_argument("--output", default=f"outputs/{stage}")
    parser.add_argument("--resume", default="", help="Resume a complete checkpoint from this training stage")
    parser.add_argument("--student_encoder", default="vit_small_patch16_224")
    parser.add_argument("--global_size", type=int, default=224)
    parser.add_argument("--local_size", type=int, default=96)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--global_views", type=int, default=2)
    parser.add_argument("--local_views", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accelerator", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=float, default=5.0)
    parser.add_argument("--drop_path_rate", type=float, default=0.1)
    parser.add_argument("--lamb_sigreg", type=float, default=0.05)
    parser.add_argument("--inv_weight", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    if stage == "distill":
        parser.add_argument("--init_checkpoint", default="", help="Stage-1 LeJEPA checkpoint")
        parser.add_argument("--teacher_timm_model", default="vit_small_patch16_dinov3.lvd1689m")
        parser.add_argument("--teacher_weights", default="", help="Local timm-compatible weights; otherwise use pretrained timm weights")
        parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()
    args.stage = stage
    if min(args.epochs, args.batch_size, args.accum, args.devices, args.global_views, args.patch_size) <= 0:
        parser.error("Epochs, batch size, accumulation, devices, global views and patch size must be positive")
    if args.num_workers < 0 or args.local_views < 0:
        parser.error("Worker and local-view counts must be nonnegative")
    if args.global_size < args.patch_size or args.global_size % args.patch_size:
        parser.error("global_size must be a positive multiple of patch_size")
    if args.local_views and (args.local_size < args.patch_size or args.local_size % args.patch_size):
        parser.error("local_size must be a positive multiple of patch_size")
    if not 0 <= args.warmup_epochs < args.epochs or not 0 <= args.drop_path_rate < 1:
        parser.error("Require 0 <= warmup_epochs < epochs and 0 <= drop_path_rate < 1")
    if min(args.lr, args.lamb_sigreg, args.inv_weight) <= 0 or args.weight_decay < 0:
        parser.error("Learning rate and objective weights must be positive; weight decay must be nonnegative")
    if stage == "distill" and (args.beta <= 0 or not (args.init_checkpoint or args.resume)):
        parser.error("Distillation requires a positive beta and --init_checkpoint or --resume")
    if stage == "distill" and args.init_checkpoint and args.resume:
        parser.error("Choose initialization or full training resume")
    return args


def main(stage="distill"):
    args = parse_args(stage)
    configure_runtime()
    pl.seed_everything(args.seed, workers=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.ckpt").exists() and not args.resume:
        raise FileExistsError("Output already contains last.ckpt; use --resume or a new output directory")
    if args.resume:
        saved = load_checkpoint(args.resume)
        previous = saved.get("hyper_parameters", {})
        operational = {"output", "resume", "init_checkpoint", "num_workers", "accelerator", "precision"}
        changed = {k for k in set(previous) | set(vars(args))
                   if k not in operational and previous.get(k) != vars(args).get(k)}
        if changed or not saved.get("optimizer_states") or not saved.get("lr_schedulers"):
            raise ValueError(f"Resume requires the same model, objective and schedule; changed={sorted(changed)}")
        if saved.get("epoch", -1) >= args.epochs:
            raise ValueError("Checkpoint has already completed the requested training budget")
        del saved
    loader = build_loader(args)
    if len(loader) == 0 or math.ceil(len(loader.dataset) / args.devices) < args.batch_size:
        raise ValueError("Dataset is too small for one complete batch per device")
    model = LeJEPA(encoder_name=args.student_encoder, lamb=args.lamb_sigreg,
                   n_slices=1000, n_points=17, drop_path_rate=args.drop_path_rate)
    # Keep the estimator per rank, independent of process-launch timing.
    model.sigreg._is_ddp = False
    model.sigreg.ep._is_ddp = False
    model.sigreg.ep.world_size = 1
    if stage == "distill" and args.init_checkpoint:
        load_student_weights(model, args.init_checkpoint)
    teacher = build_teacher(args) if stage == "distill" else None
    module = TeJEPA(args, model, teacher)
    checkpoint = pl.pytorch.callbacks.ModelCheckpoint(
        dirpath=output, save_last=True, save_top_k=0, every_n_epochs=1,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator=args.accelerator, devices=args.devices,
        precision=args.precision,
        strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
        accumulate_grad_batches=args.accum, gradient_clip_val=1.0,
        callbacks=[checkpoint, pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step")],
        logger=pl.pytorch.loggers.CSVLogger(str(output), name="metrics"),
        default_root_dir=output, log_every_n_steps=20,
        num_sanity_val_steps=0, enable_model_summary=False,
    )
    trainer.fit(module, train_dataloaders=loader, ckpt_path=args.resume or None, weights_only=False)


if __name__ == "__main__":
    main()
