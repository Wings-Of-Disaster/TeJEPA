"""Frozen ImageNet evaluation with a LayerNorm + Linear classifier."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data import evaluation_datasets
from models import load_backbone


@torch.no_grad()
def extract_features(backbone, loader, device):
    backbone.eval()
    features, labels = [], []
    for images, target in tqdm(loader, desc="Extract features"):
        embedding = backbone(images.to(device, non_blocking=True))
        if embedding.ndim != 2 or not torch.isfinite(embedding).all():
            raise ValueError("Expected finite, two-dimensional encoder features")
        features.append(embedding.float().cpu())
        labels.append(target.cpu())
    if not features:
        raise ValueError("Empty feature loader")
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def accuracy(head, loader, device):
    head.eval()
    correct1 = correct5 = count = 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = head(features)
        ranking = logits.topk(min(5, logits.size(1)), dim=1).indices
        correct1 += (ranking[:, 0] == labels).sum().item()
        correct5 += (ranking == labels[:, None]).any(dim=1).sum().item()
        count += labels.numel()
    if not count:
        raise ValueError("Empty validation loader")
    return {"top1": 100 * correct1 / count, "top5": 100 * correct5 / count, "samples": count}


def fit_probe(train_features, train_labels, validation_features, validation_labels,
              num_classes, args, device):
    torch.manual_seed(args.seed)
    head = nn.Sequential(nn.LayerNorm(train_features.size(1)),
                         nn.Linear(train_features.size(1), num_classes)).to(device)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels), batch_size=args.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        TensorDataset(validation_features, validation_labels), batch_size=args.batch_size,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader),
    )
    history = []
    for epoch in range(args.epochs):
        head.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            loss = F.cross_entropy(head(features), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite probe loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
        metrics = accuracy(head, validation_loader, device)
        history.append({"epoch": epoch + 1, **metrics})
        print(f"Epoch {epoch + 1}/{args.epochs}: top1={metrics['top1']:.3f} top5={metrics['top5']:.3f}")
    return head, history


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", default="vit_small_patch16_224")
    parser.add_argument("--output", default="outputs/imagenet")
    parser.add_argument("--head", default="", help="Evaluate an existing classifier without fitting it again")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--feature_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.feature_batch_size) <= 0 or args.num_workers < 0:
        parser.error("Epochs and batch sizes must be positive; worker count must be nonnegative")
    if args.lr <= 0 or args.weight_decay < 0:
        parser.error("Learning rate must be positive; weight decay must be nonnegative")
    return args


def main():
    args = parse_args()
    output = Path(args.output)
    if (output / "metrics.json").exists() or (output / "linear_head.pt").exists():
        raise FileExistsError("Choose a new output directory to preserve existing evaluation results")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    backbone = load_backbone(args.encoder, args.checkpoint).to(device)
    train, validation = evaluation_datasets(args.data_dir)
    num_classes = len(train.classes)
    loader_options = dict(batch_size=args.feature_batch_size, num_workers=args.num_workers,
                          pin_memory=device.type == "cuda", shuffle=False)
    validation_features, validation_labels = extract_features(
        backbone, DataLoader(validation, **loader_options), device,
    )
    if args.head:
        saved = torch.load(args.head, map_location="cpu", weights_only=True)
        if saved["classes"] != train.classes or saved["encoder"] != args.encoder:
            raise ValueError("Saved classifier class mapping or encoder differs")
        # The classifier is meaningful only for the exact encoder weights it was fitted on.
        if saved["encoder_sha256"] != checkpoint_digest(args.checkpoint):
            raise ValueError("Saved classifier belongs to a different encoder checkpoint")
        head = nn.Sequential(nn.LayerNorm(validation_features.size(1)),
                             nn.Linear(validation_features.size(1), num_classes)).to(device)
        head.load_state_dict(saved["probe"], strict=True)
        history = []
    else:
        train_features, train_labels = extract_features(
            backbone, DataLoader(train, **loader_options), device,
        )
        # The encoder is no longer needed; only the classifier has gradients.
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()
        head, history = fit_probe(train_features, train_labels, validation_features,
                                  validation_labels, num_classes, args, device)
    metrics = accuracy(head, DataLoader(TensorDataset(validation_features, validation_labels),
                                       batch_size=args.batch_size), device)
    protocol = {"encoder": args.encoder, "epochs": args.epochs, "seed": args.seed,
                "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
                "features": "frozen final-layer CLS", "classifier": "LayerNorm + Linear",
                "preprocessing": "bicubic resize 256, center crop 224, ImageNet normalization",
                "selection": "final epoch", "classifier_reused": bool(args.head)}
    if args.head:
        protocol = {**saved["protocol"], "classifier_reused": True}
    torch.save({"probe": head.cpu().state_dict(), "encoder": args.encoder,
                "encoder_sha256": checkpoint_digest(args.checkpoint),
                "classes": train.classes, "protocol": protocol}, output / "linear_head.pt")
    (output / "metrics.json").write_text(json.dumps(
        {"validation": metrics, "protocol": protocol, "history": history}, indent=2,
    ) + "\n")
    print(json.dumps(metrics))


def checkpoint_digest(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
