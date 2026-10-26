"""Strict checkpoint loading for LeJEPA and timm encoders."""

from collections.abc import Mapping

import timm
import torch


def load_checkpoint(path):
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(str(path))
    # Lightning checkpoints also contain optimizer and training state.
    # Only load checkpoint files from a trusted source.
    return torch.load(path, map_location="cpu", weights_only=False)


def state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected a checkpoint mapping")
    for key in ("state_dict", "model"):
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise ValueError("Checkpoint has no model state")
    return dict(checkpoint)


def strip_prefix(state, prefix):
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def load_student_weights(model, path):
    state = state_dict(load_checkpoint(path))
    for prefix in ("model.", "student."):
        if any(key.startswith(prefix + "backbone.") for key in state):
            state = strip_prefix(state, prefix)
            break
    model.load_state_dict(state, strict=True)


def load_backbone(encoder, path):
    backbone = timm.create_model(encoder, pretrained=False, num_classes=0)
    state = state_dict(load_checkpoint(path))
    for prefix in ("model.backbone.", "student.backbone.", "backbone."):
        if any(key.startswith(prefix) for key in state):
            state = strip_prefix(state, prefix)
            break
    backbone.load_state_dict(state, strict=True)
    backbone.eval().requires_grad_(False)
    return backbone
