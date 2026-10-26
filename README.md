# TeJEPA

Teacher-guided LeJEPA training and frozen ImageNet classification.

The training pipeline first learns a LeJEPA student, then adds semantic supervision
from a frozen visual teacher. Stage 2 retains multi-view prediction and SIGReg:

```text
L = 0.95 L_prediction + 0.05 L_SIGReg + beta L_distillation
L_distillation = L_CLS + 0.5 L_patch + L_RKD + 0.3 L_Gram
```

The default student is ViT-S/16; the default teacher is DINOv3 ViT-S/16.
The teacher remains frozen, and its weights are excluded from student checkpoints.
Inference and linear evaluation use the student encoder.

## Installation

Use Python 3.12. Install a matching PyTorch and torchvision pair for your device,
then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The implementation uses `stable-pretraining==0.1.8`, `timm==1.0.26`, and
`lightning==2.6.1`. Install the appropriate PyTorch build before the requirements
if your environment needs a specific accelerator build.

## ImageNet layout

```text
imagenet/
  train/
    n01440764/*.JPEG
    ...
  val/
    n01440764/*.JPEG
    ...
```

A flat `val/*.JPEG` directory is also supported when `meta/val.txt` contains
`filename zero_based_label` pairs. Those labels must follow the lexicographically
sorted training class-directory order used by `ImageFolder`.

## Training

Both stages default to **40 epochs**, with AdamW, five warmup epochs, and cosine
decay over the requested training budget. Training uses two global 224-pixel crops
and six local 96-pixel crops, without patch masking. SIGReg is computed over pooled
views within each rank. RKD uses each rank's first global-view batch, so changing
the per-device batch size changes that statistic even at the same effective batch.

Stage 1 pretrains the LeJEPA student:

```bash
python pretrain.py \
  --data_dir /path/to/imagenet \
  --output outputs/pretrain \
  --epochs 40 --devices 1
```

Stage 2 initializes that student and trains TeJEPA:

```bash
python train.py \
  --data_dir /path/to/imagenet \
  --init_checkpoint outputs/pretrain/last.ckpt \
  --teacher_weights /path/to/teacher/model.safetensors \
  --output outputs/tejepa \
  --epochs 40 --devices 1
```

An existing compatible LeJEPA checkpoint can be used directly for Stage 2.
Teacher weights must match `--teacher_timm_model`; omit `--teacher_weights` to use
timm's pretrained weights, subject to the model provider's access requirements.
For multiple devices, set `--devices` to the desired count; Lightning launches DDP.
The effective optimizer batch is `batch_size * accum * devices` (defaults: 64, 4, 1).

Only a rolling `last.ckpt` is kept, alongside CSV metrics. It contains the student,
alignment heads, optimizer, and scheduler state. Resume an interrupted Stage 2 run
with the same configuration and total epoch budget:

```bash
python train.py \
  --data_dir /path/to/imagenet \
  --resume outputs/tejepa/last.ckpt \
  --teacher_weights /path/to/teacher/model.safetensors \
  --output outputs/tejepa \
  --epochs 40 --devices 1
```

For CPU execution, add `--accelerator cpu --precision 32-true`. Changing the student
patch size also requires matching `--patch_size`, `--global_size`, and `--local_size`.

## ImageNet evaluation

Freeze the encoder, extract its final-layer CLS features once, and fit a
LayerNorm + Linear classifier for 30 epochs. Features stay in host memory; no
large feature-cache files are written. ViT-S ImageNet training features require
roughly 2 GB of host memory, plus validation features and extraction overhead.
The classifier uses AdamW (learning rate 0.001, weight decay 0.000001), batch size
256, and cosine decay. Reported top-1/top-5 accuracy uses the final classifier;
validation accuracy is not used to choose its epoch or hyperparameters.

```bash
python eval_imagenet.py \
  --data_dir /path/to/imagenet \
  --checkpoint outputs/tejepa/last.ckpt \
  --encoder vit_small_patch16_224 \
  --output outputs/imagenet
```

The output directory contains `metrics.json` and `linear_head.pt`. To evaluate an
existing classifier without fitting it again, pass `--head` and a fresh output
directory. The checkpoint hash, encoder name, and class mapping must match:

```bash
python eval_imagenet.py \
  --data_dir /path/to/imagenet \
  --checkpoint outputs/tejepa/last.ckpt \
  --head outputs/imagenet/linear_head.pt \
  --output outputs/imagenet_recheck
```

Use `--device cpu` for CPU evaluation. All paths are supplied by the user; datasets
and checkpoints are not included in this repository.

## Files

| File | Purpose |
| --- | --- |
| `pretrain.py` | Stage-1 LeJEPA entry point |
| `train.py` | Full TeJEPA training and shared training components |
| `eval_imagenet.py` | Frozen ImageNet linear evaluation |
| `data.py` | ImageNet validation loading and evaluation transforms |
| `models.py` | Checkpoint loading and frozen encoder construction |

## Acknowledgments

Built on [LeJEPA / stable-pretraining](https://github.com/galilai-group/stable-pretraining),
[timm](https://github.com/huggingface/pytorch-image-models), and
[DINOv3](https://github.com/facebookresearch/dinov3).
