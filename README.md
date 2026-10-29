# TeJEPA: one-stage teacher-guided JEPA with TeReg

A randomly initialized student learns from the JEPA multi-view prediction objective
and a frozen visual teacher in one training stage. The recipe retains **two global
crops, six local crops, and prediction**. **TeReg replaces SIGReg** as the
regularizer; it is not an additional regularizer alongside SIGReg. Distillation
uses CLS, RKD, teacher-reliability-weighted patches, and spectrally compressed Gram matching.

$$
L=0.95L_{\mathrm{prediction}}
+L_{\mathrm{CLS}}+L_{\mathrm{RKD}}
+\lambda_pL_{\mathrm{patch}}^{\mathrm{new}}
+\lambda_gL_{\mathrm{Gram}}^{\mathrm{new}}+\lambda_TL_{\mathrm{TeReg}}.
$$

**[完整方法说明（中文）](METHOD.md)** includes the derivation, calibration,
implementation details, and precise scope of the Lean proofs.

No student checkpoint or preliminary pretraining stage is required. The teacher
must have pretrained weights. This one-stage protocol has **no completed
ImageNet accuracy result in this branch**; functional tests do not establish
representation quality or prevent every possible training collapse.

## Install

Use Python 3.12, install PyTorch/torchvision for your device, then:

```bash
pip install -r requirements.txt
```

## Data

Training uses `ImageFolder`; labels do not enter the training loss.

```text
imagenet/
  train/<class>/*.JPEG
  val/<class>/*.JPEG
```

Frozen evaluation also accepts flat `val/` images with `meta/val.txt` entries
`filename zero_based_label`. Class indices must follow sorted training directories.

## Train

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --data_dir /path/to/imagenet \
  --teacher_weights /path/to/teacher/model.safetensors \
  --output outputs/tereg \
  --epochs 40 --batch_size 64 --seed 0
```

The defaults are a random ViT-S/16 student and frozen DINOv3 ViT-S/16 teacher.
Full-dimensional TeReg requires equal student/teacher feature dimensions. Training
uses AdamW, five warmup epochs, and cosine decay over **40 epochs**. The effective
batch above is 512; RKD retains per-rank statistics, while TeReg collects a
differentiable global batch. There is no separate Gaussian SIGReg term, module,
or weight.

Startup first fits frozen coordinates on training images, then measures initial
backbone gradients to set patch/Gram scales and a TeReg gradient budget. It makes
**zero optimizer updates** during calibration. TeReg and teacher distillation
are active from the **first optimizer step**, with no Stage1 or later switch.
All fitted buffers and frequency counters are saved in the checkpoint;
calibration is skipped on resume.

Only rolling `last.ckpt` is retained. Resume with the same recipe and epoch budget:

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --data_dir /path/to/imagenet \
  --teacher_weights /path/to/teacher/model.safetensors \
  --output outputs/tereg --epochs 40 --batch_size 64 --seed 0 \
  --resume outputs/tereg/last.ckpt
```

For a single device, use `python train.py ...`. For CPU, use `--device cpu
--precision fp32`. Non-finite losses/gradients stop training before an optimizer
update. Checkpoints exclude the full teacher model. Logs contain loss components,
overlap statistics, and backbone geometry diagnostics; there is no automatic
accuracy-based stopping rule in this compact implementation.

## Frozen ImageNet evaluation

```bash
python eval_imagenet.py \
  --data_dir /path/to/imagenet \
  --checkpoint outputs/tereg/last.ckpt \
  --output outputs/imagenet
```

This freezes the encoder, extracts final CLS features, and fits a LayerNorm +
Linear classifier for 30 epochs. Learning rate is fixed at 0.001; validation is
not used to select the classifier epoch. Features remain in host memory rather
than being saved as large caches. A saved classifier can be reused with `--head`;
the checkpoint hash and class mapping must match.

## Checks

```bash
python -m unittest discover -s tests -v
```

Tests cover local prediction gradients, absence of SIGReg, TeReg activation from
the first training batch, crop/flip correspondence,
spectral derivatives, TeReg's joint target and distributed gradients, calibration
without optimizer updates, full two-process training, resume, and frozen evaluation.
Integration tests use tiny models and synthetic images, not accuracy benchmarks.
Optional Lean properties are in `formal/`; see [METHOD.md](METHOD.md#形式化验证的范围).

## Files

| File | Role |
| --- | --- |
| `train.py` | Fresh one-stage training, calibration, checkpoint/resume |
| `models.py` | Multi-view prediction, improved teacher losses, and TeReg |
| `tereg.py` | Fixed backbone coordinates and joint characteristic matching |
| `losses.py` | Reliable patch matching and spectral Gram compression |
| `data.py` | Crop geometry, local views, calibration/evaluation data |
| `distributed.py` | Differentiable global feature gathering |
| `eval_imagenet.py` | Frozen encoder evaluation |

## Foundations

Built on [LeJEPA](https://arxiv.org/abs/2511.08544),
[stable-pretraining](https://github.com/galilai-group/stable-pretraining),
[timm](https://github.com/huggingface/pytorch-image-models), and
[DINOv3](https://github.com/facebookresearch/dinov3).
Distribution matching uses the established [MMD framework](https://jmlr.org/papers/v13/gretton12a.html).
