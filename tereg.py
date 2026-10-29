"""Teacher-anchored joint characteristic-function regularization.

Fixed full-dimensional coordinates are fitted to the current random student and
frozen teacher on training images only. No student pretraining asset is used.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _paired(student, teacher):
    if student.ndim != 2 or student.shape != teacher.shape or not len(student):
        raise ValueError("Expected nonempty paired N x D features of equal dimension")
    return teacher.detach()


def gaussian_kernel(x, y, bandwidth):
    if not math.isfinite(float(bandwidth)) or bandwidth <= 0:
        raise ValueError("Bandwidth must be positive and finite")
    distances = (x.square().sum(1)[:, None] + y.square().sum(1)[None, :]
                 - 2 * x @ y.T).clamp_min(0)
    return torch.exp(-distances / (2 * bandwidth ** 2))


def joint_kernel_reference(student, teacher, feature_bandwidth, anchor_bandwidth):
    """Exact, O(N^2 D), paired V-statistic; no factor N or adaptive batch scale.

    MMD^2(law(S,T), law(T,T)) with a product of two Gaussian kernels.
    Teacher tensors are stopped, including their use in the anchor kernel.
    Small negative roundoff is left visible rather than clipping the objective.
    """
    teacher = _paired(student, teacher)
    anchor = gaussian_kernel(teacher, teacher, anchor_bandwidth)
    ss = gaussian_kernel(student, student, feature_bandwidth)
    tt = gaussian_kernel(teacher, teacher, feature_bandwidth)
    st = gaussian_kernel(student, teacher, feature_bandwidth)
    return (anchor * (ss + tt - st - st.T)).mean()


def joint_characteristic_sketch(student, teacher, omega, eta):
    """O(N D K) Fourier estimator of the exact joint kernel reference.

    omega ~ N(0, I / feature_bandwidth^2), eta independently sampled as
    N(0, I / anchor_bandwidth^2). Both are D x K and shared for S and T.
    Written in real arithmetic for accelerator portability. Frequencies must
    be resampled across training steps; finite fixed sketches are not identifying.
    """
    teacher = _paired(student, teacher)
    if omega.ndim != 2 or omega.shape != eta.shape or omega.size(0) != student.size(1):
        raise ValueError("Expected equal D x K frequency matrices")
    if omega.size(1) == 0:
        raise ValueError("At least one frequency is required")
    omega, eta = omega.detach(), eta.detach()
    a, b, anchor = student @ omega, teacher @ omega, teacher @ eta
    dc, ds = a.cos() - b.cos(), a.sin() - b.sin()
    real = (anchor.cos() * dc - anchor.sin() * ds).mean(0)
    imag = (anchor.sin() * dc + anchor.cos() * ds).mean(0)
    return (real.square() + imag.square()).mean()


class FixedTeacherCoordinates(nn.Module):
    """Frozen full-dimensional rotation and scalar units fitted on training pairs.

    Fixed calibration RMS rescales all directions equally in each model. No whitening, PCA
    truncation, student batch normalization, or trainable MLP can hide rank loss.
    This equal-dimensional construction is deliberately specific to our ViT-S
    teacher/student pair. Fit once at the current random student initialization.
    """

    def __init__(self, rotation, student_mean, teacher_mean, scale, student_scale=None):
        super().__init__()
        dim = student_mean.numel()
        rotation = rotation.detach().clone()
        if rotation.shape != (dim, dim) or teacher_mean.shape != student_mean.shape:
            raise ValueError("Coordinates must have equal full dimension")
        identity = torch.eye(dim, dtype=rotation.dtype, device=rotation.device)
        if not torch.allclose(rotation.T @ rotation, identity, atol=1e-5, rtol=1e-5):
            raise ValueError("The frozen readout must be orthogonal")
        scale = torch.as_tensor(scale, dtype=rotation.dtype, device=rotation.device)
        if scale.ndim != 0 or not torch.isfinite(scale) or scale <= 0:
            raise ValueError("The fixed teacher scale must be positive and finite")
        student_scale = scale if student_scale is None else torch.as_tensor(
            student_scale, dtype=rotation.dtype, device=rotation.device)
        if student_scale.ndim != 0 or not torch.isfinite(student_scale) or student_scale <= 0:
            raise ValueError("The fixed initial student scale must be positive and finite")
        for name, value in dict(rotation=rotation, student_mean=student_mean,
                                teacher_mean=teacher_mean, scale=scale, student_scale=student_scale).items():
            self.register_buffer(name, value.detach().clone())

    @classmethod
    @torch.no_grad()
    def fit(cls, student, teacher):
        teacher = _paired(student, teacher)
        # Tiny calibration factorization on CPU; never in an encoder train step.
        s, t = student.detach().double().cpu(), teacher.double().cpu()
        sm, tm = s.mean(0), t.mean(0)
        s, t = s - sm, t - tm
        u, _, vh = torch.linalg.svd(s.T @ t, full_matrices=False)
        return cls(u @ vh, sm, tm, t.square().mean().sqrt(), s.square().mean().sqrt())

    def forward(self, student, teacher):
        teacher = _paired(student, teacher)
        aligned = ((student - self.student_mean) @ self.rotation) / self.student_scale
        target = (teacher - self.teacher_mean) / self.scale
        return aligned, target


class TeReg(nn.Module):
    """Checkpointed random-frequency estimator on paired global backbone features."""

    def __init__(self, dimension, sketches=1024):
        super().__init__()
        self.coordinates = FixedTeacherCoordinates(torch.eye(dimension),
            torch.zeros(dimension), torch.zeros(dimension), torch.tensor(1.))
        self.sketches = int(sketches)
        if self.sketches < 1:
            raise ValueError('Positive sketch count required')
        self.register_buffer('feature_bandwidth', torch.tensor(1.))
        self.register_buffer('anchor_bandwidth', torch.tensor(1.))
        self.register_buffer('frequency_step', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def initialize(self, student, teacher):
        fitted = FixedTeacherCoordinates.fit(student, teacher).float()
        self.coordinates.load_state_dict(fitted.state_dict())
        target = (teacher.detach().float().cpu()-fitted.teacher_mean)/fitted.scale
        generator = torch.Generator().manual_seed(9173)
        indices = torch.randint(len(target), (2,8192), generator=generator)
        valid = indices[0] != indices[1]
        distance = (target[indices[0,valid]]-target[indices[1,valid]]).square().sum(1)
        bandwidth = distance.median().sqrt()
        if not torch.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError('Degenerate teacher calibration features')
        self.feature_bandwidth.copy_(bandwidth)
        self.anchor_bandwidth.copy_(bandwidth)
        return dict(images=len(student), dimension=student.size(1),
                    student_scale=float(fitted.student_scale), teacher_scale=float(fitted.scale),
                    bandwidth=float(bandwidth))

    def forward(self, student, teacher, distributed=True):
        # Same seed on all ranks, new frequencies every step, preserved on resume.
        generator = torch.Generator().manual_seed(19771 + int(self.frequency_step))
        dim = student.size(1)
        omega = torch.randn(dim, self.sketches, generator=generator)
        eta = torch.randn(dim, self.sketches, generator=generator)
        if self.training:
            self.frequency_step.add_(1)
        s, t = self.coordinates(student.float(), teacher.detach().float())
        if distributed:
            from distributed import gather
            s, t = gather(s), gather(t)
        omega = omega.to(s.device) / self.feature_bandwidth
        eta = eta.to(s.device) / self.anchor_bandwidth
        loss = joint_characteristic_sketch(s, t, omega, eta)
        with torch.no_grad():
            diagnostics = dict(raw=loss.detach(), fixed_readout_mse=(s-t).square().mean(),
                               images=s.new_tensor(len(s)),
                               student_trace=s.var(0, unbiased=False).sum(),
                               teacher_trace=t.var(0, unbiased=False).sum())
        return loss, diagnostics
