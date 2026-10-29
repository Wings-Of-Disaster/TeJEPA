"""Check TeReg's proposed mathematics on CPU, independently of encoder training."""
import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tereg import (FixedTeacherCoordinates, TeReg, gaussian_kernel,
                         joint_characteristic_sketch, joint_kernel_reference)


def runtime_asset():
    return 4


def ddp_worker(rank, path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+path, rank=rank, world_size=2)
    try:
        torch.manual_seed(19)
        model = torch.nn.Linear(4,4)
        reference = copy.deepcopy(model)
        parallel = DistributedDataParallel(model)
        x, t = torch.randn(16,4), torch.randn(16,4)
        region = slice(rank*8,(rank+1)*8)
        loss,_ = TeReg(runtime_asset(),32)(parallel(x[region]),t[region])
        loss.backward()
        expected,_ = TeReg(runtime_asset(),32)(reference(x),t,distributed=False)
        expected.backward()
        for p,q in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5)
    finally:
        dist.destroy_process_group()


class TeRegTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.rng = torch.Generator().manual_seed(731)
        # Intentionally non-isotropic, shifted and non-Gaussian teacher.
        self.t = torch.randn(24, 5, generator=self.rng, dtype=torch.float64)
        self.t[:, 0] = self.t[:, 0].square() * 3 + 7
        self.t[:, 1] *= .1
        self.omega = torch.randn(5, 8192, generator=self.rng, dtype=torch.float64) / 3
        self.eta = torch.randn(5, 8192, generator=self.rng, dtype=torch.float64) / 3

    def test_non_gaussian_teacher_is_exact_zero_target_and_stationary(self):
        s = self.t.clone().requires_grad_()
        value = joint_characteristic_sketch(s, self.t, self.omega, self.eta)
        self.assertEqual(float(value.detach()), 0.)
        value.backward()
        self.assertEqual(float(s.grad.norm()), 0.)
        torch.testing.assert_close(joint_kernel_reference(s, self.t, 3., 3.),
                                   torch.zeros((), dtype=s.dtype), atol=1e-15, rtol=0)

    def test_teacher_permutation_fools_marginal_but_not_joint_matching(self):
        s = self.t.roll(1, 0)
        marginal = joint_characteristic_sketch(s, self.t, self.omega, self.eta * 0)
        joint = joint_characteristic_sketch(s, self.t, self.omega, self.eta)
        self.assertLess(float(marginal), 1e-25)
        self.assertGreater(float(joint), .001)

    def test_collapse_has_nonzero_loss_and_feature_gradient(self):
        s = self.t.mean(0).repeat(len(self.t), 1).requires_grad_()
        loss = joint_kernel_reference(s, self.t, 3., 3.)
        loss.backward()
        self.assertGreater(float(loss.detach()), .001)
        self.assertGreater(float(s.grad.norm()), .001)

    def test_fourier_formula_matches_exact_kernel_in_value_and_gradient(self):
        s = (self.t + .7 * torch.randn(self.t.shape, generator=self.rng,
                                      dtype=self.t.dtype)).requires_grad_()
        exact = joint_kernel_reference(s, self.t, 3., 3.)
        sketch = joint_characteristic_sketch(s, self.t, self.omega, self.eta)
        g_exact, = torch.autograd.grad(exact, s)
        g_sketch, = torch.autograd.grad(sketch, s)
        torch.testing.assert_close(sketch, exact, atol=.002, rtol=.1)
        self.assertGreater(float(torch.nn.functional.cosine_similarity(
            g_exact.flatten(), g_sketch.flatten(), dim=0)), .98)

    def test_no_sample_count_multiplier(self):
        s = self.t.roll(1, 0).clone().requires_grad_()
        a = joint_characteristic_sketch(s, self.t, self.omega, self.eta)
        b = joint_characteristic_sketch(s.repeat(3, 1), self.t.repeat(3, 1),
                                         self.omega, self.eta)
        torch.testing.assert_close(a, b, atol=1e-15, rtol=1e-12)
        ga, = torch.autograd.grad(a, s)
        gb, = torch.autograd.grad(b, s)
        torch.testing.assert_close(ga, gb, atol=1e-15, rtol=1e-12)

    def test_teacher_and_frequencies_receive_no_gradient(self):
        t = self.t.clone().requires_grad_()
        s = self.t.roll(1, 0).clone().requires_grad_()
        omega = self.omega.clone().requires_grad_()
        eta = self.eta.clone().requires_grad_()
        joint_characteristic_sketch(s, t, omega, eta).backward()
        self.assertIsNone(t.grad)
        self.assertIsNone(omega.grad)
        self.assertIsNone(eta.grad)
        self.assertGreater(float(s.grad.norm()), 0.)

    def test_exact_gradient_bound(self):
        s = self.t.roll(1, 0).clone().requires_grad_()
        value = joint_kernel_reference(s, self.t, 3., 3.)
        value.backward()
        bound = 4 / (len(s) * 3 * math.sqrt(math.e))
        self.assertLessEqual(float(s.grad.norm(dim=1).max()), bound + 1e-12)

    def test_analytic_independent_gaussian_noise_cannot_satisfy_target(self):
        # Exact population MMD for T~N(0,diag(3,.2)); S=T+epsilon.
        cov = torch.diag(torch.tensor([3., .2], dtype=torch.float64))
        q = torch.cat([torch.cat([cov, cov], 1), torch.cat([cov, cov], 1)], 0)
        eye = torch.eye(4, dtype=torch.float64)
        values = []
        for sigma in [0., .1, .3, 1.]:
            p = q.clone(); p[:2, :2] += sigma ** 2 * torch.eye(2)
            # Both kernel bandwidths 1; Gaussian expectation uses a determinant.
            value = (torch.linalg.det(eye + 2 * p).rsqrt()
                     + torch.linalg.det(eye + 2 * q).rsqrt()
                     - 2 * torch.linalg.det(eye + p + q).rsqrt())
            values.append(float(value))
        self.assertAlmostEqual(values[0], 0., places=14)
        self.assertTrue(all(a < b for a, b in zip(values, values[1:])))

    def test_fixed_coordinates_preserve_full_backbone_geometry(self):
        rotation, _ = torch.linalg.qr(torch.randn(5, 5, generator=self.rng,
                                                dtype=torch.float64))
        s = self.t @ rotation + 3
        coords = FixedTeacherCoordinates.fit(s, self.t)
        aligned, target = coords(s, self.t)
        torch.testing.assert_close(aligned, target, atol=1e-12, rtol=1e-12)
        centered = s - s.mean(0)
        torch.testing.assert_close(aligned @ aligned.T,
                                   centered @ centered.T / coords.scale.square(),
                                   atol=1e-12, rtol=1e-12)
        self.assertEqual(len(list(coords.parameters())), 0)
        broken = centered.clone(); broken[:, 2:] = 0
        mapped, _ = coords(broken, self.t)
        self.assertEqual(int(torch.linalg.matrix_rank(mapped - mapped.mean(0))), 2)

    def test_exact_kernel_gradcheck(self):
        s = self.t[:4].roll(1, 0).clone().requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: joint_kernel_reference(x, self.t[:4], 3., 3.), (s,)))

    def test_runtime_frequency_resume_and_frozen_coordinates(self):
        x = self.t[:,:4].float().clone().requires_grad_()
        target = x.detach().roll(1,0)
        module = TeReg(runtime_asset(),32)
        module(x,target,distributed=False)
        restored = TeReg(runtime_asset(),32)
        restored.load_state_dict(module.state_dict())
        a,_ = module(x,target,distributed=False)
        b,_ = restored(x,target,distributed=False)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertEqual(int(module.frequency_step),2)
        self.assertEqual(len(list(module.parameters())),0)
        a.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_fixed_scales_remove_initial_global_amplitude_gap_without_rank_change(self):
        s = 5*self.t + 3
        coords = FixedTeacherCoordinates.fit(s, self.t)
        aligned, target = coords(s,self.t)
        torch.testing.assert_close(aligned,target,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(coords.student_scale,5*coords.scale,atol=1e-12,rtol=1e-12)
        fixed = coords.student_scale.clone()
        scaled,_ = coords(2*s-coords.student_mean,self.t)
        torch.testing.assert_close(coords.student_scale,fixed,atol=0,rtol=0)
        self.assertGreater(float((scaled-aligned).norm()),1.)

    def test_runtime_ddp_matches_full_batch_gradient(self):
        with tempfile.TemporaryDirectory(prefix='tereg-gloo-') as directory:
            mp.spawn(ddp_worker,args=(str(Path(directory)/'store'),),nprocs=2,join=True)



if __name__ == '__main__':
    unittest.main()
