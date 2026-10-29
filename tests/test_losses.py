import sys
from pathlib import Path
import unittest
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from losses import correspondence_grid,reliable_patch_loss,compressed_gram_loss,_StableGramRoot,gram_root_iteration


class FullLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(44)
        self.box=torch.tensor([[[0.,0.,1.,1.]],[[0.,0.,1.,1.]]])
        self.flip=torch.zeros(2,1,dtype=torch.bool)

    def test_same_crop_patch_centers_and_flip_are_exact(self):
        grid,valid=correspondence_grid(self.box[0],self.flip[0],self.box[1],self.flip[1],2,2)
        expected=torch.tensor([[[[-.5,-.5],[.5,-.5]],[[-.5,.5],[.5,.5]]]])
        torch.testing.assert_close(grid,expected);self.assertTrue(valid.all())
        flipped,_=correspondence_grid(self.box[0],self.flip[0],self.box[1],~self.flip[1],2,2)
        torch.testing.assert_close(flipped[...,0],-grid[...,0])

    def test_crop_shift_maps_to_original_coordinates_and_masks_edges(self):
        a=torch.tensor([[0.,0.,1.,1.]])
        b=torch.tensor([[0.,.5,1.,.5]])
        grid,mask=correspondence_grid(a,torch.tensor([False]),b,torch.tensor([False]),2,2)
        self.assertFalse(mask[0,:,0].any());self.assertTrue(mask[0,:,1].all())
        torch.testing.assert_close(grid[0,:,1,0],torch.zeros(2))

    def test_perfect_teacher_in_both_flipped_views_has_zero_error(self):
        t=torch.randn(1,2,2,5)
        pair=torch.stack([t.reshape(1,4,5),t.flip(2).reshape(1,4,5)])
        flips=self.flip.clone();flips[1]=True
        loss,info=reliable_patch_loss(pair.clone().requires_grad_(),pair,self.box,flips)
        self.assertLess(abs(float(loss.detach())),1e-6)
        self.assertGreater(float(info['reliability_mean']),.999)

    def test_teacher_weights_have_no_gradient_and_student_receives_one(self):
        t=torch.randn(2,1,4,5,requires_grad=True);s=torch.randn_like(t,requires_grad=True)
        loss,_=reliable_patch_loss(s,t,self.box,self.flip);loss.backward()
        self.assertIsNone(t.grad);self.assertGreater(s.grad.norm().item(),0)

    def test_disjoint_crops_use_finite_uniform_two_view_supervision(self):
        boxes=torch.tensor([[[0.,0.,.3,.3]],[[.7,.7,.3,.3]]])
        s=torch.randn(2,1,4,5);t=torch.randn_like(s)
        loss,info=reliable_patch_loss(s,t,boxes,self.flip)
        expected=(1-(F.normalize(s,dim=-1)*F.normalize(t,dim=-1)).sum(-1)).mean()
        torch.testing.assert_close(loss,expected)
        self.assertEqual(float(info['no_overlap_fraction']),1)

    def test_repeated_eigenvalue_frechet_gradient_matches_finite_difference(self):
        g=torch.eye(4,dtype=torch.float64).repeat(2,1,1).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda x:_StableGramRoot.apply(x,.01),(g,),fast_mode=True))

    def test_zero_eigenvalues_have_finite_gradient(self):
        g=torch.zeros(2,4,4,dtype=torch.float64,requires_grad=True)
        y=_StableGramRoot.apply(g,.01);y.sum().backward()
        self.assertTrue(torch.isfinite(g.grad).all())
        torch.testing.assert_close(g.grad,torch.full_like(g,5.))

    def test_spectral_root_satisfies_shifted_square_identity(self):
        x=torch.randn(3,7,4,dtype=torch.float64);g=x@x.transpose(-1,-2)
        root=_StableGramRoot.apply(g,.01)+.1*torch.eye(7,dtype=g.dtype)
        torch.testing.assert_close(root@root,g+.01*torch.eye(7,dtype=g.dtype),atol=1e-12,rtol=1e-12)

    def test_compressed_gram_zero_at_teacher_and_rotation_invariance(self):
        t=F.normalize(torch.randn(2,6,8,dtype=torch.float64),dim=-1)
        q,_=torch.linalg.qr(torch.randn(8,8,dtype=torch.float64))
        s=(t@q).requires_grad_();loss=compressed_gram_loss(s,t);loss.backward()
        self.assertLess(float(loss.detach()),1e-25);self.assertLess(s.grad.norm().item(),1e-12)

    def test_spectral_feature_gradient_and_normalized_frobenius_reduction(self):
        t=F.normalize(torch.randn(2,5,7,dtype=torch.float64),dim=-1)
        s=F.normalize(torch.randn_like(t),dim=-1).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda x:compressed_gram_loss(x,t),(s,),fast_mode=True))
        loss=compressed_gram_loss(s,t)
        self.assertGreater(loss.item(),0);self.assertLess(loss.item(),4)

    def test_fast_root_and_gradient_agree_with_exact_frechet_at_near_rank_one(self):
        for scale in [0.,.001,1.]:
            x=F.normalize(torch.ones(2,16,5,dtype=torch.float64)+scale*torch.randn(2,16,5,dtype=torch.float64),dim=-1)
            g=(x@x.transpose(-1,-2)).requires_grad_()
            fast=gram_root_iteration(g);exact=_StableGramRoot.apply(g,.01)
            torch.testing.assert_close(fast,exact,atol=1e-10,rtol=1e-9)
            weight=torch.randn_like(g)
            ga,=torch.autograd.grad((fast*weight).sum(),g,retain_graph=True)
            gb,=torch.autograd.grad((exact*weight).sum(),g)
            torch.testing.assert_close(ga,gb,atol=1e-9,rtol=1e-8)


if __name__=='__main__':unittest.main(verbosity=2)
