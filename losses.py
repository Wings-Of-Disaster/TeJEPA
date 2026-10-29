"""Paired-view patch reliability and stable spectral Gram compression.

Crop boxes are normalized (top,left,height,width) in the ORIGINAL image.
Horizontal flips are recorded explicitly. No image labels enter these losses.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def correspondence_grid(box_a, flip_a, box_b, flip_b, height, width):
    """A patch centers -> original image -> B grid, align_corners=False.

    Keep centers in B's interpolation support; padding never counts as visible.
    """
    y = (torch.arange(height, device=box_a.device, dtype=box_a.dtype)+.5)/height
    x = (torch.arange(width, device=box_a.device, dtype=box_a.dtype)+.5)/width
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    xx = xx.expand(len(box_a), -1, -1)
    xx = torch.where(flip_a[:, None, None], 1-xx, xx)
    ox = box_a[:, 1, None, None]+box_a[:, 3, None, None]*xx
    oy = box_a[:, 0, None, None]+box_a[:, 2, None, None]*yy
    bx = (ox-box_b[:, 1, None, None])/box_b[:, 3, None, None]
    by = (oy-box_b[:, 0, None, None])/box_b[:, 2, None, None]
    bx = torch.where(flip_b[:, None, None], 1-bx, bx)
    valid = ((bx >= .5/width-1e-6) & (bx <= 1-.5/width+1e-6) &
             (by >= .5/height-1e-6) & (by <= 1-.5/height+1e-6))
    return torch.stack((2*bx-1, 2*by-1), -1), valid


def reliable_patch_loss(student, teacher, boxes, flips, floor=.1, temperature=.2):
    """Symmetric overlap matching, weighted per image by teacher consistency.

    student/teacher: [2,B,P,D], already dimension/position aligned; boxes[2,B,4].
    If neither direction has an overlap sample, use uniform two-view matching
    for that image, rather than drop its supervision or divide by zero.
    """
    if student.shape != teacher.shape or student.ndim != 4 or student.size(0) != 2:
        raise ValueError('Expected paired 2 x B x P x D tokens')
    if not 0 < floor <= 1 or temperature <= 0:
        raise ValueError('Invalid reliability parameters')
    p = student.size(2); h = math.isqrt(p)
    if h*h != p: raise ValueError('Square patch grids required')
    s, t = F.normalize(student.float(), dim=-1), F.normalize(teacher.detach().float(), dim=-1)
    numerators, denominators, counts, mean_weights = [], [], [], []
    for a,b in [(0,1),(1,0)]:
        grid, valid = correspondence_grid(boxes[a].float(), flips[a], boxes[b].float(), flips[b], h,h)
        def sample(z):
            image = z.transpose(1,2).reshape(len(z),z.size(-1),h,h)
            mapped = F.grid_sample(image,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
            return F.normalize(mapped.flatten(2).transpose(1,2),dim=-1)
        tb, sb = sample(t[b]), sample(s[b])
        consistency = (t[a]*tb).sum(-1).clamp(-1,1)
        weight = (floor+(1-floor)*torch.exp(-(1-consistency)/temperature)).detach()
        error = .5*((1-(s[a]*t[a]).sum(-1))+(1-(sb*tb).sum(-1)))
        mask = valid.flatten(1).to(weight)
        numerators.append((error*weight*mask).sum(1))
        denominators.append((weight*mask).sum(1))
        counts.append(mask.sum(1));mean_weights.append((weight*mask).sum())
    denominator = sum(denominators)
    fallback = (1-(s*t).sum(-1)).mean(dim=(0,2))
    per_image = torch.where(denominator>0,sum(numerators)/denominator.clamp_min(1e-12),fallback)
    return per_image.mean(), dict(overlap_fraction=sum(counts).sum()/(2*len(denominator)*p),
        no_overlap_fraction=(denominator==0).float().mean(),
        reliability_mean=sum(mean_weights)/sum(counts).sum().clamp_min(1))


class _StableGramRoot(torch.autograd.Function):
    """sqrt(G+delta I)-sqrt(delta)I with its Frechet derivative.

    For sqrt eigenvalues r_i, divided differences are 1/(r_i+r_j), including
    repeated eigenvalues. No differentiation through eigenvectors or eigenvalue
    gaps: the derivative stays finite because delta is strictly positive.
    """
    @staticmethod
    def forward(ctx, gram, delta):
        eigenvalues, vectors = torch.linalg.eigh((gram+gram.transpose(-1,-2))*.5)
        if eigenvalues.min() < -1e-3:
            raise FloatingPointError('Gram lost positive semidefiniteness beyond roundoff')
        roots = (eigenvalues+delta).clamp_min(delta*.5).sqrt()
        ctx.save_for_backward(vectors,roots)
        return (vectors*(roots-math.sqrt(delta)).unsqueeze(-2))@vectors.transpose(-1,-2)

    @staticmethod
    def backward(ctx, grad):
        vectors,roots = ctx.saved_tensors
        symmetric = (grad+grad.transpose(-1,-2))*.5
        inner = vectors.transpose(-1,-2)@symmetric@vectors
        result = vectors@(inner/(roots.unsqueeze(-1)+roots.unsqueeze(-2)))@vectors.transpose(-1,-2)
        return result,None


def gram_root_iteration(gram,delta=.01,iterations=18):
    """Newton--Schulz square root; avoids eigenvectors and their singular gaps.

    Trace scaling bounds eigenvalues in (0,1]. The positive spectral shift
    supplies a lower eigenvalue bound. Eigh/Frechet remains an audit reference;
    approximation value and gradient are checked against it on conditioned and
    near-rank-one Gram matrices before this implementation is used in training.
    """
    eye=torch.eye(gram.size(-1),device=gram.device,dtype=gram.dtype)
    a=(gram+gram.transpose(-1,-2))*.5+delta*eye
    scale=a.diagonal(dim1=-2,dim2=-1).sum(-1)[...,None,None]
    y=a/scale;z=eye.expand_as(y)
    for _ in range(iterations):
        update=.5*(3*eye-z@y)
        y=y@update;z=update@z
    result=y*scale.sqrt()-math.sqrt(delta)*eye
    return (result+result.transpose(-1,-2))*.5


def compressed_gram_loss(student, teacher, delta=.01, eps=1e-8):
    """Proposal gamma=1/2, shifted spectral compression + Frobenius normalization.

    Features [...,P,D] are normalized patch tokens. Unlike old elementwise MSE,
    this loss uses squared Frobenius norm, so its weight MUST be recalibrated.
    """
    if delta <= 0: raise ValueError('Positive spectral shift required')
    def representation(tokens):
        gram = tokens@tokens.transpose(-1,-2)
        result = gram_root_iteration(gram,delta)
        return result/(torch.linalg.matrix_norm(result,ord='fro',dim=(-2,-1),keepdim=True)+eps)
    s = representation(student)
    with torch.no_grad(): t = representation(teacher.detach())
    return (s-t).square().sum(dim=(-2,-1)).mean()
