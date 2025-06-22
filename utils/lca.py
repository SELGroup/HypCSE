

"""
LCA construction utils.
Considering that a batch of data is input, such as a whole graph,
which is suitable for calculating the volume of the LCA node.
"""
import torch
from utils.math import arctanh

MIN_NORM = 1e-10


def reflection_center(mu, eps=1e-12):
    """
    Center of inversion circle
    Compute the symmetric point of x with respect to the circle at infinity.
    mu: B * d
    B: batch size, d: poincare ball dimension
    """
    # return mu / torch.sum(mu ** 2, dim=-1, keepdim=True)
    return mu / (torch.sum(mu ** 2, dim=-1, keepdim=True).clamp(min=eps))


def isometric_transform(a, x, eps=1e-12):
    """
    Reflection (circle inversion of x through orthogonal circle centered at a).
    https://mphitchman.com/geometry/section3-2.html
    a: B * d
    x: B * d
    """
    r2 = torch.sum(a ** 2, dim=-1, keepdim=True) - 1.
    u = x - a
    # return r2 / torch.sum(u ** 2, dim=-1, keepdim=True) * u + a
    return r2 / (torch.sum(u ** 2, dim=-1, keepdim=True).clamp(min=eps)) * u + a


def euc_reflection(x, a):
    """
    Euclidean reflection (also hyperbolic) of x
    Along the geodesic that goes through a and the origin
    (straight line)
    x: B * d
    a: B * d
    """
    xTa = torch.sum(x * a, dim=-1, keepdim=True)
    norm_a_sq = torch.sum(a ** 2, dim=-1, keepdim=True).clamp_min(MIN_NORM)
    proj = xTa * a / norm_a_sq  # project vector ox on the vector oa
    return 2 * proj - x


def _halve(x):
    """ computes the point on the geodesic segment from o to x at half the distance """
    return x / (1. + torch.sqrt(1 - torch.sum(x ** 2, dim=-1, keepdim=True)))


def hyp_dist_o(x):
    """
    Computes hyperbolic distance between x and the origin.
    """
    x_norm = x.norm(dim=-1, p=2, keepdim=True)
    return 2 * arctanh(x_norm)


def hyp_lca(a, b, return_coord=True):
    """
    Computes projection of the origin on the geodesic between a and b, at scale c
    """
    # assert not torch.isnan(a).any()
    # assert not torch.isnan(b).any()
    # print(a)
    # print(a.size())
    r = reflection_center(a)
    # assert not torch.isnan(r).any()
    b_inv = isometric_transform(r, b)
    # assert not torch.isnan(b_inv).any()
    o_inv = a
    o_inv_ref = euc_reflection(o_inv, b_inv)
    # assert not torch.isnan(o_inv_ref).any()
    o_ref = isometric_transform(r, o_inv_ref)
    # assert not torch.isnan(o_ref).any()
    proj = _halve(o_ref)
    # assert not torch.isnan(proj).any()
    if not return_coord:
        return hyp_dist_o(proj)
    else:
        return proj, hyp_dist_o(proj)

