import geoopt
import torch
import geoopt.manifolds.stereographic.math as math
import geoopt.manifolds.lorentz.math as lmath


class Poincare(geoopt.PoincareBall):
    def __init__(self, c=1.0, learnable=False):
        super(Poincare, self).__init__(c=c, learnable=learnable)

    def from_lorentz(self, x, dim=-1):
        x = x.to(self.c.device)
        dn = x.size(dim) - 1
        return x.narrow(dim, 1, dn) / (x.narrow(dim, 0, 1) + torch.sqrt(self.c))

    def to_lorentz(self, x, dim=-1, eps=1e-6):
        x = x.to(self.c.device)
        x_norm_square = torch.sum(x * x, dim=dim, keepdim=True)
        res = (
                torch.sqrt(self.c)
                * torch.cat((1 + x_norm_square, 2 * x), dim=dim)
                / (1.0 - x_norm_square + eps)
        )
        return res

    def Frechet_mean(self, embeddings, weights=None, keepdim=False):
        z = self.to_lorentz(embeddings)
        if weights is None:
            z = torch.sum(z, dim=0, keepdim=True)
        else:
            z = torch.sum(z * weights, dim=0, keepdim=keepdim)
        denorm = lmath.inner(z, z, keepdim=keepdim)
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        z = z / denorm
        z = self.from_lorentz(z).to(embeddings.device)
        return z


def project(x):
    MIN_NORM = 1e-15
    BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}
    """Projects points on the manifold."""
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)