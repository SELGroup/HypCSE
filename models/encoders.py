import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import LorentzGraphConvolution
from utils.utils import select_activation
import math
from geoopt import ManifoldParameter


# The old GraphEncoder.
class LorentzGraphEncoder(nn.Module):
    def __init__(self, manifold, n_layers, in_features, n_hidden, out_dim,
                 dropout, nonlin=None, use_att=False, use_bias=False):
        super(LorentzGraphEncoder, self).__init__()
        self.manifold = manifold
        self.layers = nn.ModuleList([])
        self.layers.append(LorentzGraphConvolution(self.manifold, in_features,
                                                   n_hidden, use_bias, dropout=dropout, use_att=use_att, nonlin=None))
        for i in range(n_layers - 2):
            self.layers.append(LorentzGraphConvolution(self.manifold, n_hidden,
                                                       n_hidden, use_bias, dropout=dropout, use_att=use_att, nonlin=nonlin))
        self.layers.append(LorentzGraphConvolution(self.manifold, n_hidden,
                                                       out_dim, use_bias, dropout=dropout, use_att=use_att, nonlin=nonlin))

    def encode(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x


class FermiDiracDecoder(nn.Module):
    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = torch.sigmoid((self.r - dist) / self.t)
        return probs


class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs



class LorentzDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, manifold, dim, n_classes, bias=True):
        super(LorentzDecoder, self).__init__(c)
        self.manifold = manifold
        self.input_dim = dim
        self.output_dim = n_classes
        self.use_bias = bias
        self.cls = ManifoldParameter(self.manifold.random_normal((n_classes, dim), std=1./math.sqrt(dim)), manifold=self.manifold)
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_classes))
        self.decode_adj = False

    def decode(self, x, adj):
        return (2 + 2 * self.manifold.cinner(x, self.cls)) + self.bias