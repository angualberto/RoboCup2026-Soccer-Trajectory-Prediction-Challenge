import sys
import os
import math
import torch
import torch.nn as nn

_clifford_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'clifford-group-equivariant-neural-networks-master')
if _clifford_path not in sys.path:
    sys.path.insert(0, _clifford_path)

from algebra.cliffordalgebra import CliffordAlgebra
from models.modules.linear import MVLinear
from models.modules.gp import SteerableGeometricProductLayer
from models.modules.mvsilu import MVSiLU
from models.modules.mvlayernorm import MVLayerNorm
from models.modules.normalization import NormalizationLayer


class FastGeometricProductLayer(SteerableGeometricProductLayer):
    """
    Drop-in replacement for SteerableGeometricProductLayer with
    cached weight structure — eliminates per-forward-pass large tensor creation.

    The original layer constructs a full (out, in, n_blades, n_blades, n_blades)
    weight tensor from the sparse (out, in, paths) parameter on every forward pass.
    This version precomputes the fixed (paths, n_blades, n_blades, n_blades)
    mapping and contracts efficiently via einsum.
    """

    def __init__(self, algebra, features, include_first_order=True, normalization_init=0):
        super().__init__(algebra, features, include_first_order, normalization_init)
        self._cache_mapping()

    def _cache_mapping(self):
        """Precompute the fixed mapping from sparse weight to full blade tensor."""
        n_paths = self.product_paths.sum()
        # Build a (n_paths, n_subspaces, n_subspaces, n_subspaces) mask
        mask = torch.zeros(n_paths, *self.product_paths.size(),
                           dtype=torch.float32, device=self.algebra.cayley.device)
        path_idx = 0
        for i in range(self.product_paths.size(0)):
            for j in range(self.product_paths.size(1)):
                for k in range(self.product_paths.size(2)):
                    if self.product_paths[i, j, k]:
                        mask[path_idx, i, j, k] = 1.0
                        path_idx += 1
        # Expand subspaces to full blades via repeat_interleave
        subspaces = self.algebra.subspaces
        mapping = (
            mask.repeat_interleave(subspaces, dim=-3)
                .repeat_interleave(subspaces, dim=-2)
                .repeat_interleave(subspaces, dim=-1)
        )
        # Multiply by Cayley table
        mapping = mapping * self.algebra.cayley  # (n_paths, n_blades, n_blades, n_blades)
        self.register_buffer('_cached_mapping', mapping)

    def _get_weight(self):
        """
        weight:  (features, n_paths)
        mapping: (n_paths, n_blades, n_blades, n_blades)
        result:  (features, n_blades, n_blades, n_blades)  — 4D, matching parent.
        """
        return torch.einsum('fp, pijk -> fijk', self.weight, self._cached_mapping)


class CEGPBlock(nn.Module):
    """
    Clifford-equivariant block: MVLinear → MVSiLU → GeometricProduct → MVLayerNorm.
    Uses FastGeometricProductLayer for efficient weight computation.
    """

    def __init__(self, algebra, in_features, out_features, normalization_init=0):
        super().__init__()
        self.linear1 = MVLinear(algebra, in_features, out_features, subspaces=True)
        self.silu = MVSiLU(algebra, out_features, invariant='norm')
        self.gp = FastGeometricProductLayer(
            algebra, out_features,
            include_first_order=True,
            normalization_init=normalization_init,
        )
        self.norm = MVLayerNorm(algebra, out_features)

    def forward(self, x):
        return self.norm(self.gp(self.silu(self.linear1(x))))


class CliffordEncoder(nn.Module):
    """
    Clifford-equivariant GNN encoder using Cl(2,0).

    Replaces NeuralPropensityNet (GAT) with rotation-equivariant
    geometric product layers for encoding agent interactions.

    Input:  node_features (B, 22, 11)
    Output: latent        (B, 22, hidden_dim)   — same interface as GAT.

    Feature map (11 → 7 Cl(2,0) multivector channels):
        ch 0: position (x, y)          → grade-1 vector
        ch 1: velocity (vx, vy)        → grade-1 vector
        ch 2: ball direction (dx, dy)  → grade-1 vector
        ch 3: goal direction (dx, dy)  → grade-1 vector
        ch 4: stamina                  → grade-0 scalar
        ch 5: ball distance            → grade-0 scalar
        ch 6: goal distance            → grade-0 scalar

    Each channel is a full Cl(2,0) multivector (4 blades):
        [scalar, e1, e2, e12].
    """

    def __init__(self, node_in_features=11, hidden_dim=64, dropout=0.2, n_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.algebra = CliffordAlgebra((1.0, 1.0))  # Cl(2,0)
        self.n_layers = n_layers

        # Input embedding: 7 feature channels → hidden_dim channels
        self.input_proj = MVLinear(self.algebra, 7, hidden_dim, subspaces=True)

        # Fast Clifford blocks
        layers = []
        for i in range(n_layers):
            layers.append(CEGPBlock(self.algebra, hidden_dim, hidden_dim))
        self.blocks = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)

    def encode(self, node_features, adj_batch=None):
        B, N, _ = node_features.shape
        device = node_features.device

        # ---- Build 7 multivector channels from the 11 raw features ----
        mv_channels = []

        pos = node_features[..., 0:2]
        ch0 = torch.zeros(B, N, 4, device=device)
        ch0[..., 1:3] = pos
        mv_channels.append(ch0)

        vel = node_features[..., 2:4]
        ch1 = torch.zeros(B, N, 4, device=device)
        ch1[..., 1:3] = vel
        mv_channels.append(ch1)

        bdir = node_features[..., 6:8]
        ch2 = torch.zeros(B, N, 4, device=device)
        ch2[..., 1:3] = bdir
        mv_channels.append(ch2)

        gdir = node_features[..., 9:11]
        ch3 = torch.zeros(B, N, 4, device=device)
        ch3[..., 1:3] = gdir
        mv_channels.append(ch3)

        stamina = node_features[..., 4:5]
        ch4 = torch.zeros(B, N, 4, device=device)
        ch4[..., 0:1] = stamina
        mv_channels.append(ch4)

        bdist = node_features[..., 5:6]
        ch5 = torch.zeros(B, N, 4, device=device)
        ch5[..., 0:1] = bdist
        mv_channels.append(ch5)

        gdist = node_features[..., 8:9]
        ch6 = torch.zeros(B, N, 4, device=device)
        ch6[..., 0:1] = gdist
        mv_channels.append(ch6)

        x = torch.stack(mv_channels, dim=2)  # (B, N, 7, 4)

        # ---- Forward through Clifford layers ----
        x = x.view(B * N, 7, 4)
        x = self.input_proj(x)
        x = self.blocks(x)

        latent = x[..., 0]                     # (B*N, hidden_dim)
        latent = latent.view(B, N, self.hidden_dim)
        latent = self.dropout(latent)
        return latent

    def forward(self, node_features, adj_batch):
        return self.encode(node_features, adj_batch)
