import torch
import torch.nn as nn

from layers.Embed import PositionalEmbedding
from layers.StandardNorm import Normalize
from layers.TimeFilter_layers import TimeFilter_Backbone


class PatchEmbed(nn.Module):
    """Patch embedding for TimeFilter: unfold + linear projection + positional encoding."""
    def __init__(self, dim, patch_len, stride=None, pos=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)
        self.pos = pos
        if self.pos:
            self.pe = PositionalEmbedding(dim, max_len=10000)

    def forward(self, x):
        # x: [B, C*T] (flattened)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [B, N*L, P]
        x = self.patch_proj(x)  # [B, N*L, D]
        if self.pos:
            x += self.pe(x)
        return x


class Model(nn.Module):
    """
    TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting.
    Paper link: https://arxiv.org/abs/2501.13041 (ICML 2025)

    Adapted as a D3U conditioning model.
    Forward interface: (x_enc, x_mark_enc, x_dec, x_mark_dec) -> (dec_out, loss, enc_out)
    """

    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.c_out
        # Use _c suffixed params for D3U conditioning model consistency
        self.dim = configs.d_model_c
        self.d_ff = configs.d_ff
        self.patch_len = getattr(configs, 'patch_len', 8)
        self.stride = self.patch_len
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1)  # L

        # Filter hyperparameters
        self.alpha = getattr(configs, 'alpha', 0.1)
        if self.alpha is None:
            self.alpha = 0.1
        self.top_p = getattr(configs, 'top_p', 0.0)
        if self.top_p is None:
            self.top_p = 0.0

        # Number of attention heads (use _c suffix)
        n_heads = configs.n_heads_c

        # Positional embedding flag (argparse passes int 0/1)
        pos = bool(getattr(configs, 'pos', 1))

        # Patch embedding
        self.patch_embed = PatchEmbed(self.dim, self.patch_len, self.stride, pos)

        # TimeFilter Backbone
        e_layers = configs.e_layers_c
        self.backbone = TimeFilter_Backbone(
            self.dim, self.n_vars, self.d_ff,
            n_heads, e_layers, self.top_p,
            configs.dropout,
            self.seq_len * self.n_vars // self.patch_len
        )

        # Prediction head
        self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)

        # Instance normalization (RevIN-style, without learnable affine by default)
        self.use_RevIN = False
        self.norm = Normalize(configs.enc_in, affine=self.use_RevIN)

    def forecast(self, x_enc):
        """Return the forecast and [B, C, patches, hidden] conditioning state."""
        batch, _, _ = x_enc.shape
        x = self.norm(x_enc, 'norm')
        x = x.permute(0, 2, 1).reshape(batch, -1)
        x = self.patch_embed(x)
        x, auxiliary_loss = self.backbone(x, None, self.alpha, self.training)
        enc_out = x.reshape(batch, self.n_vars, self.num_patches, self.dim)
        forecast = self.head(enc_out.flatten(start_dim=-2)).permute(0, 2, 1)
        forecast = self.norm(forecast, 'denorm')
        return forecast, auxiliary_loss, enc_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        forecast, auxiliary_loss, enc_out = self.forecast(x_enc)
        return forecast[:, -self.pred_len:, :], auxiliary_loss, enc_out
