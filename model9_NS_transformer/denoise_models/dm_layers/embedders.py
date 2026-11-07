"""D3U patch and position embeddings used by the main PatchDN denoiser."""

import math
import torch
from torch import nn

class Time_series_PatchEmbed(nn.Module):
    """ time series to Patch Embedding
    """
    def __init__(
            self,
            context_window=96,
            patch_size=16,
            stride=8,
            padding_patch='end',
            embed_dim=768,
    ):
        super().__init__()

        self.context_window=context_window
        self.patch_size=patch_size
        self.stride = stride
        self.padding_patch = padding_patch
        self.embed_dim=embed_dim

        self.patch_num = int((self.context_window - self.patch_size)/self.stride + 1)
        if self.padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
            self.patch_num += 1

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(self.patch_size,self.embed_dim, bias=False)
        self.position_embedding = PositionalEmbedding(self.embed_dim)

        # Residual dropout
        # self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        n_vars = x.shape[1]
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)                   # z: [bs x nvars x patch_num x patch_size]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) # [B * N, patch_num, patch_len]
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x) # [B * N, patch_num, d_model]
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
 
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

