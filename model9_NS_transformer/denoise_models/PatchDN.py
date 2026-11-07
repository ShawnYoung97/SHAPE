

import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Attention, Mlp
from einops import repeat
from .dm_layers.embedders import Time_series_PatchEmbed
# from model9_NS_transformer.ns_layers.RevIN import RevIN
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, d_model, hidden_size):
        super().__init__()
        self.embedding_table = nn.Linear(d_model, hidden_size , bias=True)

    def forward(self, y):
        embeddings=self.embedding_table(y)

        return embeddings


#################################################################################
#                                 Core PatchDN Model                                #
#################################################################################

class PatchDNBlock(nn.Module):
    """D3U PatchDN block with adaptive layer-normalization conditioning."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio),
                       act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of PatchDN.
    """
    def __init__(self, hidden_size, patch_num, context_window):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(hidden_size*patch_num, context_window , bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class PatchDN(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        MTS_args,
        depth=2,
        mlp_ratio=4.0,
        learn_sigma=False,
        pe='zeros',
        learn_pe=True,

    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels =MTS_args.enc_in
        self.out_channels =self.in_channels * 2 if learn_sigma else self.in_channels
        self.num_heads = MTS_args.n_heads_d
        self.context_window= MTS_args.pred_len
        self.condition_hidden_size= MTS_args.d_model_c
        self.hidden_size= MTS_args.d_model_d

        ###patch 参数
        self.patch_size = MTS_args.patch_size
        self.stride = MTS_args.stride
        self.padding_patch = MTS_args.padding_patch

        self.x_embedder = Time_series_PatchEmbed( self.context_window,self.patch_size,self.stride,self.padding_patch,self.hidden_size)
        patch_num= self.x_embedder.patch_num
        condition_padding = self.stride if self.padding_patch == 'end' else 0
        _cond_patch_num_override = getattr(MTS_args, 'cond_patch_num', None)
        if _cond_patch_num_override is not None:
            self.condition_patch_num = _cond_patch_num_override
        else:
            self.condition_patch_num = int((MTS_args.seq_len + condition_padding - self.patch_size) / self.stride + 1)

        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.y_embedder = LabelEmbedder(self.condition_patch_num * self.condition_hidden_size,
                                       self.hidden_size)

        # self.pos_embed = nn.Parameter(torch.zeros(1, patch_num, self.hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            PatchDNBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(self.hidden_size, patch_num, self.context_window)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.patch_num ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        nn.init.normal_(self.x_embedder.value_embedding.weight, std=0.02)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, x, t, y):
        """Predict residuals from [B, H, C] inputs and TimeFilter states."""
        x = x.permute(0, 2, 1)
        batch, channels, horizon = x.shape
        x = self.x_embedder(x)
        t = repeat(t, "b -> b d", d=channels).reshape(-1)
        t = self.t_embedder(t)
        batch, channels, patches, hidden = y.shape
        y = self.y_embedder(y.reshape(batch * channels, patches * hidden))
        condition = t + y
        for block in self.blocks:
            x = block(x, condition)
        x = self.final_layer(x, condition)
        return x.reshape(batch, channels, horizon).permute(0, 2, 1)
