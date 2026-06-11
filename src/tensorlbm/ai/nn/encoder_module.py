import torch
from torch import nn
from einops.layers.torch import Rearrange
from tensorlbm.ai.nn.attention_module import PreNorm, StandardAttention, FeedForward, ReLUFeedForward, LinearAttention
class TransformerCatNoCls(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 attn_type,  # ['standard', 'galerkin', 'fourier']
                 use_ln=False,  # true
                 scale=16,  # can be list, or an int)
                 dropout=0.,
                 relative_emb_dim=3,
                 min_freq=1/64,
                 attention_init='orthogonal',
                 init_gain=None,
                 use_relu=False,
                 cat_pos=False
                 ):
        super().__init__()
        assert attn_type in ['standard', 'galerkin', 'fourier']

        if isinstance(scale, int):
            scale = [scale] * depth
        assert len(scale) == depth

        self.layers = nn.ModuleList([])
        self.attn_type = attn_type
        self.use_ln = use_ln

        if attn_type == 'standard':
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList([
                        PreNorm(dim, StandardAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)
                        if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout))]),
                )
        else:
            for d in range(depth):
                if scale[d] != -1 or not cat_pos:
                    attn_module = LinearAttention(dim, attn_type,
                                                  heads=heads, dim_head=dim_head, dropout=dropout,
                                                  relative_emb=True, scale=scale[d],
                                                  relative_emb_dim=relative_emb_dim,
                                                  min_freq=min_freq,
                                                  init_method=attention_init,
                                                  init_gain=init_gain,
                                                  use_ln=False,
                                                  )
                else:
                    attn_module = LinearAttention(dim, attn_type,
                                                  heads=heads, dim_head=dim_head, dropout=dropout,
                                                  cat_pos=True,
                                                  pos_dim=relative_emb_dim,
                                                  relative_emb=False,
                                                  init_method=attention_init,
                                                  init_gain=init_gain
                                                  )
                if not use_ln:
                    self.layers.append(
                        nn.ModuleList([
                            attn_module,
                            FeedForward(dim, mlp_dim, dropout=dropout)
                            if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout)
                        ]),
                    )
                else:
                    self.layers.append(
                        nn.ModuleList([
                            nn.LayerNorm(dim),
                            attn_module,
                            nn.LayerNorm(dim),
                            FeedForward(dim, mlp_dim, dropout=dropout)
                            if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout),
                        ]),
                    )

    def forward(self, x, pos_embedding):
        # x in [b n c], pos_embedding in [b n 2]
        b, n, c = x.shape

        for layer_no, attn_layer in enumerate(self.layers):
            if not self.use_ln:
                [attn, ffn] = attn_layer

                x = attn(x, pos_embedding) + x
                x = ffn(x) + x
            else:
                [ln1, attn, ln2, ffn] = attn_layer
                x = ln1(x)
                x = attn(x, pos_embedding) + x  # RoPE + LN + attn + x
                x = ln2(x)
                x = ffn(x) + x
        return x


class IrregSTEncoder2D(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 time_window,
                 in_emb_dim,  # embedding dim of token
                 out_channels,
                 heads,
                 depth,  # depth of transformer / how many layers of attention
                 res,
                 use_ln=True,
                 emb_dropout=0.05  # dropout of embedding
                 ):
        super().__init__()
        self.tw = time_window
        # here assume the input is in the shape [b, t, n, c]
        self.to_embedding = nn.Sequential(
            Rearrange('b t n c -> b c t n'),
            nn.Conv2d(input_channels, in_emb_dim, kernel_size=(self.tw, 1), stride=(self.tw, 1), padding=(0, 0),
                      bias=False),
            nn.GELU(),
            nn.Conv2d(in_emb_dim, in_emb_dim, kernel_size=(1, 1,), stride=(1, 1), padding=(0, 0), bias=False),
            Rearrange('b c 1 n -> b n c'),
        )
        self.dropout = nn.Dropout(emb_dropout)
        if depth > 4:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', use_ln,
                                                     scale=[32, 16, 8, 8] + [1] * (depth - 4),
                                                     min_freq=1 / res,
                                                     attention_init='orthogonal')
        else:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', use_ln,
                                                     scale=[32] + [16] * (depth - 2) + [1],
                                                     min_freq=1 / res,
                                                     attention_init='orthogonal')
        self.ln = nn.LayerNorm(in_emb_dim)

        self.to_out = nn.Sequential(
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.ReLU(),
            nn.Linear(in_emb_dim, out_channels, bias=False),
        )

    def forward(self,
                x,  # [b, t, n, c]
                input_pos,  # [b, n, 2]
                ):
        x = self.to_embedding(x)
        # x_node = self.node_embedding(node_type.squeeze(-1))
        # x = self.combine_embedding(torch.cat([x, x_node], dim=-1))
        x_skip = x

        x = self.dropout(x)

        x = self.s_transformer.forward(x, input_pos)
        #
        x = self.ln(x + x_skip)
        #
        x = self.to_out(x)
        return x