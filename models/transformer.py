import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """Multi headed self-attention accoring.
       Attention Is All You Need - Vaswani et al. https://arxiv.org/abs/1706.03762
    """
    def __init__(self, num_heads: int, dim_in: int, dim_hidden: int, dim_out: int, *args, **kwargs):
        """_summary_

        Args:
            num_heads: Number of attention heads
            dim_in: Input dimensions
            dim_hidden: Full hidden dimensions for all heads
            dim_out: Output dimensions
        """
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.dim_hidden = dim_hidden
        self.qkv_linear = nn.Linear(dim_in, 3 * dim_hidden)
        self.dim_head = dim_hidden // self.num_heads
        self.output = nn.Linear(dim_hidden, dim_out)

        nn.init.xavier_uniform_(self.qkv_linear.weight)
        self.qkv_linear.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.fill_(0)

    def forward(self, x: torch.Tensor, return_attention=False):
        """Computes self-attention for input tensor using unified q,k,v embeddings.

        Args:
            x: Input tensor
            return_attention: Return attention matrix. Defaults to False.

        Returns:
            Attention logits and optional attention matrix
        """
        batches, len_input, _ = x.shape
        qkv = self.qkv_linear(x)
        qkv = qkv.reshape(batches, len_input, self.num_heads, 3 * self.dim_head)
        qkv = torch.permute(qkv, (0, 2, 1, 3))  # [batch, heads, len_in, dim_head]
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # each [B, H, L, D]

        attention = q @ k.transpose(
            -2, -1
        )  # [B, H, L, D] x [B, H, D, L] -> [B, H, L, L]
        attention = attention / math.sqrt(self.dim_head)
        attention = torch.softmax(attention, dim=-1)
        logits = attention @ v  # [B, H, H] x [B, H, L] -> [B, H, L, H]
        logits = torch.permute(logits, (0, 2, 1, 3))  # [B, L, H, H]
        logits = torch.reshape(logits, (batches, len_input, self.dim_hidden))
        out = self.output(logits)  # [B, L, hidden] x [hidden, out] -> [B, L, out]
        if return_attention:
            return out, attention
        else:
            return out


class Encoder(nn.Module):
    """ Transformer encoder block according to:
        Attention Is All You Need - Vaswani et al. https://arxiv.org/abs/1706.03762    
    """
    def __init__(self, num_heads: int, dim_in: int, dim_mlp: int, dropout : float = 0.0, *args, **kwargs):
        """Initialize encoder block.

        Args:
            num_heads: Number of attention heads
            dim_in: Input dimension
            dim_mlp: Dimensions for internal 2 layer MLP
            dropout: Dropout probability between layers. Defaults to 0.0.
        """
        super().__init__(*args, **kwargs)
        self.multi_head_attn = MultiHeadAttention(num_heads, dim_in, dim_in, dim_in)
        self.layer_norm_1 = nn.LayerNorm(dim_in)
        self.mlp1 = nn.Linear(dim_in, dim_mlp)
        self.dropout_mlp = nn.Dropout(dropout)
        self.mlp2 = nn.Linear(dim_mlp, dim_in)
        self.layer_norm_2 = nn.LayerNorm(dim_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """ Compute encoder forward pass.

        Args:
            x: Input tensor

        Returns:
            Encoder logits
        """
        attention = self.multi_head_attn(x)
        x = x + self.dropout(attention)
        x = self.layer_norm_1(x)
        logits = self.mlp1(x)
        logits = self.dropout_mlp(logits)
        logits = torch.relu(logits)
        logits = self.mlp2(logits)
        x = x + self.dropout(logits)
        x = self.layer_norm_2(x)
        return x


class TransformerEncoder(nn.Module):
    """Stacked transformer encoder using multiple encoder blocks."""
    def __init__(self, num_encoders: int, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([Encoder(**kwargs) for _ in range(num_encoders)])

    def forward(self, x: torch.Tensor, mask=None):
        for block in self.blocks:
            x = block(x)
        return x

    def get_attention_maps(self, x: torch.Tensor, mask=None):
        attention_maps = []
        for l in self.blocks:
            _, attn_map = l.multi_head_attn(x, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps
