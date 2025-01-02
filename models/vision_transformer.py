import torch
import torch.nn as nn
from models.transformer import TransformerEncoder

def img_to_patch(x, patch_size, flatten_channels=True):
    """Image to patch function from UvA DL course.
        https://uvadlc-notebooks.readthedocs.io/

    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        dim_embedded: int,
        dim_mlp: int,
        num_encoders: int,
        num_classes: int,
        num_patches: int,
        dropout: float = 0.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.patch_proj = nn.Linear(patch_size * patch_size * 3, dim_embedded)
        self.transformer = TransformerEncoder(
            num_encoders=num_encoders,
            dim_in=dim_embedded,
            dim_mlp=2 * (dim_embedded),
            num_heads=8,
            dropout=0.1,
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim_embedded),
            nn.Linear(dim_embedded, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, num_classes),
        )
        self.dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_embedded))
        self.position_embedding = nn.Parameter(
            torch.randn(1, 1 + num_patches, dim_embedded)
        )

    def forward(self, x):
        x = img_to_patch(
            x, self.patch_size, flatten_channels=True
        )  # [B, H'*W', C*p_H*p_W]
        B, T, _ = x.shape
        x = self.patch_proj(x)  # [B, H'*W', dim_embedded]
        cls_token = self.cls_token.repeat(B, 1, 1)  # [B, 1, dim_embedded]
        x = torch.concat([cls_token, x], dim=1)  # [B, H'*W'+1, dim_embedded]
        x = x + self.position_embedding[:, : T + 1]

        x = self.dropout(x)
        x = x.transpose(0, 1)  # [H'*W'+1, B, dim_embedded]
        x = self.transformer(x)  # [H'*W'+1, B, dim_embedded]

        cls = x[0]  # take added CLS embedding for classification
        out = self.mlp_head(cls)
        return out
