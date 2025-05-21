import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
import math

class ResidualBlock(nn.Module):
    def __init__(self, infilters, filters, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(infilters, filters, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(infilters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(filters)
        
        self.shortcut_conv = None
        if infilters != filters:
            self.shortcut_conv = nn.Conv2d(infilters, filters, kernel_size=1, stride=stride, padding=0)
        
    def forward(self, x, context=None):
        shortcut = x
        out = self.conv1(F.silu(self.bn1(x)))
        out = self.conv2(F.silu(self.bn2(out)))
        
        # If shortcut needs adjustment for channel dimensions
        if self.shortcut_conv:
            shortcut = self.shortcut_conv(shortcut)
        
        out += shortcut
        return out

class EncoderBlock(nn.Module):
    def __init__(self, infilters, filters, kernel_size=3):
        super(EncoderBlock, self).__init__()
        self.res_blocks = nn.ModuleList([ResidualBlock(infilters, filters, kernel_size), ResidualBlock(filters, filters, kernel_size)])
        self.pool = nn.Conv2d(filters, filters, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        for block in self.res_blocks:
            x = block(x)
        p = self.pool(x)
        # return x, p
        return p

class DeformableAttention(nn.Module):
    def __init__(self, n_heads, d_head):
        super(DeformableAttention, self).__init__()
        self.embed_dims = n_heads * d_head
        self.num_heads = n_heads
        self.output_proj = nn.Linear(self.embed_dims, self.embed_dims)
        
        # Parameters for deformable attention
        self.num_levels = 1
        self.num_points = 8
        
        # Create MultiScaleDeformableAttention instance
        self.deform_attn = MultiScaleDeformableAttention(
            embed_dims=self.embed_dims,
            num_heads=self.num_heads,
            num_levels=self.num_levels,
            num_points=self.num_points,
            batch_first=True,
            im2col_step=1  # Set to 1 to support any batch size
        )
        
        self.value_proj = nn.Linear(self.embed_dims, self.embed_dims)
        
    def forward(self, inputs):
        assert isinstance(inputs, list)
        if len(inputs) == 1:
            inputs.append(None)
            
        x, context = inputs
        context = x if context is None else context
        
        # For deformable attention, input needs to be arranged as [B, L, C]
        B, L, _ = x.shape
        
        # Project value
        value = self.value_proj(context)
        
        # Create reference points - for sequence data, use a simple grid in normalized space [0, 1]
        reference_points = torch.zeros((B, L, self.num_levels, 2), device=x.device)
        # Set reference points to center (0.5, 0.5)
        reference_points += 0.5
        
        # Create spatial shapes info
        spatial_shapes = torch.tensor([[L, 1]], device=x.device, dtype=torch.long)
        level_start_index = torch.tensor([0], device=x.device, dtype=torch.long)
        
        # Use the MultiScaleDeformableAttention module to compute attention
        if L <= self.num_points:
            # For very short sequences, just use a simpler approach
            output = value
        else:
            # Use the deformable attention module
            output = self.deform_attn(
                query=x,
                key=None,  # key is generated inside the module
                value=value,
                query_pos=None,
                key_padding_mask=None,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index
            )
            
        # Apply output projection if needed
        return self.output_proj(output) if hasattr(self, 'output_proj') else output

class SpatialTransformer(nn.Module):
    def __init__(self, channels, n_heads, d_head):
        super(SpatialTransformer, self).__init__()
        assert channels == n_heads * d_head
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)  # GroupNorm instead of LayerNorm
        self.proj_in =  nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(channels, n_heads, d_head)])
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.view(b, c, -1).permute(0, 2, 1).contiguous()  # Reshape to (b, h * w, c)
        
        for block in self.transformer_blocks:
            x = block(x, context)
        
        x = x.permute(0, 2, 1).view(b, c, h, w).contiguous()  # Reshape back to (b, c, h, w)
        return self.proj_out(x) + x_in

class GEGLU(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim, dim_out * 2)
        self.dim_out = dim_out

    def forward(self, x):
        xp = self.proj(x)
        x, gate = xp[..., : self.dim_out], xp[..., self.dim_out :]
        return x * F.gelu(gate)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head):
        super(BasicTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = DeformableAttention(n_heads, d_head)

        self.norm3 = nn.LayerNorm(dim)
        self.geglu = GEGLU(dim, dim * 4)
        self.dense = nn.Linear(dim * 4, dim)

    def forward(self, x, context=None):
        x = self.attn1([self.norm1(x)]) + x
        return self.dense(self.geglu(self.norm3(x))) + x