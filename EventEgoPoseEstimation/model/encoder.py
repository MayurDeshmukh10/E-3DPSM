import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_head):
        super(SelfAttention, self).__init__()
        self.to_q = nn.Linear(n_heads * d_head, n_heads * d_head, bias=False)
        self.to_k = nn.Linear(n_heads * d_head, n_heads * d_head, bias=False)
        self.to_v = nn.Linear(n_heads * d_head, n_heads * d_head, bias=False)
        
        self.scale = d_head ** -0.5
        self.num_heads = n_heads
        self.head_size = d_head
        self.to_out = nn.Linear(n_heads * d_head, n_heads * d_head)

    def forward(self, inputs):
        assert isinstance(inputs, list)
        if len(inputs) == 1:
            inputs.append(None)

        x, context = inputs           
        context = x if context is None else context
        
        # Project inputs to queries, keys, and values
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape for multi-head attention
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_size).permute(0, 2, 1, 3).contiguous()
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_size).permute(0, 2, 3, 1).contiguous()
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_size).permute(0, 2, 1, 3).contiguous()

        # Scaled dot-product attention
        score = torch.matmul(q, k) * self.scale
        weights = F.softmax(score, dim=-1)
        attention = torch.matmul(weights, v)

        # Reshape back and apply output linear layer
        attention = attention.permute(0, 2, 1, 3).contiguous()
        h_ = attention.view(attention.shape[0], -1, self.num_heads * self.head_size)
        return self.to_out(h_)


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
        self.attn1 = SelfAttention(n_heads, d_head)

        self.norm3 = nn.LayerNorm(dim)
        self.geglu = GEGLU(dim, dim * 4)
        self.dense = nn.Linear(dim * 4, dim)

    def forward(self, x, context=None):
        x = self.attn1([self.norm1(x)]) + x
        return self.dense(self.geglu(self.norm3(x))) + x
