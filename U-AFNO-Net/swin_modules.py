import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
from timm.models.swin_transformer import WindowAttention, window_partition, window_reverse


class Mlp(nn.Module):
    """MLP module with GELU activation and dropout."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """将图像分割成patches并嵌入的模块"""

    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[
            1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class SwinBlock(nn.Module):
    """Swin Transformer Block with AFNO-style structure"""

    def __init__(
            self,
            dim,
            num_heads=8,
            window_size=7,
            shift_size=0,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Attention and MLP modules
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(window_size, window_size), num_heads=num_heads,
            qkv_bias=True, attn_drop=drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x shape: [B, H, W, C] (matching AFNO format)
        B, H, W, C = x.shape

        # Save input for residual
        shortcut = x

        # Layer norm
        x = self.norm1(x)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, (self.window_size, self.window_size))  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # Window attention (no mask for simplicity)
        attn_windows = self.attn(x_windows, mask=None)  # nW*B, window_size*window_size, C

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, (self.window_size, self.window_size), H, W)  # B H W C

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # First residual connection
        x = shortcut + self.drop_path(x)

        # MLP with second residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinNet(nn.Module):
    """Simple Swin Transformer with AFNO-style structure"""

    def __init__(
            self,
            img_size=(224, 224),
            patch_size=(16, 16),
            in_chans=3,
            out_chans=3,
            embed_dim=768,
            depth=12,
            num_blocks=8,
            window_size=7,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_features = self.embed_dim = embed_dim
        self.window_size = window_size
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Patch embedding (same as AFNO)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Calculate spatial dimensions
        self.h = img_size[0] // patch_size[0]
        self.w = img_size[1] // patch_size[1]

        # Build Swin blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            # Alternate between shifted and non-shifted windows
            shift_size = 0 if (i % 2 == 0) else window_size // 2

            # Ensure shift_size is valid
            if min(self.h, self.w) <= window_size:
                shift_size = 0  # No shifting if window is too large

            block = SwinBlock(
                dim=embed_dim,
                num_heads=num_blocks,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            self.blocks.append(block)

        # Final norm and head (same as AFNO)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, out_chans * patch_size[0] * patch_size[1], bias=False)

        # Initialize weights
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Reshape to spatial format for processing
        x = x.reshape(B, self.h, self.w, self.embed_dim)

        # Apply Swin blocks
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        # Reorganize output dimensions (same as AFNO)
        x = x.view(x.shape[0], self.h, self.w,
                   self.patch_size[0], self.patch_size[1], self.out_chans)
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, C, h, p1, w, p2]
        x = x.reshape(x.shape[0], self.out_chans,
                      self.h * self.patch_size[0], self.w * self.patch_size[1])

        return x

if __name__ == "__main__":
    print("SwinNet 简单测试")
    print("-" * 40)
    
    # 模型参数
    img_size = (256, 256)
    patch_size = (4, 4)
    in_chans = 4
    out_chans = 4
    embed_dim = 768
    depth = 8
    num_blocks = 8  # 注意：这里实际上是num_heads
    window_size = 4
    
    # 创建模型
    print("创建SwinNet模型...")
    model = SwinNet(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        out_chans=out_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_blocks=num_blocks,  # 实际上是num_heads
        window_size=window_size,
        mlp_ratio=4.0,
        drop_rate=0.1,
        drop_path_rate=0.1
    )
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数数量: {total_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 ** 2:.2f} MB")
    
    # 打印模型配置
    print(f"\n模型配置:")
    print(f"  - 图像尺寸: {img_size}")
    print(f"  - Patch大小: {patch_size}")
    print(f"  - 输入通道: {in_chans}")
    print(f"  - 输出通道: {out_chans}")
    print(f"  - 嵌入维度: {embed_dim}")
    print(f"  - 模型深度: {depth}")
    print(f"  - 注意力头数: {num_blocks}")
    print(f"  - 窗口大小: {window_size}")
    
    # 准备测试数据
    batch_size = 16
    input_tensor = torch.randn(batch_size, in_chans, *img_size)
    print(f"\n输入形状: {input_tensor.shape}")
    
    # 前向传播测试
    print("\n进行前向传播测试...")
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"输出形状: {output.shape}")
    print(f"期望输出形状: [{batch_size}, {out_chans}, {img_size[0]}, {img_size[1]}]")
    
    # 验证输出形状
    assert output.shape == (batch_size, out_chans, img_size[0], img_size[1]), "输出形状不正确!"
    print("\n✓ 输出形状正确!")
    
    # 测试不同的配置
    print("\n" + "="*50)
    print("测试不同配置...")
    print("="*50)
    
    configs = [
        {"img_size": (128, 128), "patch_size": (4, 4), "window_size": 4, "embed_dim": 96, "depth": 4},
        {"img_size": (256, 256), "patch_size": (8, 8), "window_size": 8, "embed_dim": 192, "depth": 12},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n配置 {i+1}: {config}")
        
        model_test = SwinNet(
            img_size=config["img_size"],
            patch_size=config["patch_size"],
            window_size=config["window_size"],
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            in_chans=4,
            out_chans=4,
            num_blocks=8
        )
        
        x = torch.randn(1, 4, *config["img_size"])
        output = model_test(x)
        
        params = sum(p.numel() for p in model_test.parameters())
        print(f"输入: {x.shape} -> 输出: {output.shape}")
        print(f"参数量: {params:,}")
        assert output.shape == (1, 4, *config["img_size"]), f"配置 {i+1} 输出形状错误!"
        print(f"✓ 配置 {i+1} 测试通过!")
    
    print("\n✅ 所有测试完成！")