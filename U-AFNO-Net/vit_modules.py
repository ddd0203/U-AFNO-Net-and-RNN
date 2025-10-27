import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Attention


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


class ViTBlock(nn.Module):
    """Vision Transformer Block with AFNO-style structure"""

    def __init__(
            self,
            dim,
            num_heads=8,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, attn_drop=drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x shape: [B, H, W, C] (matching AFNO format)
        B, H, W, C = x.shape
        # Reshape to sequence format for attention
        x_seq = x.reshape(B, H * W, C)
        
        # Self-attention with residual
        x_seq = x_seq + self.drop_path(self.attn(self.norm1(x_seq)))
        
        # MLP with residual
        x_seq = x_seq + self.drop_path(self.mlp(self.norm2(x_seq)))
        
        # Reshape back to spatial format
        x = x_seq.reshape(B, H, W, C)
        return x


class ViTNet(nn.Module):
    """Simple ViT with AFNO-style structure"""

    def __init__(
            self,
            img_size=(224, 224),
            patch_size=(16, 16),
            in_chans=3,
            out_chans=3,
            embed_dim=768,
            depth=12,
            num_blocks=8,
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

        # ViT blocks
        self.blocks = nn.ModuleList([
            ViTBlock(
                dim=embed_dim,
                num_heads=num_blocks,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

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
        
        # Apply ViT blocks
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

def test_simple_vit():
    """简单测试ViT模型是否能正常运行"""
    
    # 设置随机种子以便复现
    torch.manual_seed(42)
    
    # 模型参数
    img_size = (224, 224)
    patch_size = (16, 16)
    in_chans = 3
    out_chans = 3
    embed_dim = 768
    depth = 12
    
    # 创建模型
    print("创建模型...")
    model = ViTNet(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        out_chans=out_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_blocks=8,
        mlp_ratio=4.0,
        drop_rate=0.1,
        drop_path_rate=0.1
    )
    
    # 打印模型信息
    print(f"模型创建成功!")
    print(f"图像大小: {img_size}")
    print(f"Patch大小: {patch_size}")
    print(f"嵌入维度: {embed_dim}")
    print(f"模型深度: {depth}")
    
    # 创建随机输入
    batch_size = 2
    x = torch.randn(batch_size, in_chans, *img_size)
    print(f"\n输入张量形状: {x.shape}")
    
    # 前向传播
    print("\n进行前向传播...")
    with torch.no_grad():
        output = model(x)
    
    print(f"输出张量形状: {output.shape}")
    print(f"期望输出形状: [{batch_size}, {out_chans}, {img_size[0]}, {img_size[1]}]")
    
    # 验证输出形状
    assert output.shape == (batch_size, out_chans, img_size[0], img_size[1]), "输出形状不正确!"
    print("\n✓ 输出形状正确!")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量:")
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    print("\n测试通过! 模型运行正常 ✓")


def test_different_configs():
    """测试不同配置"""
    print("\n" + "="*50)
    print("测试不同配置...")
    print("="*50)
    
    configs = [
        {"img_size": (128, 128), "patch_size": (16, 16), "embed_dim": 384, "depth": 6},
        {"img_size": (256, 256), "patch_size": (32, 32), "embed_dim": 512, "depth": 8},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n配置 {i+1}: {config}")
        
        model = ViTNet(
            img_size=config["img_size"],
            patch_size=config["patch_size"],
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            in_chans=3,
            out_chans=3
        )
        
        x = torch.randn(1, 3, *config["img_size"])
        output = model(x)
        
        print(f"输入: {x.shape} -> 输出: {output.shape}")
        assert output.shape == (1, 3, *config["img_size"]), f"配置 {i+1} 输出形状错误!"
        print(f"✓ 配置 {i+1} 测试通过!")


if __name__ == "__main__":
    # 运行基本测试
    test_simple_vit()
    
    # 运行不同配置测试
    test_different_configs()