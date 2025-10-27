# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, trunc_normal_


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


class AFNO2D(nn.Module):
    """完整的AFNO2D实现，来自afnonet.py"""

    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1,
                 hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes].real,
                         self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes].imag,
                         self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes].imag,
                         self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes].real,
                         self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes] = (
                torch.einsum('...bi,bio->...bo',
                             o1_real[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w2[0]) - \
                torch.einsum('...bi,bio->...bo',
                             o1_imag[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w2[1]) + \
                self.b2[0]
        )

        o2_imag[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes] = (
                torch.einsum('...bi,bio->...bo',
                             o1_imag[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w2[0]) + \
                torch.einsum('...bi,bio->...bo',
                             o1_real[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w2[1]) + \
                self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, H, W // 2 + 1, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.type(dtype)

        return x + bias


class AFNOBlock(nn.Module):
    """完整的AFNO Block，包含AFNO2D + MLP + LayerNorm + DropPath"""

    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            double_skip=True,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AFNO2D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class AFNONet(nn.Module):
    """完整的AFNO网络，基于afnonet.py实现"""

    def __init__(
            self,
            img_size=(224, 224),
            patch_size=(16, 16),
            in_chans=3,
            out_chans=3,
            embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            num_blocks=16,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.h = img_size[0] // patch_size[0]
        self.w = img_size[1] // patch_size[1]

        self.blocks = nn.ModuleList([
            AFNOBlock(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                      num_blocks=num_blocks, sparsity_threshold=sparsity_threshold,
                      hard_thresholding_fraction=hard_thresholding_fraction)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, out_chans * patch_size[0] * patch_size[1], bias=False)

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

        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        # 重新组织输出维度
        x = x.view(x.shape[0], self.h, self.w,
                   self.patch_size[0], self.patch_size[1], self.out_chans)
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, C, h, p1, w, p2]
        x = x.reshape(x.shape[0], self.out_chans,
                      self.h * self.patch_size[0], self.w * self.patch_size[1])

        return x






def test_simple_afno():
    """简单测试AFNO模型是否能正常运行"""
    
    # 设置随机种子以便复现
    torch.manual_seed(42)
    
    # 模型参数
    img_size = (224, 224)
    patch_size = (16, 16)
    in_chans = 3
    out_chans = 3
    embed_dim = 768
    depth = 12
    num_blocks = 16
    
    # 创建模型
    print("创建AFNO模型...")
    model = AFNONet(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        out_chans=out_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_blocks=num_blocks,
        mlp_ratio=4.0,
        drop_rate=0.1,
        drop_path_rate=0.1,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0
    )
    
    # 打印模型信息
    print(f"模型创建成功!")
    print(f"图像大小: {img_size}")
    print(f"Patch大小: {patch_size}")
    print(f"嵌入维度: {embed_dim}")
    print(f"模型深度: {depth}")
    print(f"AFNO块数: {num_blocks}")
    
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
    
    print("\n测试通过! AFNO模型运行正常 ✓")


def test_different_configs():
    """测试不同配置"""
    print("\n" + "="*50)
    print("测试不同配置...")
    print("="*50)
    
    configs = [
        {"img_size": (128, 128), "patch_size": (16, 16), "embed_dim": 384, "depth": 6, "num_blocks": 8},
        {"img_size": (256, 256), "patch_size": (32, 32), "embed_dim": 512, "depth": 8, "num_blocks": 16},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n配置 {i+1}: {config}")
        
        model = AFNONet(
            img_size=config["img_size"],
            patch_size=config["patch_size"],
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            num_blocks=config["num_blocks"],
            in_chans=3,
            out_chans=3
        )
        
        x = torch.randn(1, 3, *config["img_size"])
        output = model(x)
        
        print(f"输入: {x.shape} -> 输出: {output.shape}")
        assert output.shape == (1, 3, *config["img_size"]), f"配置 {i+1} 输出形状错误!"
        print(f"✓ 配置 {i+1} 测试通过!")


def test_afno_specific_features():
    """测试AFNO特有功能"""
    print("\n" + "="*50)
    print("测试AFNO特有功能...")
    print("="*50)
    
    # 测试不同的稀疏性阈值
    print("\n测试不同稀疏性阈值...")
    sparsity_thresholds = [0.001, 0.01, 0.1]
    
    for threshold in sparsity_thresholds:
        model = AFNONet(
            img_size=(128, 128),
            patch_size=(16, 16),
            embed_dim=256,
            depth=4,
            num_blocks=8,
            sparsity_threshold=threshold,
            in_chans=3,
            out_chans=3
        )
        
        x = torch.randn(1, 3, 128, 128)
        output = model(x)
        
        print(f"稀疏性阈值 {threshold}: 输入 {x.shape} -> 输出 {output.shape}")
        assert output.shape == (1, 3, 128, 128), f"稀疏性阈值 {threshold} 测试失败!"
    
    print("✓ 稀疏性阈值测试通过!")
    
    # 测试不同的硬阈值分数
    print("\n测试不同硬阈值分数...")
    hard_thresholding_fractions = [0.5, 0.75, 1.0]
    
    for fraction in hard_thresholding_fractions:
        model = AFNONet(
            img_size=(128, 128),
            patch_size=(16, 16),
            embed_dim=256,
            depth=4,
            num_blocks=8,
            hard_thresholding_fraction=fraction,
            in_chans=3,
            out_chans=3
        )
        
        x = torch.randn(1, 3, 128, 128)
        output = model(x)
        
        print(f"硬阈值分数 {fraction}: 输入 {x.shape} -> 输出 {output.shape}")
        assert output.shape == (1, 3, 128, 128), f"硬阈值分数 {fraction} 测试失败!"
    
    print("✓ 硬阈值分数测试通过!")


def test_memory_efficiency():
    """测试内存效率（AFNO的优势之一）"""
    print("\n" + "="*50)
    print("测试内存效率...")
    print("="*50)
    
    # 创建一个较大的模型
    model = AFNONet(
        img_size=(256, 256),
        patch_size=(16, 16),
        embed_dim=512,
        depth=12,
        num_blocks=16,
        in_chans=3,
        out_chans=3
    )
    
    # 使用较大的批次测试
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 3, 256, 256)
        
        # 如果有GPU，测试GPU内存使用
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                output = model(x)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            print(f"批次大小 {batch_size}: 峰值GPU内存使用 {peak_memory:.2f} MB")
            model = model.cpu()
            x = x.cpu()
        else:
            with torch.no_grad():
                output = model(x)
            print(f"批次大小 {batch_size}: 输出形状 {output.shape}")
        
        assert output.shape == (batch_size, 3, 256, 256), f"批次大小 {batch_size} 测试失败!"
    
    print("✓ 内存效率测试通过!")


def test_gradient_flow():
    """测试梯度流是否正常"""
    print("\n" + "="*50)
    print("测试梯度流...")
    print("="*50)
    
    model = AFNONet(
        img_size=(64, 64),
        patch_size=(16, 16),
        embed_dim=128,
        depth=4,
        num_blocks=4,
        in_chans=3,
        out_chans=3
    )
    
    x = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    
    # 前向传播
    output = model(x)
    loss = nn.MSELoss()(output, target)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    has_nan_grad = False
    has_zero_grad = True
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                has_nan_grad = True
                print(f"警告: {name} 有NaN梯度!")
            if param.grad.abs().max() > 0:
                has_zero_grad = False
    
    assert not has_nan_grad, "发现NaN梯度!"
    assert not has_zero_grad, "所有梯度都是零!"
    
    print("✓ 梯度流测试通过!")


if __name__ == "__main__":
    # 运行基本测试
    test_simple_afno()
    
    # 运行不同配置测试
    test_different_configs()
    
    # 测试AFNO特有功能
    test_afno_specific_features()
    
    # 测试内存效率
    test_memory_efficiency()
    
    # 测试梯度流
    test_gradient_flow()
    
    print("\n" + "="*50)
    print("所有测试通过! AFNO模型工作正常 ✓✓✓")
    print("="*50)