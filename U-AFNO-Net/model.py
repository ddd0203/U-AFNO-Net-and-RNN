import numpy as np
import torch
from torch import nn
import h5py
from unet_modules import (UNetEncoder, UNetDecoder, UNetEncoderCompact, 
                         UNetDecoderCompact, DoubleConv, Down, Up, OutConv)
from afno2d_modules import AFNONet
from vit_modules import ViTNet
from swin_modules import SwinNet


# =============== AFNO 中间网络 ===============

class Mid_AFNONet(nn.Module):
    """使用完整AFNO架构的中间网络"""

    def __init__(self, channel_in, N_T, img_size=(16, 16),
                 patch_size=(4, 4), embed_dim=None,
                 mlp_ratio=4.0, drop_rate=0.1, drop_path_rate=0.1,
                 num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1.0):
        super().__init__()
        self.H, self.W = img_size
        self.patch_size = patch_size

        # 如果没有指定embed_dim，使用channel_in
        if embed_dim is None:
            embed_dim = channel_in

        # 使用完整的AFNO网络
        self.afno_net = AFNONet(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=channel_in,
            out_chans=channel_in,  # 输出通道数与输入相同
            embed_dim=embed_dim,
            depth=N_T,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction
        )

    def forward(self, x):
        """
        使用完整AFNO架构处理:
        Input shape: (B, T*C, H, W)
        Output shape: (B, T*C, H, W)
        """
        return self.afno_net(x)

class Mid_ViTNet(nn.Module):
    """使用完整AFNO架构的中间网络"""

    def __init__(self, channel_in, N_T, img_size=(16, 16),
                 patch_size=(4, 4), embed_dim=None,
                 mlp_ratio=4.0, drop_rate=0.1, drop_path_rate=0.1,
                 num_blocks=8):
        super().__init__()
        self.H, self.W = img_size
        self.patch_size = patch_size

        # 如果没有指定embed_dim，使用channel_in
        if embed_dim is None:
            embed_dim = channel_in

        # 使用完整的AFNO网络
        self.vit_net = ViTNet(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=channel_in,
            out_chans=channel_in,  # 输出通道数与输入相同
            embed_dim=embed_dim,
            depth=N_T,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            num_blocks=num_blocks
        )

    def forward(self, x):
        """
        使用完整AFNO架构处理:
        Input shape: (B, T*C, H, W)
        Output shape: (B, T*C, H, W)
        """
        return self.vit_net(x)

class Mid_SwinNet(nn.Module):
    """使用完整AFNO架构的中间网络"""

    def __init__(self, channel_in, N_T, img_size=(16, 16),
                 patch_size=(4, 4), embed_dim=None,
                 mlp_ratio=4.0, drop_rate=0.1, drop_path_rate=0.1,
                 num_blocks=8,window_size=8):
        super().__init__()
        self.H, self.W = img_size
        self.patch_size = patch_size

        # 如果没有指定embed_dim，使用channel_in
        if embed_dim is None:
            embed_dim = channel_in

        # 使用完整的AFNO网络
        self.swin_net = SwinNet(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=channel_in,
            out_chans=channel_in,  # 输出通道数与输入相同
            embed_dim=embed_dim,
            depth=N_T,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            num_blocks=num_blocks,
            window_size=window_size
        )

    def forward(self, x):
        """
        使用完整AFNO架构处理:
        Input shape: (B, T*C, H, W)
        Output shape: (B, T*C, H, W)
        """
        return self.swin_net(x)


# =============== UNetAFNO 主模型 ===============

class UNetAFNO(nn.Module):
    def __init__(self, shape_in, hid_S=16, N_T=8,
                 img_size=(32, 32),
                 # 架构选择参数
                 mask_path='mask.h5',
                 input_steps=4,
                 output_steps=1,
                 use_unet=True,  # 是否使用UNet编码器解码器
                 use_compact_unet=False,  # 是否使用紧凑版UNet（用于小特征图）
                 bilinear=False,  # UNet是否使用双线性插值
                 # 中间网络选择参数
                 middle_type='afno',  # 中间网络类型：'afno', 'vit', 'swin'
                 # AFNO参数
                 patch_size=(4, 4), embed_dim=None,
                 mlp_ratio=4.0, drop_rate=0.1, drop_path_rate=0.1, window_size=8,
                 num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1.0):
        super().__init__()
        T_in, C, H, W = shape_in

        self.input_steps = input_steps
        self.output_steps = output_steps
        self.input_channels = C

        if T_in != input_steps:
          raise ValueError(f"shape_in[0]({T_in}) 与 input_steps({input_steps}) 不匹配")


        self.use_unet = use_unet
        self.use_compact_unet = use_compact_unet
        self.middle_type = middle_type
        self.mask = self._load_mask(mask_path)

        encoder_input_channels = C * input_steps
        decoder_output_channels = C * output_steps

        # 计算中间网络的通道数
        if use_unet and use_compact_unet:
            middle_channels = hid_S * 8  # 紧凑版最深层通道数
        elif use_unet:
            middle_channels = hid_S * 16  # 标准UNet最深层通道数
        else:
            middle_channels = hid_S

        # 选择中间网络类型
        if middle_type == 'afno':
            self.hid = Mid_AFNONet(
                channel_in=middle_channels,
                N_T=N_T,
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                num_blocks=num_blocks,
                sparsity_threshold=sparsity_threshold,
                hard_thresholding_fraction=hard_thresholding_fraction
            )
            print(f"使用AFNO中间网络，通道数: {middle_channels}")

        if middle_type == 'vit':
            self.hid = Mid_ViTNet(
                channel_in=middle_channels,
                N_T=N_T,
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                num_blocks=num_blocks
            )
            print(f"使用ViT中间网络，通道数: {middle_channels}")
        if middle_type == 'swin':
            self.hid = Mid_SwinNet(
                channel_in=middle_channels,
                N_T=N_T,
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                num_blocks=num_blocks,
                window_size=window_size
            )
            print(f"使用Swin中间网络，通道数: {middle_channels}")

        # 选择编码器和解码器架构
        if use_unet:
            if use_compact_unet:
                # 使用紧凑版UNet
                self.enc = UNetEncoderCompact(encoder_input_channels, hid_S, bilinear=bilinear)
                self.dec = UNetDecoderCompact(hid_S, decoder_output_channels, bilinear=bilinear)
                print("使用紧凑版UNet编码器-解码器架构")
            else:
                # 使用标准UNet
                self.enc = UNetEncoder(encoder_input_channels, hid_S, bilinear=bilinear)
                self.dec = UNetDecoder(hid_S, decoder_output_channels, bilinear=bilinear)
                print("使用标准UNet编码器-解码器架构")
    def _load_mask(self, path):
      with h5py.File(path, 'r') as f:
        mask = f['fields'][:]
      mask_tensor = torch.from_numpy(mask.astype(np.float32))
      expanded_mask = mask_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)

      return expanded_mask

    def _apply_mask(self, x):
      return x * self.mask.to(x.device)


    def forward(self, x_raw):
        B, T_in, C, H, W = x_raw.shape
        assert T_in == self.input_steps, f"输入时间步长不匹配: 期望{self.input_steps}, 得到{T_in}"
        x = x_raw.view(B, T_in * C, H, W)
        #x = x_raw.view(B, T_in, C, H, W)

        if self.use_unet:
            # UNet编码器：输出深层特征和skip connections
            embed, skip_connections = self.enc(x)
            _, C_, H_, W_ = embed.shape

            hid = self.hid(embed)

            # 中间网络处理
            #z = embed.view(B, T, C_, H_, W_)
            #hid = self.hid(z.reshape(B, T * C_, H_, W_))

            # UNet解码器：使用skip connections
            #hid = hid.reshape(B * T, C_, H_, W_)
            Y = self.dec(hid, skip_connections)

        return self._apply_mask(Y.reshape(B, self.output_steps, self.input_channels, H, W))



# =============== 纯AFNO模型（用于消融实验） ===============

class PureAFNO(nn.Module):
    """纯AFNO模型，直接基于AFNONet"""
    
    def __init__(self, shape_in, input_steps=4, output_steps=1, mask_path='mask.h5',
           img_size=(64, 64), patch_size=(4, 4), 
           embed_dim=256, depth=12, mlp_ratio=4.0, drop_rate=0.1, 
           drop_path_rate=0.1, num_blocks=8, sparsity_threshold=0.01, 
           hard_thresholding_fraction=1.0):
        super().__init__()
        T_in, C, H, W = shape_in


        self.input_steps = input_steps
        self.output_steps = output_steps
        self.input_channels = C

        if T_in != input_steps:
          raise ValueError(f"shape_in[0]({T_in}) 与 input_steps({input_steps}) 不匹配")

        self.mask = self._load_mask(mask_path)

        input_channels = C * input_steps
        output_channels = C * output_steps
        
        # AFNO核心网络
        self.afno_net = AFNONet(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=input_channels,
            out_chans=output_channels,
            embed_dim=embed_dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction
        )
        
        print(f"创建纯AFNO模型：输入形状{shape_in} -> 嵌入维度{embed_dim}")

    def _load_mask(self, path):
      with h5py.File(path, 'r') as f:
        mask = f['fields'][:]
      mask_tensor = torch.from_numpy(mask.astype(np.float32))
      expanded_mask = mask_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)

      return expanded_mask

    def _apply_mask(self, x):
      return x * self.mask.to(x.device)
        
    def forward(self, x):
        B, T_in, C, H, W = x.shape
        
        # 合并时间和通道维度
        x = x.view(B, T_in * C, H, W)
        
        # AFNO处理
        x = self.afno_net(x)

        
        # 恢复时间维度
        x = x.view(B, self.output_steps, self.input_channels, H, W)
        
        
        return self._apply_mask(x)



class PureUNet(nn.Module):
    """纯UNet模型，与UNetEncoderCompact+UNetDecoderCompact结构完全一致"""
    
    def __init__(self, shape_in, input_steps=4, output_steps=1, mask_path='mask.h5', hid_S=32, bilinear=False, **kwargs):
        super().__init__()
        T_in, C, H, W = shape_in

        self.input_steps = input_steps
        self.output_steps = output_steps
        self.input_channels = C

        self.mask = self._load_mask(mask_path)

        if T_in != input_steps:
          raise ValueError(f"shape_in[0]({T_in}) 与 input_steps({input_steps}) 不匹配")
        
        encoder_input_channels = C * input_steps
        decoder_output_channels = C * output_steps
         
        self.inc = DoubleConv(encoder_input_channels, hid_S)
        
        self.down1 = Down(hid_S, hid_S * 2)        
        self.down2 = Down(hid_S * 2, hid_S * 4)   
        
        factor = 2 if bilinear else 1
        self.down3 = Down(hid_S * 4, hid_S * 8 // factor)  
        
        
        self.up1 = Up(hid_S * 8, hid_S * 4 // factor, bilinear)  
        self.up2 = Up(hid_S * 4, hid_S * 2 // factor, bilinear)  
        self.up3 = Up(hid_S * 2, hid_S, bilinear)                
        
        
        self.outc = OutConv(hid_S, decoder_output_channels)

    def _load_mask(self, path):
      with h5py.File(path, 'r') as f:
        mask = f['fields'][:]
      mask_tensor = torch.from_numpy(mask.astype(np.float32))
      expanded_mask = mask_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)

      return expanded_mask

    def _apply_mask(self, x):
      return x * self.mask.to(x.device)
        
        
    def forward(self, x):
        B, T_in, C, H, W = x.shape
        assert T_in == self.input_steps, f"输入时间步长不匹配: 期望{self.input_steps}, 得到{T_in}"
        
        x = x.view(B, T_in * C, H, W)
        
        x1 = self.inc(x)        
        x2 = self.down1(x1)    
        x3 = self.down2(x2)     
        x4 = self.down3(x3)     
        
        
        x = self.up1(x4, x3)    
        x = self.up2(x, x2)     
        x = self.up3(x, x1)     
        
        # 输出层
        x = self.outc(x)        
        
        # 恢复时间维度：(B, T*C, H, W) -> (B, T, C, H, W)
        x = x.view(B, self.output_steps, self.input_channels, H, W)
        
        return self._apply_mask(x)


# =============== 便捷的模型构造函数（扩展版） ===============

def create_unet_afno(shape_in, **kwargs):
    """创建使用标准UNet+AFNO的SimVP模型"""
    return UNetAFNO(shape_in, use_unet=True, use_compact_unet=False, 
                middle_type='vit', **kwargs)

def create_compact_unet_afno(shape_in, **kwargs):
    """创建使用紧凑UNet+AFNO的SimVP模型"""
    return UNetAFNO(shape_in, use_unet=True, use_compact_unet=True, 
                middle_type='afno', **kwargs)

def create_pure_afno(shape_in, **kwargs):
    """创建纯AFNO模型"""
    return PureAFNO(shape_in, **kwargs)


def create_pure_unet(shape_in, **kwargs):
    """创建纯UNet模型"""
    return PureUNet(shape_in, **kwargs)


# =============== 统一的模型工厂函数 ===============

def create_model(model_name, shape_in, **kwargs):
    """
    统一的模型创建函数，用于消融实验
    
    Args:
        model_name: 模型名称，支持以下选项：
            - 'simvp_unet_afno': 标准UNet + AFNO
            - 'simvp_compact_unet_afno': 紧凑UNet + AFNO  
            - 'simvp_unet_only': UNet + UNet中间网络
            - 'simvp_cnn_baseline': UNet + CNN中间网络
            - 'simvp_original_afno': 原始架构 + AFNO
            - 'pure_afno': 纯AFNO模型
            - 'pure_unet': 纯UNet模型
        shape_in: 输入形状 (T, C, H, W)
        **kwargs: 其他模型参数
    """
    
    model_factory = {
        'unet_afno': create_unet_afno,
        'compact_unet_afno': create_compact_unet_afno,
        'pure_afno': create_pure_afno,
        'pure_unet': create_pure_unet,
    }
    
    if model_name not in model_factory:
        raise ValueError(f"不支持的模型名称: {model_name}. 支持的模型: {list(model_factory.keys())}")
    
    model = model_factory[model_name](shape_in, **kwargs)
    print(f"✓ 成功创建模型: {model_name}")
    
    return model


if __name__ == '__main__':
    # 测试所有模型类型
    input_shape = (4, 4, 256, 256)  # (T, C, H, W)
    input_data = torch.randn(2, *input_shape)  # (B, T, C, H, W)
    
    print("=" * 80)
    print("消融实验模型测试")
    print("=" * 80)

    
    # unet_afno测试参数
    test_params = {
        'hid_S': 64,
        'N_T': 8,
        'img_size': (32, 32),
        'patch_size': (1, 1),
        'input_steps': 4,
        'output_steps': 1,
        'mask_path': './data/256x256_h5new/mask.h5',

        'embed_dim': 128,
      
    }
    

    # afno测试参数
    '''
    test_params = {
        'depth': 8,
        'img_size': (64, 64),
        'patch_size': (8, 8),
        'embed_dim': 128,
      
    }
    '''

    # unet测试参数
    '''
    test_params = {
        'depth': 3,
        'hid_channels': 16,
      
    }
    '''
    
    
    # 所有模型类型
    model_types = [
        'compact_unet_afno'
        
    ]
    
    for model_type in model_types:
        try:
            print(f"\n测试模型: {model_type}")
            print("-" * 40)
            
            model = create_model(model_type, input_shape, **test_params)
            
            # 前向传播测试
            with torch.no_grad():
                output = model(input_data)
            
            # 参数统计
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"输入形状: {input_data.shape}")
            print(f"输出形状: {output.shape}")
            print(f"总参数量: {total_params:,}")
            print(f"可训练参数: {trainable_params:,}")
            print(f"内存占用: {total_params * 4 / 1024**2:.2f} MB")
            
        except Exception as e:
            print(f"❌ 模型 {model_type} 测试失败: {str(e)}")
    
    print("\n" + "=" * 80)
    print("模型测试完成！")
    print("=" * 80)