import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from convlstm_modules import ConvLSTMCell
from predrnn_modules import SpatioTemporalLSTMCell
from mim_modules import MIMBlock, MIMN
from predrnnv2_modules import SpatioTemporalLSTMCellv2
from utils import reshape_patch, reshape_patch_back


class ConvLSTM(nn.Module):
    """ConvLSTM模型 - 简洁版本"""

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(ConvLSTM, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.patch_size = configs.patch_size
        self.patch_height = H // configs.patch_size
        self.patch_width = W // configs.patch_size

        # 加载mask
        self.mask = self._load_mask(configs.mask_path) if hasattr(configs, 'mask_path') else None

        # MSE损失函数
        self.MSE_criterion = nn.MSELoss()

        # 构建ConvLSTM层
        cell_list = []
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channel, num_hidden[i], self.patch_height, self.patch_width,
                             configs.filter_size, configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)

        # 输出层
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)


    def _load_mask(self, path):
      with h5py.File(path, 'r') as f:
        mask = f['fields'][:]
      mask_tensor = torch.from_numpy(mask.astype(np.float32))
      expanded_mask = mask_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)

      return expanded_mask

    def _apply_mask(self, x):
      if self.mask is not None:
        return x * self.mask.to(x.device)

    def forward(self, frames_tensor, mask_true, **kwargs):
        """前向传播

        Args:
            frames_tensor: (batch, length, height, width, channel)
            mask_true: scheduled sampling的mask
        """
        # 将输入从 [batch, length, channel, height, width] 转换为 [batch, length, height, width, channel]
        frames_tensor = frames_tensor.permute(0, 1, 3, 4, 2).contiguous()
        #print(f"Debug: frames_tensor shape = {frames_tensor.shape}")
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        device = frames.device

        # 使用工具函数转换为patch形式
        frames_patch = reshape_patch(frames_tensor, self.patch_size)
        frames_patch = frames_patch.permute(0, 1, 4, 2, 3).contiguous()

        # 初始化隐藏状态
        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], self.patch_height, self.patch_width]).to(device)
            h_t.append(zeros)
            c_t.append(zeros)

        # 时序循环
        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            # 反向计划采样
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames_patch[:, t]
                else:
                    net = mask_true[:, t - 1] * frames_patch[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                # 标准计划采样
                if t < self.configs.pre_seq_length:
                    net = frames_patch[:, t]
                else:
                    net = mask_true[:, t - self.configs.pre_seq_length] * frames_patch[:, t] + \
                          (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

            # 通过ConvLSTM层
            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            # 生成输出
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()

        # 使用工具函数还原patch
        next_frames = reshape_patch_back(next_frames, self.patch_size)

        next_frames = next_frames.permute(0, 1, 4, 2, 3).contiguous()

        # 应用mask
        next_frames = self._apply_mask(next_frames)

        # 计算损失
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None

        return next_frames, loss


class PredRNN(nn.Module):
    """PredRNN模型 - 预测循环神经网络"""

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(PredRNN, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.patch_size = configs.patch_size
        self.patch_height = H // configs.patch_size
        self.patch_width = W // configs.patch_size

        # 加载mask
        self.mask = self._load_mask(configs.mask_path) if hasattr(configs, 'mask_path') else None

        # MSE损失函数
        self.MSE_criterion = nn.MSELoss()

        # 构建SpatioTemporalLSTM层
        cell_list = []
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], self.patch_height, self.patch_width,
                                       configs.filter_size, configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)

        # 输出层
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def _load_mask(self, path):
        with h5py.File(path, 'r') as f:
            mask = f['fields'][:]
        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        expanded_mask = mask_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return expanded_mask

    def _apply_mask(self, x):
        if self.mask is not None:
            return x * self.mask.to(x.device)
        return x

    def forward(self, frames_tensor, mask_true, **kwargs):
        """前向传播

        Args:
            frames_tensor: (batch, length, height, width, channel)
            mask_true: scheduled sampling的mask
        """
        # 将输入从 [batch, length, channel, height, width] 转换为 [batch, length, height, width, channel]
        frames_tensor = frames_tensor.permute(0, 1, 3, 4, 2).contiguous()
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        device = frames.device

        # 使用工具函数转换为patch形式
        frames_patch = reshape_patch(frames_tensor, self.patch_size)
        frames_patch = frames_patch.permute(0, 1, 4, 2, 3).contiguous()

        # 初始化隐藏状态
        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], self.patch_height, self.patch_width]).to(device)
            h_t.append(zeros)
            c_t.append(zeros)

        # 初始化memory
        memory = torch.zeros([batch, self.num_hidden[0], self.patch_height, self.patch_width]).to(device)

        # 时序循环
        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            # 反向计划采样
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames_patch[:, t]
                else:
                    net = mask_true[:, t - 1] * frames_patch[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                # 标准计划采样
                if t < self.configs.pre_seq_length:
                    net = frames_patch[:, t]
                else:
                    net = mask_true[:, t - self.configs.pre_seq_length] * frames_patch[:, t] + \
                          (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

            # 通过SpatioTemporalLSTM层
            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            # 生成输出
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()

        # 使用工具函数还原patch
        next_frames = reshape_patch_back(next_frames, self.patch_size)
        next_frames = next_frames.permute(0, 1, 4, 2, 3).contiguous()

        # 应用mask
        next_frames = self._apply_mask(next_frames)

        # 计算损失
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None

        return next_frames, loss




class MIM(nn.Module):
    """MIM模型 - Memory In Memory"""

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(MIM, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.patch_size = configs.patch_size
        self.patch_height = H // configs.patch_size
        self.patch_width = W // configs.patch_size

        # 加载mask
        self.mask = self._load_mask(configs.mask_path) if hasattr(configs, 'mask_path') else None

        # MSE损失函数
        self.MSE_criterion = nn.MSELoss()

        # 构建层：第一层使用SpatioTemporalLSTMCell，其他层使用MIMBlock
        stlstm_layer = []
        stlstm_layer_diff = []
        
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            if i < 1:
                # 第一层使用SpatioTemporalLSTMCell
                stlstm_layer.append(
                    SpatioTemporalLSTMCell(in_channel, num_hidden[i], self.patch_height, self.patch_width,
                                           configs.filter_size, configs.stride, configs.layer_norm))
            else:
                # 其他层使用MIMBlock
                stlstm_layer.append(
                    MIMBlock(in_channel, num_hidden[i], self.patch_height, self.patch_width,
                             configs.filter_size, configs.stride, configs.layer_norm))
        
        # 构建MIMN层用于处理层间差分
        for i in range(num_layers - 1):
            stlstm_layer_diff.append(
                MIMN(num_hidden[i], num_hidden[i + 1], self.patch_height, self.patch_width,
                     configs.filter_size, configs.stride, configs.layer_norm))
            
        self.stlstm_layer = nn.ModuleList(stlstm_layer)
        self.stlstm_layer_diff = nn.ModuleList(stlstm_layer_diff)

        # 输出层
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def _load_mask(self, path):
        with h5py.File(path, 'r') as f:
            mask = f['fields'][:]
        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        expanded_mask = mask_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return expanded_mask

    def _apply_mask(self, x):
        if self.mask is not None:
            return x * self.mask.to(x.device)
        return x

    def forward(self, frames_tensor, mask_true, **kwargs):
        """前向传播"""
        frames_tensor = frames_tensor.permute(0, 1, 3, 4, 2).contiguous()
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        device = frames.device

        frames_patch = reshape_patch(frames_tensor, self.patch_size)
        frames_patch = frames_patch.permute(0, 1, 4, 2, 3).contiguous()

        next_frames = []
        h_t = []
        c_t = []
        hidden_state_diff = []
        cell_state_diff = []

        # 初始化状态
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], self.patch_height, self.patch_width]).to(device)
            h_t.append(zeros)
            c_t.append(zeros)
            hidden_state_diff.append(None)
            cell_state_diff.append(None)

        # 初始化共享的st_memory
        st_memory = torch.zeros([batch, self.num_hidden[0], self.patch_height, self.patch_width]).to(device)

        # 时序循环
        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            # 反向计划采样
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames_patch[:, t]
                else:
                    net = mask_true[:, t - 1] * frames_patch[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.pre_seq_length:
                    net = frames_patch[:, t]
                else:
                    net = mask_true[:, t - self.configs.pre_seq_length] * frames_patch[:, t] + \
                          (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

            # 保存前一时刻的h_t[0]用于计算差分
            preh = h_t[0]

            # 第一层：使用SpatioTemporalLSTMCell
            h_t[0], c_t[0], st_memory = self.stlstm_layer[0](net, h_t[0], c_t[0], st_memory)

            # 其他层：使用MIMBlock和MIMN
            for i in range(1, self.num_layers):
                if t > 0:
                    if i == 1:
                        # 第二层：计算输入差分 (h_t[0] - preh)
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](
                            h_t[i - 1] - preh, hidden_state_diff[i - 1], cell_state_diff[i - 1])
                    else:
                        # 其他层：使用上一层的差分输出
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](
                            hidden_state_diff[i - 2], hidden_state_diff[i - 1], cell_state_diff[i - 1])
                else:
                    # t=0时，初始化MIMN
                    self.stlstm_layer_diff[i - 1](torch.zeros_like(h_t[i - 1]), None, None)

                # 通过MIMBlock
                h_t[i], c_t[i], st_memory = self.stlstm_layer[i](
                    h_t[i - 1], hidden_state_diff[i - 1], h_t[i], c_t[i], st_memory)

            # 生成输出
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        next_frames = reshape_patch_back(next_frames, self.patch_size)
        next_frames = next_frames.permute(0, 1, 4, 2, 3).contiguous()
        
        # 应用mask
        next_frames = self._apply_mask(next_frames)

        # 计算损失
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None

        return next_frames, loss

class PredRNNv2(nn.Module):
    """PredRNNv2 Model

    Implementation of `PredRNN: A Recurrent Neural Network for Spatiotemporal
    Predictive Learning <https://arxiv.org/abs/2103.09504v4>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(PredRNNv2, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.patch_size = configs.patch_size
        self.patch_height = H // configs.patch_size
        self.patch_width = W // configs.patch_size
        self.mask = self._load_mask(configs.mask_path) if hasattr(configs, 'mask_path') else None

        self.MSE_criterion = nn.MSELoss()
        cell_list = []

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCellv2(in_channel, num_hidden[i], self.patch_height, self.patch_width,
                                         configs.filter_size, configs.stride, configs.layer_norm))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=1,
                                   stride=1, padding=0, bias=False)
        # shared adapter
        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Conv2d(
            adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)

    def _load_mask(self, path):
        with h5py.File(path, 'r') as f:
            mask = f['fields'][:]
        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        expanded_mask = mask_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return expanded_mask

    def _apply_mask(self, x):
        if self.mask is not None:
            return x * self.mask.to(x.device)

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames_tensor = frames_tensor.permute(0, 1, 3, 4, 2).contiguous()
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        device = frames.device
        frames_patch = reshape_patch(frames_tensor, self.patch_size)
        frames_patch = frames_patch.permute(0, 1, 4, 2, 3).contiguous()


        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []

        decouple_loss = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch, self.num_hidden[i], self.patch_height, self.patch_width]).to(device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros(
            [batch, self.num_hidden[0], self.patch_height, self.patch_width]).to(device)

        for t in range(self.configs.total_length - 1):

            if self.configs.reverse_scheduled_sampling == 1:
                # reverse schedule sampling
                if t == 0:
                    net = frames_patch[:, t]
                else:
                    net = mask_true[:, t - 1] * frames_patch[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                # schedule sampling
                if t < self.configs.pre_seq_length:
                    net = frames_patch[:, t]
                else:
                    net = mask_true[:, t - self.configs.pre_seq_length] * frames_patch[:, t] + \
                          (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

            h_t[0], c_t[0], memory, delta_c, delta_m = \
                self.cell_list[0](net, h_t[0], c_t[0], memory)
            delta_c_list[0] = F.normalize(
                self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(
                self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = \
                    self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(
                    self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(
                    self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()

        next_frames = reshape_patch_back(next_frames, self.patch_size)
        next_frames = next_frames.permute(0, 1, 4, 2, 3).contiguous()
        
        next_frames = self._apply_mask(next_frames)


        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None

        return next_frames, loss




def create_model(model_name, num_layers, num_hidden, configs, **kwargs):
    """模型工厂函数"""
    model_dict = {
        'convlstm': ConvLSTM,
        'mim': MIM,
        'predrnn': PredRNN,
        'predrnnv2': PredRNNv2,
    }
    
    if model_name.lower() not in model_dict:
        raise ValueError(f"Unsupported model: {model_name}. Available models: {list(model_dict.keys())}")
    
    model_class = model_dict[model_name.lower()]
    return model_class(num_layers, num_hidden, configs, **kwargs)


if __name__ == '__main__':
    import os
    
    # 快速创建mask文件
    def quick_create_mask():
        os.makedirs('./data', exist_ok=True)
        mask_path = './data/test_mask.h5'
        
        # 创建简单mask
        mask_data = np.ones((64, 64), dtype=np.float32)
        with h5py.File(mask_path, 'w') as f:
            f.create_dataset('fields', data=mask_data)
        return mask_path

    # 创建配置类
    class TestConfig:
        def __init__(self):
            self.in_shape = (20, 1, 64, 64)  # (T, C, H, W)
            self.patch_size = 8
            self.filter_size = 5
            self.stride = 1
            self.layer_norm = True
            self.pre_seq_length = 10
            self.aft_seq_length = 10
            self.total_length = 20
            self.reverse_scheduled_sampling = 0
            self.mask_path = quick_create_mask()  # 创建mask文件
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建测试配置
    configs = TestConfig()

    # 创建模型
    num_layers = 4
    num_hidden = [64, 64, 64, 64]
    
    print("=== 测试所有模型 ===\n")
    
    # 移动到设备
    device = torch.device(configs.device)
    
    # 创建测试数据
    batch_size = 2
    frames = torch.randn(batch_size, configs.total_length, 1, 64, 64).to(device)
    mask = torch.ones(batch_size, configs.total_length - 1, 
                      64 // configs.patch_size, 64 // configs.patch_size, 
                      configs.patch_size ** 2 * 1).to(device)

    # 测试所有模型
    models_to_test = ['convlstm', 'mim', 'predrnn', 'predrnnv2']
    
    for model_name in models_to_test:
        print(f"{'='*50}")
        print(f"测试 {model_name.upper()} 模型")
        print(f"{'='*50}")
        
        try:
            # 使用工厂函数创建模型
            model = create_model(model_name, num_layers, num_hidden, configs)
            model = model.to(device)
            
            print(f"✓ 创建{model_name.upper()}模型成功")
            print(f"  - 层数: {num_layers}")
            print(f"  - 隐藏单元: {num_hidden}")
            print(f"  - Patch大小: {configs.patch_size}")
            
            # 打印模型结构信息
            if model_name == 'mim':
                print(f"  - 第1层: SpatioTemporalLSTMCell")
                print(f"  - 第2-{num_layers}层: MIMBlock + MIMN")
            
            # 前向传播测试
            with torch.no_grad():
                output, loss = model(frames, mask)
            
            print(f"\n前向传播结果:")
            print(f"  - 输出形状: {output.shape}")
            print(f"  - 损失值: {loss.item():.4f}")
            
            # 参数统计
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"\n模型参数统计:")
            print(f"  - 总参数量: {total_params:,}")
            print(f"  - 可训练参数: {trainable_params:,}")
            print(f"  - 内存占用: {total_params * 4 / 1024 ** 2:.2f} MB")
            
            print(f"\n✅ {model_name.upper()}测试完成！\n")
            
        except Exception as e:
            print(f"❌ {model_name.upper()}测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 模型比较
    print(f"\n{'='*50}")
    print("=== 模型比较总结 ===")
    print(f"{'='*50}")
    
    comparison_results = []
    for model_name in models_to_test:
        try:
            model = create_model(model_name, num_layers, num_hidden, configs)
            model = model.to(device)
            
            with torch.no_grad():
                output, loss = model(frames, mask)
            
            total_params = sum(p.numel() for p in model.parameters())
            
            comparison_results.append({
                'name': model_name.upper(),
                'params': total_params,
                'loss': loss.item()
            })
        except:
            pass
    
    if comparison_results:
        print(f"{'模型':<10} {'参数量':<15} {'损失值':<10}")
        print("-" * 35)
        for result in comparison_results:
            print(f"{result['name']:<10} {result['params']:<15,} {result['loss']:<10.4f}")
    
    print("\n✅ 所有测试完成！")