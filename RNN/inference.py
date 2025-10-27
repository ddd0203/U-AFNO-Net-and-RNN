import os
import argparse
import h5py
import numpy as np
import torch
from tqdm import tqdm
import json
from datetime import datetime

from convlstm_model import create_model
from utils import set_seed, print_log


class ConvLSTMInference:
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()

        # 设置随机种子
        set_seed(args.seed)

        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)

        # 加载模型
        self._load_model()

        # 加载测试数据
        self._load_test_data()

        self.dt = 2

    def _setup_device(self):
        """设置设备"""
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.args.gpu}')
            print(f'使用GPU: {self.args.gpu}')
        else:
            device = torch.device('cpu')
            print('使用CPU')
        return device

    def _load_model(self):
        """加载训练好的模型"""
        print("加载模型...")

        # 创建模型
        self.model = create_model(
            model_name=self.args.model,
            num_layers=self.args.num_layers,
            num_hidden=self.args.num_hidden,
            configs=self.args
        ).to(self.device)

        # 加载权重
        checkpoint = torch.load(self.args.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        print(f"✓ 模型加载成功: {self.args.model} - {self.args.model_path}")

    def _load_test_data(self):
        """加载测试数据"""
        print("加载测试数据...")

        self.test_file = h5py.File(self.args.test_data_path, 'r')
        self.test_data = self.test_file['fields']  # (C, T, H, W)

        self.n_channels, self.n_timesteps, self.height, self.width = self.test_data.shape
        print(f"✓ 数据形状: {self.test_data.shape}")
        print(f"通道数: {self.n_channels}, 时间步数: {self.n_timesteps}")

    def _prepare_initial_input(self, start_time):
        """
        准备初始输入数据：连续5个时间步（4个输入+1个目标）
        Args:
            start_time: 起始时间索引
        Returns:
            input_tensor: (1, 5, C, H, W) - 5个连续时间步的数据
        """
        # 检查是否有足够的数据
        if start_time < self.args.pre_seq_length - 1:
            raise ValueError(f"起始时间 {start_time} 不足，需要至少 {self.args.pre_seq_length - 1} 个历史时间步")

        # 获取连续5个时间步的数据
        time_indices = []
        for i in range(self.args.pre_seq_length):
            time_idx = start_time - (self.args.pre_seq_length - 1 - i) * self.dt
            time_indices.append(time_idx)
        

        # 提取数据: (C, 4, H, W)
        input_data = self.test_data[:, time_indices, :, :]
        
        # 转换为tensor并调整维度: (C, 5, H, W) -> (5, C, H, W) -> (1, 5, C, H, W)
        input_tensor = torch.from_numpy(input_data).float()
        # 创建与新位置匹配的全零切片
        zeros_channel = torch.zeros_like(input_tensor[:, :1])
        # 在第二维度拼接
        input_tensor = torch.cat([input_tensor, zeros_channel], dim=1)#(C, 5, H, W)
        input_tensor = input_tensor.permute(1, 0, 2, 3)
        input_tensor = input_tensor.unsqueeze(0)

        return input_tensor.to(self.device)

    def _prepare_mask(self, batch_size):
        """准备mask（推理时不使用scheduled sampling）"""
        mask_true = torch.zeros(
            (batch_size, self.args.total_length - 1,
             self.height // self.args.patch_size,
             self.width // self.args.patch_size,
             self.args.patch_size ** 2 * self.n_channels)
        ).to(self.device)
        return mask_true

    def _update_input_with_prediction(self, current_input, prediction):
   
    
        current_input = current_input.contiguous()
        prediction = prediction.contiguous()
    
   
        recent_steps = current_input[:, 1:, :, :, :]  # (1, 4, C, H, W)
        last_prediction = prediction[:, -1:, :, :, :]  # (1, 1, C, H, W)

        recent_without_last = recent_steps[:, :-1, :, :, :]  # (1, 3, C, H, W)

        # 将 last_prediction 替换为新的最后一个时间步
        new_input = torch.cat([recent_without_last, last_prediction], dim=1)  # (1, 4, C, H, W)
        zeros_channel = torch.zeros_like(new_input[:, :1])
        new_input = torch.cat([new_input, zeros_channel], dim=1)

        return new_input.contiguous()

    def autoregressive_forecast(self, start_time, forecast_length):
        """自回归预报 - 真正使用预测值"""
        print(f"开始自回归预报 - 起始时间: {start_time}, 预报长度: {forecast_length}")
        print(f"模型配置: {self.args.pre_seq_length}输入 + {self.args.aft_seq_length}输出")
    
        # 检查是否有足够的历史数据
        if start_time < self.args.pre_seq_length:
          raise ValueError(f"起始时间 {start_time} 不足，需要至少 {self.args.pre_seq_length} 个历史时间步")
    
        # 只在开始时准备初始输入（真实数据）
        # 准备初始输入：连续5个时间步
        current_input = self._prepare_initial_input(start_time)  # (1, 5, C, H, W)
        print(f"初始输入形状: {current_input.shape}")

    
    
        # 存储预报结果
        forecasts = []
        ground_truths = []
    
        # 获取对应的真实值时间步（用于评估）
        actual_forecast_steps = 0
    
        with torch.no_grad():
            for step in tqdm(range(forecast_length), desc="自回归预报中"):
                try:
                    # 准备mask（不使用scheduled sampling）
                    mask_true = self._prepare_mask(current_input.shape[0])
                
                    # 模型预报：输入5帧，输出4帧预测
                    pred_frames, _ = self.model(current_input, mask_true, return_loss=False)
                    # pred_frames shape: (1, 4, C, H, W)
                
                    # 保存最后一帧预测结果（对应start_time+2+step）
                    forecast_frame = pred_frames[:, -1, :, :, :].cpu().numpy()  # (1, C, H, W)
                    forecasts.append(forecast_frame)
                
                    # 获取对应的真实值
                    gt_time = start_time + (step + 1) * self.dt  # 预测的是start_time+2开始的值
                    if gt_time < self.n_timesteps:
                        gt_frame = self.test_data[:, gt_time, :, :]  # (C, H, W)
                        gt_frame = np.expand_dims(gt_frame, 0)  # (1, C, H, W)
                        ground_truths.append(gt_frame)
                        actual_forecast_steps += 1
                    else:
                        print(f"达到数据末尾，实际预测步数: {actual_forecast_steps}")
                        break
                
                    # 更新输入序列：使用预测值替换真实值
                    current_input = self._update_input_with_prediction(current_input, pred_frames)
                    
                except Exception as e:
                    print(f"\n❌ 预报步骤 {step} 出错:")
                    print(f"   当前输入形状: {current_input.shape}")
                    if 'pred_frames' in locals():
                        print(f"   预测输出形状: {pred_frames.shape}")
                    print(f"   错误信息: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    break
    
        # 转换为numpy数组
        if forecasts:
            forecasts = np.concatenate(forecasts, axis=0)  # (actual_steps, C, H, W)
        else:
            forecasts = np.array([])
        
        if ground_truths:
            ground_truths = np.concatenate(ground_truths, axis=0)
        else:
            ground_truths = np.array([])
    
        print(f"✓ 自回归预报完成，实际预测长度: {len(forecasts)}")
    
        return forecasts, ground_truths

    def compute_metrics(self, forecasts, ground_truths):
        """计算评估指标"""
        if len(ground_truths) == 0:
            return {}

        # 确保形状一致
        min_length = min(len(forecasts), len(ground_truths))
        forecasts = forecasts[:min_length]
        ground_truths = ground_truths[:min_length]

        # 计算MSE和MAE
        mse = np.mean((forecasts - ground_truths) ** 2)
        mae = np.mean(np.abs(forecasts - ground_truths))

        # 按变量计算
        var_mse = np.mean((forecasts - ground_truths) ** 2, axis=(0, 2, 3))
        var_mae = np.mean(np.abs(forecasts - ground_truths), axis=(0, 2, 3))

        return {
            'mse': float(mse),
            'mae': float(mae),
            'variable_mse': var_mse.tolist(),
            'variable_mae': var_mae.tolist(),
            'forecast_length': min_length
        }

    def save_results(self, forecasts, ground_truths, start_time, forecast_length, metrics):
        """保存预报结果"""
        output_file = os.path.join(
            self.args.output_dir,
            f"forecast_{self.args.model}_start{start_time:04d}_length{forecast_length:03d}.h5"
        )

        print(f"保存结果: {output_file}")

        with h5py.File(output_file, 'w') as f:
            # 保存预报数据
            forecast_data = forecasts.transpose(1, 0, 2, 3)
            f.create_dataset('forecast', data=forecast_data, compression='gzip')

            # 保存真实值
            if len(ground_truths) > 0:
                gt_data = ground_truths.transpose(1, 0, 2, 3)
                f.create_dataset('ground_truth', data=gt_data, compression='gzip')

            # 保存元数据
            f.attrs['start_time'] = start_time
            f.attrs['forecast_length'] = forecast_length
            f.attrs['model_name'] = self.args.model
            f.attrs['num_layers'] = self.args.num_layers
            f.attrs['num_hidden'] = str(self.args.num_hidden)
            f.attrs['patch_size'] = self.args.patch_size
            f.attrs['timestamp'] = datetime.now().isoformat()
            f.attrs['mse'] = metrics.get('mse', 0.0)
            f.attrs['mae'] = metrics.get('mae', 0.0)

        print(f"✓ 结果已保存")

    def run_inference(self):
        """执行推理"""
        print("=" * 60)
        print(f"开始{self.args.model.upper()}模型预报推理")
        print(f"模型配置: {self.args.num_layers}层, 隐藏单元{self.args.num_hidden}")
        print(f"测试数据: {self.args.test_data_path}")
        print("=" * 60)

        all_metrics = []

        for start_time in self.args.start_times:
            print(f"\n处理起始时间: {start_time}")

            try:
                # 自回归预报
                forecasts, ground_truths = self.autoregressive_forecast(
                    start_time, self.args.forecast_length
                )

                # 计算指标
                metrics = self.compute_metrics(forecasts, ground_truths)
                metrics['start_time'] = start_time
                all_metrics.append(metrics)

                # 打印结果
                if metrics:
                    print(f"MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")

                # 保存结果
                self.save_results(
                    forecasts, ground_truths, start_time,
                    len(forecasts), metrics
                )

            except Exception as e:
                print(f"✗ 处理起始时间 {start_time} 出错: {str(e)}")
                continue

        # 保存整体指标
        if all_metrics:
            metrics_file = os.path.join(self.args.output_dir, f'{self.args.model}_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=2)

            # 计算平均指标
            valid_metrics = [m for m in all_metrics if 'mse' in m]
            if valid_metrics:
                avg_mse = np.mean([m['mse'] for m in valid_metrics])
                avg_mae = np.mean([m['mae'] for m in valid_metrics])
                print(f"\n平均指标 - MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}")

        print("\n" + "=" * 60)
        print("推理完成!")
        print("=" * 60)

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'test_file'):
            self.test_file.close()


def main():
    parser = argparse.ArgumentParser(description='ConvLSTM Weather Forecasting Inference')

    # 核心参数
    parser.add_argument('--model', required=True,
                        choices=['convlstm', 'predrnn', 'predrnnv2', 'mim'],
                        help='模型名称')
    parser.add_argument('--model_path', required=True, help='训练好的模型路径')
    parser.add_argument('--test_data_path', required=True, help='测试数据路径')

    # 预报参数
    parser.add_argument('--start_times', type=int, nargs='+', default=[10, 100, 200],
                        help='预报起始时间列表')
    parser.add_argument('--forecast_length', type=int, default=20, help='预报长度')
    parser.add_argument('--output_dir', default='./inference_results/', help='输出目录')

    # 模型配置参数（需要与训练时保持一致）
    parser.add_argument('--in_shape', type=int, nargs=4, default=[20, 1, 64, 64],
                        help='输入形状 [T, C, H, W]')
    parser.add_argument('--num_layers', type=int, default=4, help='模型层数')
    parser.add_argument('--num_hidden', type=int, nargs='+', default=[64, 64, 64, 64],
                        help='每层隐藏单元数')
    parser.add_argument('--patch_size', type=int, default=8, help='Patch大小')
    parser.add_argument('--filter_size', type=int, default=5, help='卷积核大小')
    parser.add_argument('--stride', type=int, default=1, help='步长')
    parser.add_argument('--layer_norm', type=bool, default=True, help='是否使用层归一化')
    parser.add_argument('--mask_path', type=str, default='./data/test_mask.h5', help='掩码文件路径')

    # 序列长度参数
    parser.add_argument('--pre_seq_length', type=int, default=10, help='输入序列长度')
    parser.add_argument('--aft_seq_length', type=int, default=10, help='输出序列长度')
    parser.add_argument('--total_length', type=int, default=20, help='总序列长度')
    parser.add_argument('--reverse_scheduled_sampling', type=int, default=0,
                        help='是否使用反向计划采样')

    # 设备参数
    parser.add_argument('--use_gpu', type=bool, default=True, help='使用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU编号')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')

    args = parser.parse_args()

    # 参数验证
    if len(args.num_hidden) != args.num_layers:
        raise ValueError(f"num_hidden长度({len(args.num_hidden)})必须等于num_layers({args.num_layers})")

    # 创建推理器并运行
    inferencer = ConvLSTMInference(args)
    inferencer.run_inference()


if __name__ == '__main__':
    main()