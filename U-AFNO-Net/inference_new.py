import os
import argparse
import h5py
import numpy as np
import torch
from tqdm import tqdm
import json
from datetime import datetime

from model import create_model
from utils import set_seed, print_log


class WeatherInference:
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
        
        # 根据模型类型准备不同的参数
        model_params = self._prepare_model_params()
        
        # 创建模型
        shape_in = tuple(self.args.in_shape)
        self.model = create_model(
            model_name=self.args.model_name,
            shape_in=shape_in,
            **model_params
        ).to(self.device)
        
        # 加载权重
        checkpoint = torch.load(self.args.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        print(f"✓ 模型加载成功: {self.args.model_name} - {self.args.model_path}")

    def _prepare_model_params(self):
        """根据模型类型准备参数 - 参考exp.py的逻辑"""
        args = self.args
        base_params = {}
        
        if 'compact_unet_afno' in args.model_name or 'unet_afno' in args.model_name:
            # SimVP系列模型参数
            base_params.update({
                'hid_S': args.hid_S,
                'N_T': args.N_T,
                'mask_path': args.mask_path,
                'input_steps': args.input_steps,
                'output_steps': args.output_steps,
                'bilinear': args.bilinear,
                'img_size': tuple(args.img_size),
                'patch_size': tuple(args.patch_size),
                'embed_dim': args.embed_dim if args.embed_dim > 0 else None,
                'mlp_ratio': args.mlp_ratio,
                'drop_rate': 0.0,  # 推理时关闭dropout
                'drop_path_rate': 0.0,
                'num_blocks': args.num_blocks,
                'sparsity_threshold': args.sparsity_threshold,
                'hard_thresholding_fraction': args.hard_thresholding_fraction,
            })
            
        elif args.model_name == 'pure_afno':
            # 纯AFNO模型参数 - 只传递AFNO相关的参数
            base_params.update({
                'img_size': tuple(args.img_size),
                'input_steps': args.input_steps,
                'output_steps': args.output_steps,
                'mask_path':args.mask_path,
                'patch_size': tuple(args.patch_size),
                'embed_dim': args.embed_dim if args.embed_dim > 0 else 256,
                'depth': args.N_T,
                'mlp_ratio': args.mlp_ratio,
                'drop_rate': 0.0,
                'drop_path_rate': 0.0,
                'num_blocks': args.num_blocks,
                'sparsity_threshold': args.sparsity_threshold,
                'hard_thresholding_fraction': args.hard_thresholding_fraction,
            })
            
        elif args.model_name == 'pure_unet':
            # 纯UNet模型参数
            base_params.update({
                'hid_S': args.hid_S,
                'input_steps': args.input_steps,
                'output_steps': args.output_steps,
                'mask_path': args.mask_path,
                'bilinear': args.bilinear,
            })
        
        print(f"模型 {args.model_name} 使用参数: {list(base_params.keys())}")
        return base_params

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
        准备初始输入数据：连续4个时间步
        Args:
            start_time: 起始时间索引
        Returns:
            input_tensor: (1, 4, C, H, W) - 4个连续时间步的数据
        """
        # 检查是否有足够的历史数据
        if start_time < (self.args.input_steps - 1) * self.dt:
            raise ValueError(f"起始时间 {start_time} 不足，需要至少 {self.args.input_steps-1} 个历史时间步")
        
        # 获取连续4个时间步的数据: [start_time-3, start_time-2, start_time-1, start_time]
        #time_indices = list(range(start_time - self.args.input_steps + 1, start_time + 1))

        time_indices = []
        for i in range(self.args.input_steps):
            time_idx = start_time - (self.args.input_steps - 1 - i) * self.dt
            time_indices.append(time_idx)
        
        # 提取数据: (C, 4, H, W)
        input_data = self.test_data[:, time_indices, :, :]
        
        # 转换为tensor并调整维度: (C, 4, H, W) -> (4, C, H, W) -> (1, 4, C, H, W)
        input_tensor = torch.from_numpy(input_data).float()  # (C, 4, H, W)
        input_tensor = input_tensor.permute(1, 0, 2, 3)     # (4, C, H, W)
        input_tensor = input_tensor.contiguous()            # 确保内存连续
        input_tensor = input_tensor.unsqueeze(0)            # (1, 4, C, H, W)
        
        return input_tensor.to(self.device)

    def _update_input_with_prediction(self, current_input, prediction):
        """
        用新的预测结果更新输入序列
        Args:
            current_input: (1, 4, C, H, W) - 当前4步输入
            prediction: (1, 1, C, H, W) - 新的预测结果
        Returns:
            new_input: (1, 4, C, H, W) - 更新后的4步输入（去掉最老的，加入新预测）
        """
        # 确保输入tensor内存连续
        current_input = current_input.contiguous()
        prediction = prediction.contiguous()
        
        # 去掉最老的时间步（第0个），保留后3个: (1, 4, C, H, W) -> (1, 3, C, H, W)
        recent_steps = current_input[:, 1:, :, :, :]
        
        # 将新预测加到末尾: (1, 3, C, H, W) + (1, 1, C, H, W) -> (1, 4, C, H, W)
        new_input = torch.cat([recent_steps, prediction], dim=1)
        
        return new_input.contiguous()

    def autoregressive_forecast(self, start_time, forecast_length):
        """自回归预报 - 适配4步输入1步输出"""
        print(f"开始预报 - 起始时间: {start_time}, 预报长度: {forecast_length}")
        print(f"模型配置: 输入{self.args.input_steps}步, 输出{self.args.output_steps}步")
        
        # 检查时间范围
        if start_time < (self.args.input_steps - 1) * self.dt:
            raise ValueError(f"起始时间 {start_time} 不足，需要至少 {self.args.input_steps-1} 个历史时间步")
        
        if start_time + forecast_length > self.n_timesteps:
            available_length = self.n_timesteps - start_time
            print(f"警告: 预报长度超出数据范围，调整为 {available_length}")
            forecast_length = available_length
        
        # 准备初始输入：连续4个时间步
        current_input = self._prepare_initial_input(start_time)  # (1, 4, C, H, W)
        print(f"初始输入形状: {current_input.shape}")
        
        # 存储预报结果
        forecasts = []
        ground_truths = []
        
        with torch.no_grad():
            for step in tqdm(range(forecast_length), desc="预报中"):
                try:
                    # 确保输入tensor内存连续
                    current_input = current_input.contiguous()
                    
                    # 模型预报：输入4步，输出1步
                    prediction = self.model(current_input)  # (1, 1, C, H, W)
                    
                    # 保存预报结果
                    forecast_frame = prediction.squeeze(0).cpu().numpy()  # (1, C, H, W)
                    forecasts.append(forecast_frame)
                    
                    # 获取对应的真实值
                    gt_time = start_time + (step + 1) * self.dt
                    if gt_time < self.n_timesteps:
                        gt_frame = self.test_data[:, gt_time, :, :]  # (C, H, W)
                        gt_frame = np.expand_dims(gt_frame, 0)  # (1, C, H, W)
                        ground_truths.append(gt_frame)
                    
                    # 更新输入序列：滑动窗口方式
                    current_input = self._update_input_with_prediction(current_input, prediction)
                    
                except Exception as e:
                    print(f"\n❌ 预报步骤 {step} 出错:")
                    print(f"   当前输入形状: {current_input.shape}")
                    print(f"   输入是否连续: {current_input.is_contiguous()}")
                    if 'prediction' in locals():
                        print(f"   预测输出形状: {prediction.shape}")
                        print(f"   预测是否连续: {prediction.is_contiguous()}")
                    print(f"   错误信息: {str(e)}")
                    raise e
        
        # 转换为numpy数组
        forecasts = np.concatenate(forecasts, axis=0)  # (forecast_length, C, H, W)
        if ground_truths:
            ground_truths = np.concatenate(ground_truths, axis=0)  # (forecast_length, C, H, W)
        else:
            ground_truths = np.array([])
        
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
            f"forecast_start{start_time:04d}_length{forecast_length:03d}.h5"
        )
        
        print(f"保存结果: {output_file}")
        
        with h5py.File(output_file, 'w') as f:
            # 保存预报数据 (forecast_length, C, H, W) -> (C, forecast_length, H, W)
            forecast_data = forecasts.transpose(1, 0, 2, 3)
            f.create_dataset('forecast', data=forecast_data, compression='gzip')
            
            # 保存真实值
            if len(ground_truths) > 0:
                gt_data = ground_truths.transpose(1, 0, 2, 3)
                f.create_dataset('ground_truth', data=gt_data, compression='gzip')
            
            # 保存元数据
            f.attrs['start_time'] = start_time
            f.attrs['forecast_length'] = forecast_length
            f.attrs['model_name'] = self.args.model_name
            f.attrs['input_steps'] = self.args.input_steps
            f.attrs['output_steps'] = self.args.output_steps
            f.attrs['timestamp'] = datetime.now().isoformat()
            f.attrs['mse'] = metrics.get('mse', 0.0)
            f.attrs['mae'] = metrics.get('mae', 0.0)
        
        print(f"✓ 结果已保存")

    def run_inference(self):
        """执行推理"""
        print("=" * 60)
        print("开始天气预报推理")
        print(f"模型: {self.args.model_name}")
        print(f"输入配置: {self.args.input_steps}步输入 -> {self.args.output_steps}步输出")
        print(f"测试数据: {self.args.test_data_path}")
        print("=" * 60)
        
        all_metrics = []
        
        for start_time in self.args.start_times:
            # 检查起始时间是否有足够的历史数据
            if start_time < self.args.input_steps - 1:
                print(f"⚠️ 跳过起始时间 {start_time}：需要至少 {self.args.input_steps-1} 个历史时间步")
                continue
                
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
                    self.args.forecast_length, metrics
                )
                
            except Exception as e:
                print(f"✗ 处理起始时间 {start_time} 出错: {str(e)}")
                continue
        
        # 保存整体指标
        if all_metrics:
            metrics_file = os.path.join(self.args.output_dir, 'metrics.json')
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
    parser = argparse.ArgumentParser(description='Weather Forecasting Inference')
    
    # 核心参数
    parser.add_argument('--model_name', required=True,
                        choices=['compact_unet_afno', 'unet_afno', 'pure_afno', 'pure_unet'],
                        help='模型名称')
    parser.add_argument('--model_path', required=True, help='训练好的模型路径')
    parser.add_argument('--test_data_path', required=True, help='测试数据路径 (2015.h5)')
    
    # 预报参数
    parser.add_argument('--start_times', type=int, nargs='+', default=[10, 100, 200, 300],
                        help='预报起始时间列表 (注意：需要>=3以确保有足够历史数据)')
    parser.add_argument('--forecast_length', type=int, default=24, help='预报长度')
    parser.add_argument('--output_dir', default='./inference_results/64x64_2steps_CompactUNet_AFNOp1_scs_mask/', help='输出目录')
    
    # 模型配置参数（需要与训练时保持一致）
    parser.add_argument('--in_shape', type=int, nargs=4, default=[4, 4, 256, 256],
                        help='输入形状 [T, C, H, W]')
    parser.add_argument('--input_steps', type=int, default=4, help='模型输入时间步长')
    parser.add_argument('--output_steps', type=int, default=1, help='模型输出时间步长')
    parser.add_argument('--mask_path', type=str, default='./data/256x256_h5new/mask.h5',
                        help='掩码文件路径')
    
    # UNet/AFNO参数（需要与训练时保持一致）
    parser.add_argument('--hid_S', type=int, default=16, help='隐藏通道数')
    parser.add_argument('--N_T', type=int, default=8, help='中间网络层数')
    parser.add_argument('--bilinear', default=False, type=bool, help='UNet是否使用双线性插值')
    parser.add_argument('--img_size', type=int, nargs=2, default=[32, 32], help='AFNO图像尺寸')
    parser.add_argument('--patch_size', type=int, nargs=2, default=[1, 1], help='AFNO patch尺寸')
    parser.add_argument('--embed_dim', type=int, default=768, help='嵌入维度')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP比率')
    parser.add_argument('--num_blocks', type=int, default=8, help='AFNO块数')
    parser.add_argument('--sparsity_threshold', type=float, default=0.01, help='稀疏阈值')
    parser.add_argument('--hard_thresholding_fraction', type=float, default=1.0, help='硬阈值分数')
    
    # 设备参数
    parser.add_argument('--use_gpu', default=True, help='使用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU编号')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    
    args = parser.parse_args()
    
    # 参数验证
    if args.input_steps != args.in_shape[0]:
        print(f"⚠️ 参数不一致：input_steps={args.input_steps}, in_shape[0]={args.in_shape[0]}")
        print(f"自动调整 input_steps 为 {args.in_shape[0]}")
        args.input_steps = args.in_shape[0]
    
    # 检查起始时间的有效性
    valid_start_times = [t for t in args.start_times if t >= args.input_steps - 1]
    if len(valid_start_times) < len(args.start_times):
        invalid_times = [t for t in args.start_times if t < args.input_steps - 1]
        print(f"⚠️ 删除无效的起始时间 {invalid_times}（需要至少 {args.input_steps-1} 个历史时间步）")
        args.start_times = valid_start_times
    
    if not args.start_times:
        raise ValueError(f"❌ 没有有效的起始时间！请确保起始时间 >= {args.input_steps-1}")
    
    print(f"✓ 有效起始时间: {args.start_times}")
    
    # 创建推理器并运行
    inferencer = WeatherInference(args)
    inferencer.run_inference()


if __name__ == '__main__':
    main()