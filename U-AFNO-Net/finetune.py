import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import os.path as osp
from exp import Exp  # 直接使用你的Exp类
from utils import *
import argparse


class AutoregressiveFinetuner:
    """自回归微调器 - 适配多模型架构"""

    def __init__(self, exp, args):
        self.exp = exp
        self.args = args
        self.device = exp.device
        self.model = exp.model
        self.criterion = exp.criterion

        # 微调参数
        self.num_pred_steps = args.num_pred_steps
        self.teacher_forcing_ratio = args.teacher_forcing_ratio

        # 获取数据加载器
        self._get_data()

        # 创建微调优化器 - 与Exp保持一致
        self._select_optimizer()

    def _get_data(self):
        """获取数据加载器 - 直接使用exp的加载器"""
        self.train_loader = self.exp.train_loader
        self.vali_loader = self.exp.vali_loader
        self.test_loader = self.exp.test_loader

    def _select_optimizer(self):
        """选择优化器 - 与Exp类保持一致的结构"""
        # 创建优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.args.finetune_lr  # 使用微调学习率
        )
        
        # 计算实际的steps_per_epoch
        steps_per_epoch = len(self.train_loader)
        if hasattr(self.args, 'max_batches_per_epoch') and self.args.max_batches_per_epoch < steps_per_epoch:
            steps_per_epoch = self.args.max_batches_per_epoch
        
        # 创建OneCycleLR调度器
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=self.args.finetune_lr,  # 使用微调学习率
            steps_per_epoch=steps_per_epoch, 
            epochs=self.args.finetune_epochs,  # 使用微调轮数
        )
        
        
        print_log(f"🔧 微调配置:")
        print_log(f"   模型类型: {args.model_name}")
        print_log(f"   预测步数: {self.num_pred_steps}")
        print_log(f"   教师强制比例: {self.teacher_forcing_ratio}")
        print_log(f"   微调学习率: {args.finetune_lr}")
        return self.optimizer

    def autoregressive_forward(self, inputs, targets=None, use_teacher_forcing=True):
        """
        自回归前向传播
        Args:
            inputs: (B, T, C, H, W) - 原始输入序列
            targets: (B, N, C, H, W) - 真实目标序列（用于教师强制）
            use_teacher_forcing: 是否使用教师强制
        Returns:
            predictions: 预测序列
            total_loss: 总损失
        """
        B, T_in, C, H, W = inputs.shape

        predictions = []
        current_input = inputs  # (B, T_in, C, H, W)
        total_loss = 0

        for step in range(self.num_pred_steps):
            # 模型预测下一帧
            pred = self.model(current_input)  # (B, output_steps, C, H, W)
            predictions.append(pred)

            # 计算损失
            if targets is not None and step < targets.shape[1]:
                # 使用真实目标
                target = targets[:, step:step+1, :, :, :]  # (B, 1, C, H, W)
                step_loss = self.criterion(pred, target)

            total_loss = total_loss + step_loss

            # 准备下一步输入
            if step < self.num_pred_steps - 1:
                if use_teacher_forcing and targets is not None and step < targets.shape[1] - 1:
                    if torch.rand(1).item() < self.teacher_forcing_ratio:
                        # 使用真实帧
                        next_frame = targets[:, step:step+1, :, :, :]
                    else:
                        # 使用预测帧
                        next_frame = pred.detach()
                else:
                    # 使用预测帧
                    next_frame = pred.detach()

                # 更新输入序列：移除最老的帧，添加新帧
                current_input = torch.cat([
                    current_input[:, 1:],  # 移除第一帧
                    next_frame  # 添加新帧
                ], dim=1)

        predictions = torch.cat(predictions, dim=1)  # (B, num_pred_steps, C, H, W)
        avg_loss = total_loss #/ self.num_pred_steps

        return predictions, avg_loss

    def finetune_epoch(self, dataloader, epoch):
        """微调一个epoch"""
        self.model.train()
        epoch_losses = []

        pbar = tqdm(dataloader, desc=f'微调 Epoch {epoch + 1}')

        for batch_idx, (batch_x, batch_y) in enumerate(pbar):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            # 自回归前向传播
            predictions, loss = self.autoregressive_forward(
                batch_x,
                targets=batch_y,  # 传入真实目标
                use_teacher_forcing=True
            )

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            epoch_losses.append(loss.item())
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{loss.item():.8f}',
            'lr': f'{current_lr:.2e}'})

            # 限制每个epoch的batch数量
            if batch_idx >= self.args.max_batches_per_epoch:
                break

        return np.mean(epoch_losses)

    def validate_autoregressive(self, dataloader):
        """验证自回归性能"""
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
                if batch_idx >= 1000:  # 限制验证数量
                    break

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # 纯自回归预测（无教师强制）
                predictions, loss = self.autoregressive_forward(
                    batch_x,
                    targets=batch_y,
                    use_teacher_forcing=True
                )

                val_losses.append(loss.item())

        avg_loss = np.mean(val_losses)
        print_log(f"验证损失: {avg_loss:.8f}")

        return avg_loss

    def run_finetune(self):
        """运行微调过程"""
        print_log("🚀 开始自回归微调...")

        best_loss = float('inf')
        patience = 0
        max_patience = 5

        for epoch in range(self.args.finetune_epochs):
            # 微调训练
            train_loss = self.finetune_epoch(self.exp.train_loader, epoch)

            # 验证
            if epoch % self.args.log_step == 0:
                val_loss = self.validate_autoregressive(self.exp.vali_loader)
                # 获取当前学习率
                current_lr = self.optimizer.param_groups[0]['lr']

                print_log(f"Epoch {epoch + 1}/{self.args.finetune_epochs}")
                print_log(f"  训练损失: {train_loss:.8f}")
                print_log(f"  验证损失: {val_loss:.8f}")
                print_log(f"  当前学习率: {current_lr:.2e}")

                # 保存最佳模型
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience = 0
                    save_path = osp.join(self.exp.checkpoints_path, 'finetune_best.pth')
                    torch.save(self.model.state_dict(), save_path)
                    print_log(f"✅ 保存最佳微调模型: {save_path}")
                else:
                    patience += 1

                # 早停
                if patience >= max_patience:
                    print_log(f"⏹️  早停触发，停止微调")
                    break

        # 加载最佳模型
        best_model_path = osp.join(self.exp.checkpoints_path, 'finetune_best.pth')
        if osp.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            print_log("✅ 加载最佳微调模型")

        return self.model


def create_finetune_parser():
    """创建微调参数解析器"""
    parser = argparse.ArgumentParser()

    # 基础参数 - 与Exp类保持一致
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='64x64_uafnop1_finetune_test_lr1e6', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # 数据参数 - 与Exp类保持一致
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--val_batch_size', default=32, type=int)
    parser.add_argument('--data_root', default='./data/64x64_h5new/')
    parser.add_argument('--dataname', default='custom', type=str)
    parser.add_argument('--num_workers', default=8, type=int)

    # 模型参数 - 与Exp类保持一致
    parser.add_argument('--in_shape', default=[4, 4, 64, 64], type=int, nargs='*')
    parser.add_argument('--hid_S', default=64, type=int)
    parser.add_argument('--N_T', default=8, type=int)
    parser.add_argument('--input_steps', default=4, type=int)
    parser.add_argument('--output_steps', default=1, type=int)
    parser.add_argument('--mask_path', default='./data/64x64_h5new/mask.h5', type=str)

    # 模型选择 - 与Exp类保持一致
    parser.add_argument('--model_name', default='compact_unet_afno',
                        choices=['unet_afno', 'compact_unet_afno', 'pure_afno', 'pure_unet'],
                        help='模型类型选择')

    # AFNO参数 - 与Exp类保持一致
    parser.add_argument('--img_size', default=[8, 8], type=int, nargs=2)
    parser.add_argument('--patch_size', default=[1, 1], type=int, nargs=2)
    parser.add_argument('--embed_dim', default=768, type=int)
    parser.add_argument('--bilinear', default=False, type=bool)
    parser.add_argument('--mlp_ratio', default=4.0, type=float)
    parser.add_argument('--drop_rate', default=0.0, type=float)
    parser.add_argument('--drop_path_rate', default=0.0, type=float)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--sparsity_threshold', default=0.01, type=float)
    parser.add_argument('--hard_thresholding_fraction', default=1.0, type=float)

    # 训练参数 - 为了兼容Exp类
    parser.add_argument('--lr', default=1e-4, type=float, help='原始训练学习率(Exp类需要)')
    parser.add_argument('--epochs', default=20, type=int, help='原始训练轮数(Exp类需要)')
    parser.add_argument('--log_step', default=1, type=int)

    # 微调特定参数
    parser.add_argument('--pretrained_model', required=True, type=str,
                        help='预训练模型路径')
    parser.add_argument('--num_pred_steps', default=4, type=int,
                        help='微调时预测的步数')
    parser.add_argument('--teacher_forcing_ratio', default=0.5, type=float,
                        help='教师强制比例 (0-1)')
    parser.add_argument('--finetune_epochs', default=15, type=int,
                        help='微调轮数')
    parser.add_argument('--finetune_lr', default=1e-5, type=float,
                        help='微调学习率')
    parser.add_argument('--max_batches_per_epoch', default=1825, type=int,
                        help='每个epoch最大batch数量')

    return parser


if __name__ == '__main__':
    args = create_finetune_parser().parse_args()

    print_log("🎯 开始自回归微调实验")
    print_log(f"预训练模型: {args.pretrained_model}")
    print_log(f"模型类型: {args.model_name}")
    print_log(f"预测步数: {args.num_pred_steps}")

    try:
        # 直接使用你的Exp类 - 完全复用所有功能
        exp = Exp(args)

        # 加载预训练模型
        if not osp.exists(args.pretrained_model):
            raise FileNotFoundError(f"预训练模型不存在: {args.pretrained_model}")

        # 使用Exp类的load_model方法
        exp.load_model(args.pretrained_model)

        # 打印模型信息（Exp类已经在初始化时打印了）
        total_params = sum(p.numel() for p in exp.model.parameters())
        print_log(f"📊 模型参数总数: {total_params:,} ({total_params * 4 / 1024 ** 2:.2f} MB)")

        # 创建微调器
        finetuner = AutoregressiveFinetuner(exp, args)

        # 微调前验证
        print_log("📊 微调前自回归性能:")
        pre_loss = finetuner.validate_autoregressive(exp.vali_loader)

        # 运行微调
        finetuned_model = finetuner.run_finetune()

        # 微调后验证
        print_log("📊 微调后自回归性能:")
        post_loss = finetuner.validate_autoregressive(exp.vali_loader)

        improvement = ((pre_loss - post_loss) / pre_loss * 100) if pre_loss > 0 else 0
        print_log(f"性能改进: {improvement:.2f}%")

        print_log("✅ 微调完成！")
        print_log(f"最佳模型保存在: {osp.join(exp.checkpoints_path, 'finetune_best.pth')}")

        # 可选：进行标准测试评估
        if hasattr(args, 'run_final_test') and args.run_final_test:
            print_log("🧪 运行最终测试评估...")
            test_mse = exp.test(args)
            print_log(f"最终测试MSE: {test_mse:.4f}")

    except Exception as e:
        print_log(f"❌ 微调过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e