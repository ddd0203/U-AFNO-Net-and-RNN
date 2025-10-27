import argparse
from exp import Exp
import os.path as osp
import warnings

warnings.filterwarnings('ignore')


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--dataname', default='custom', type=str, help='Dataset name')
    parser.add_argument('--num_workers', default=8, type=int)

    # model parameters - 基础参数
    parser.add_argument('--in_shape', default=[10, 1, 64, 64], type=int, nargs='*',
                        help='[10, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj')
    parser.add_argument('--hid_S', default=32, type=int, help='Hidden channels for spatial processing')
    parser.add_argument('--N_T', default=8, type=int, help='Number of temporal/AFNO layers')
    parser.add_argument('--input_steps', default=4, type=int, help='模型输入时间步长')
    parser.add_argument('--output_steps', default=1, type=int, help='模型输出时间步长')
    parser.add_argument('--mask_path', type=str, default='./data/mask.h5',
                        help='Path to HDF5 mask file')

    # ======================= 新增：模型选择参数 =======================
    parser.add_argument('--model_name', default='compact_unet_afno', 
                        choices=[
                            'unet_afno',           # 标准UNet + AFNO  
                            'compact_unet_afno',   # 紧凑UNet + AFNO
                            'pure_afno',                 # 纯AFNO模型
                            'pure_unet'                  # 纯UNet模型
                        ],
                        help='模型类型选择，用于消融实验')
    
    parser.add_argument('--bilinear', default=False, type=bool, help='UNet是否使用双线性插值')

    # 图像尺寸参数 - 根据数据集特征图大小设置
    parser.add_argument('--img_size', default=[8, 8], type=int, nargs=2,
                        help='AFNO处理的特征图尺寸 [H, W]，应为编码器输出尺寸')

    # AFNO参数
    parser.add_argument('--patch_size', default=[1, 1], type=int, nargs=2, help='AFNO patch size [H, W]')
    parser.add_argument('--embed_dim', default=768, type=int,
                        help='AFNO embedding dimension, 0表示自动设置为channel_in')
    parser.add_argument('--mlp_ratio', default=4.0, type=float, help='MLP ratio in AFNO blocks')
    parser.add_argument('--drop_rate', default=0.0, type=float, help='Dropout rate')
    parser.add_argument('--drop_path_rate', default=0.0, type=float, help='Drop path rate')
    parser.add_argument('--num_blocks', default=8, type=int, help='Number of AFNO frequency blocks')
    parser.add_argument('--sparsity_threshold', default=0.01, type=float, help='Sparsity threshold for AFNO')
    parser.add_argument('--hard_thresholding_fraction', default=1.0, type=float,
                        help='Hard thresholding fraction for AFNO')

    # Training parameters
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')

    # Model mode (train or test)
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                        help='运行模式: train=训练, test=测试')
    parser.add_argument('--model_path', default='',
                        help='测试模式时需指定模型路径，为空则使用最佳checkpoint')

    return parser


def auto_calculate_img_size(in_shape, model_name):
    """根据输入形状和模型名称自动计算img_size"""
    T, C, H, W = in_shape
    
    if model_name == 'compact_unet_afno':
        # 紧凑UNet: 3次下采样 (2^3 = 8)
        img_h, img_w = H // 8, W // 8
        description = f'紧凑UNet+AFNO: {H}x{W} -> {img_h}x{img_w} (下采样8倍)'
    elif model_name == 'unet_afno':
        # 标准UNet: 4次下采样 (2^4 = 16)
        img_h, img_w = H // 16, W // 16
        description = f'标准UNet+AFNO: {H}x{W} -> {img_h}x{img_w} (下采样16倍)'
    elif model_name == 'pure_afno':
        # 纯AFNO: 直接处理原始尺寸
        img_h, img_w = H, W
        description = f'纯AFNO: {H}x{W} -> {img_h}x{img_w} (无下采样)'
    elif model_name == 'pure_unet':
        # 纯UNet: 不需要img_size
        img_h, img_w = H, W
        description = f'纯UNet: {H}x{W} (不使用img_size)'
    else:
        # 默认值
        img_h, img_w = H // 8, W // 8
        description = f'默认设置: {H}x{W} -> {img_h}x{img_w} (下采样8倍)'
    
    return [img_h, img_w], description


def validate_and_auto_config(args):
    """验证参数并自动配置"""
    
    # 自动计算img_size（仅对需要的模型）
    if args.img_size is None or (args.img_size == [8, 8] and args.model_name in ['pure_afno', 'compact_unet_afno', 'unet_afno']):
        args.img_size, size_description = auto_calculate_img_size(args.in_shape, args.model_name)
        print(f"🔧 自动计算img_size: {size_description}")
    else:
        print(f"📐 使用指定的img_size: {args.img_size}")
    
    # 验证img_size的合理性（对于需要img_size的模型）
    if 'afno' in args.model_name:
        if args.img_size[0] <= 0 or args.img_size[1] <= 0:
            raise ValueError(f"❌ 计算得到的img_size {args.img_size} 无效，请检查输入尺寸和模型类型")
        
        # 自动调整patch_size以确保兼容性
        original_patch_size = args.patch_size[:]
        
        # 确保patch_size能整除img_size
        for i in range(2):
            while args.img_size[i] % args.patch_size[i] != 0:
                args.patch_size[i] = max(1, args.patch_size[i] - 1)
        
        if args.patch_size != original_patch_size:
            print(f"🔧 自动调整patch_size: {original_patch_size} -> {args.patch_size}")
    
    
    return args




if __name__ == '__main__':
    args = create_parser().parse_args()
    
    # 显示实验指南
    if args.ex_name == 'Debug':
        print_experiment_guide()
    
    args = validate_and_auto_config(args)
    #print_config_summary(args)
    
    try:
        # 创建实验对象
        exp = Exp(args)

        if args.mode == 'train':
            print('\n' + '>' * 35 + ' 开始训练 ' + '<' * 35)
            
            # 添加训练前的最终检查
            print("🔍 训练前检查:")
            print(f"   ✓ 数据路径: {args.data_root}")
            print(f"   ✓ 模型类型: {args.model_name}")
            print(f"   ✓ 输出路径: {exp.path}")
            print(f"   ✓ 实验名称: {args.ex_name}")
            
            # 显示模型对比信息
            total_params = sum(p.numel() for p in exp.model.parameters())
            print(f"   ✓ 模型参数: {total_params:,}")
            print(f"   ✓ 模型大小: {total_params * 4 / 1024**2:.2f} MB")
            
            exp.train(args)
            print('✅ 训练完成！')
            
            # 自动进行测试
            print('\n' + '>' * 35 + ' 开始测试 ' + '<' * 35)
            mse = exp.test(args)
            print(f'✅ 测试完成！最终MSE: {mse:.4f}')

        elif args.mode == 'test':
            print('\n' + '>' * 35 + ' 开始测试 ' + '<' * 35)
            
            # 处理模型加载
            if args.model_path:
                # 用户指定了模型路径
                if not osp.exists(args.model_path):
                    raise FileNotFoundError(f"❌ 模型文件 {args.model_path} 不存在")
                exp.load_model(args.model_path)
                print(f"📁 使用指定模型: {args.model_path}")
            else:
                # 自动寻找最佳模型
                best_model_path = osp.join(exp.path, 'checkpoint.pth')
                if osp.exists(best_model_path):
                    exp.load_model(best_model_path)
                    print(f"📁 自动加载最佳模型: {best_model_path}")
                else:
                    raise FileNotFoundError(f"❌ 未找到模型文件。请指定--model_path或先完成训练")
            
            # 执行测试
            mse = exp.test(args)
            print(f'✅ 测试完成！MSE: {mse:.4f}')

    except Exception as e:
        print(f"❌ 程序执行出错: {str(e)}")
        print(f"🔧 调试信息:")
        print(f"   - 模型名称: {args.model_name}")
        print(f"   - 输入形状: {args.in_shape}")
        if 'afno' in args.model_name:
            print(f"   - 图像尺寸: {args.img_size}")
            print(f"   - Patch尺寸: {args.patch_size}")
        print(f"\n💡 常见解决方案:")
        print(f"   1. 检查数据路径是否正确")
        print(f"   2. 检查输入尺寸设置是否合理")
        print(f"   3. 尝试降低batch_size")
        print(f"   4. 检查GPU内存是否足够")
        raise e
    
    print(f"\n🎉 实验 '{args.ex_name}' 执行完成！")
    print(f"📊 结果保存在: {exp.path}")


# =============== 批量实验脚本样例 ===============

def run_ablation_experiments():
    """运行完整的消融实验"""
    import subprocess
    import time
    
    # 消融实验配置
    experiments = [
        {
            'name': 'UNet_AFNO_Full',
            'model': 'simvp_unet_afno',
            'description': '完整UNet+AFNO模型'
        },
        {
            'name': 'UNet_AFNO_Compact', 
            'model': 'simvp_compact_unet_afno',
            'description': '紧凑UNet+AFNO模型'
        },
        {
            'name': 'UNet_Only',
            'model': 'simvp_unet_only', 
            'description': 'UNet编码器+UNet中间网络'
        },
        {
            'name': 'CNN_Baseline',
            'model': 'simvp_cnn_baseline',
            'description': 'UNet编码器+CNN中间网络'
        },
        {
            'name': 'Pure_AFNO',
            'model': 'pure_afno',
            'description': '纯AFNO模型'
        },
        {
            'name': 'Pure_UNet',
            'model': 'pure_unet',
            'description': '纯UNet模型'
        }
    ]
    
    print("🧪 开始批量消融实验...")
    results = {}
    
    for exp in experiments:
        print(f"\n{'='*50}")
        print(f"🚀 运行实验: {exp['name']}")
        print(f"📝 描述: {exp['description']}")
        print(f"🏗️  模型: {exp['model']}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # 构建命令
        cmd = [
            'python', 'main.py',
            '--model_name', exp['model'],
            '--ex_name', exp['name'],
            '--epochs', '50',  # 快速实验用较少轮数
            '--mode', 'train'
        ]
        
        try:
            # 运行实验
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ 实验 {exp['name']} 成功完成")
                print(f"⏱️  耗时: {duration/60:.1f} 分钟")
                results[exp['name']] = {'status': 'success', 'duration': duration}
            else:
                print(f"❌ 实验 {exp['name']} 失败")
                print(f"错误信息: {result.stderr}")
                results[exp['name']] = {'status': 'failed', 'error': result.stderr}
                
        except Exception as e:
            print(f"❌ 实验 {exp['name']} 执行异常: {str(e)}")
            results[exp['name']] = {'status': 'error', 'error': str(e)}
    
    # 打印实验汇总
    print(f"\n{'='*60}")
    print("📊 消融实验汇总报告")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    total_count = len(results)
    
    print(f"✅ 成功实验: {success_count}/{total_count}")
    print(f"❌ 失败实验: {total_count - success_count}/{total_count}")
    
    for name, result in results.items():
        status_icon = "✅" if result['status'] == 'success' else "❌"
        duration_info = f"({result['duration']/60:.1f}min)" if result['status'] == 'success' else ""
        print(f"{status_icon} {name} {duration_info}")
    
    print(f"{'='*60}")
    

if __name__ == '__main__' and '--run_ablation' in __import__('sys').argv:
    # 运行批量消融实验
    # 使用方法: python main.py --run_ablation
    run_ablation_experiments()