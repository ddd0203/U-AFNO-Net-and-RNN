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
    parser.add_argument('--ex_name', default='ConvLSTM_Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # Model selection
    parser.add_argument('--model', default='convlstm', type=str, 
                        choices=['convlstm', 'predrnn', 'mim', 'predrnnv2'],
                        help='Model type: convlstm or mim')

    # dataset parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--dataname', default='custom', type=str, help='Dataset name')
    parser.add_argument('--num_workers', default=8, type=int)

    # ConvLSTM model parameters
    parser.add_argument('--in_shape', default=[20, 1, 64, 64], type=int, nargs='*',
                        help='Input shape (T, C, H, W)')
    parser.add_argument('--num_layers', default=4, type=int, help='Number of ConvLSTM layers')
    parser.add_argument('--num_hidden', default=[64, 64, 64, 64], type=int, nargs='*',
                        help='Number of hidden units in each layer')
    parser.add_argument('--filter_size', default=5, type=int, help='Filter size for ConvLSTM')
    parser.add_argument('--stride', default=1, type=int, help='Stride for ConvLSTM')
    parser.add_argument('--patch_size', default=4, type=int, help='Patch size')
    parser.add_argument('--layer_norm', default=True, type=bool, help='Use LayerNorm')

    # Sequence lengths
    parser.add_argument('--pre_seq_length', default=10, type=int, help='Input sequence length')
    parser.add_argument('--aft_seq_length', default=10, type=int, help='Output sequence length')
    parser.add_argument('--total_length', default=20, type=int, help='Total sequence length')

    # Scheduled sampling parameters
    parser.add_argument('--scheduled_sampling', default=True, type=bool)
    parser.add_argument('--reverse_scheduled_sampling', default=0, type=int,
                        help='0: standard sampling, 1: reverse sampling')
    parser.add_argument('--sampling_stop_iter', default=20000, type=int)
    parser.add_argument('--sampling_start_value', default=1.0, type=float)
    parser.add_argument('--sampling_changing_rate', default=0.00005, type=float)

    # Reverse scheduled sampling parameters
    parser.add_argument('--r_sampling_step_1', default=25000, type=int)
    parser.add_argument('--r_sampling_step_2', default=50000, type=int)
    parser.add_argument('--r_exp_alpha', default=5000, type=int)

    # Other parameters
    parser.add_argument('--mask_path', type=str, default='./data/mask.h5',
                        help='Path to HDF5 mask file')

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

'''
def validate_args(args):
    """验证参数配置"""
    # 验证序列长度
    if args.total_length != args.pre_seq_length + args.aft_seq_length:
        print(f"⚠️  调整total_length: {args.total_length} -> {args.pre_seq_length + args.aft_seq_length}")
        args.total_length = args.pre_seq_length + args.aft_seq_length

    # 验证in_shape的时间维度
    if args.in_shape[0] != args.total_length:
        print(f"⚠️  调整in_shape[0]: {args.in_shape[0]} -> {args.total_length}")
        args.in_shape[0] = args.total_length

    # 验证num_hidden长度
    if len(args.num_hidden) != args.num_layers:
        if len(args.num_hidden) == 1:
            # 如果只提供一个值，复制到所有层
            args.num_hidden = args.num_hidden * args.num_layers
            print(f"⚠️  扩展num_hidden到所有层: {args.num_hidden}")
        else:
            # 截断或填充
            if len(args.num_hidden) > args.num_layers:
                args.num_hidden = args.num_hidden[:args.num_layers]
            else:
                last_hidden = args.num_hidden[-1]
                args.num_hidden.extend([last_hidden] * (args.num_layers - len(args.num_hidden)))
            print(f"⚠️  调整num_hidden: {args.num_hidden}")

    # 验证patch_size
    _, _, H, W = args.in_shape
    if H % args.patch_size != 0 or W % args.patch_size != 0:
        print(f"⚠️  警告: 图像尺寸({H}x{W})不能被patch_size({args.patch_size})整除")

    return args
'''


def print_config_summary(args):
    """打印配置摘要"""
    print("\n" + "=" * 60)
    print("ConvLSTM 实验配置")
    print("=" * 60)
    print(f"📊 实验名称: {args.ex_name}")
    print(f"🖥️  设备: {'GPU:' + str(args.gpu) if args.use_gpu else 'CPU'}")
    print(f"📁 数据路径: {args.data_root}")
    print(f"📏 输入形状: {args.in_shape} (T, C, H, W)")
    print(f"🔧 模型配置:")
    print(f"   - 层数: {args.num_layers}")
    print(f"   - 隐藏单元: {args.num_hidden}")
    print(f"   - 滤波器大小: {args.filter_size}")
    print(f"   - Patch大小: {args.patch_size}")
    print(f"   - LayerNorm: {args.layer_norm}")
    print(f"📈 序列长度:")
    print(f"   - 输入序列: {args.pre_seq_length}")
    print(f"   - 输出序列: {args.aft_seq_length}")
    print(f"   - 总长度: {args.total_length}")
    print(f"🎯 训练参数:")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Batch Size: {args.batch_size}")
    print(f"   - Learning Rate: {args.lr}")
    print(f"   - Scheduled Sampling: {'反向' if args.reverse_scheduled_sampling else '标准'}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    args = create_parser().parse_args()
   #args = validate_args(args)

    print_config_summary(args)

    try:
        # 创建实验对象
        exp = Exp(args)

        if args.mode == 'train':
            print('\n' + '>' * 35 + ' 开始训练 ' + '<' * 35)
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
                if not osp.exists(args.model_path):
                    raise FileNotFoundError(f"❌ 模型文件 {args.model_path} 不存在")
                exp.load_model(args.model_path)
                print(f"📁 使用指定模型: {args.model_path}")
            else:
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
        print(f"\n❌ 程序执行出错: {str(e)}")
        print(f"\n💡 调试信息:")
        print(f"   - 输入形状: {args.in_shape}")
        print(f"   - 层数: {args.num_layers}")
        print(f"   - 隐藏单元: {args.num_hidden}")
        print(f"   - Patch大小: {args.patch_size}")
        raise e

    print(f"\n🎉 实验 '{args.ex_name}' 执行完成！")
    print(f"📊 结果保存在: {exp.path}")


# 批量实验示例
def run_convlstm_experiments():
    """运行不同配置的ConvLSTM实验"""
    import subprocess

    experiments = [
        {
            'name': 'ConvLSTM_Base',
            'num_layers': 4,
            'num_hidden': [64, 64, 64, 64],
            'patch_size': 4,
            'description': '基础ConvLSTM配置'
        },
        {
            'name': 'ConvLSTM_Deep',
            'num_layers': 8,
            'num_hidden': [32, 32, 64, 64, 64, 64, 32, 32],
            'patch_size': 4,
            'description': '深层ConvLSTM配置'
        },
        {
            'name': 'ConvLSTM_Wide',
            'num_layers': 4,
            'num_hidden': [128, 128, 128, 128],
            'patch_size': 4,
            'description': '宽ConvLSTM配置'
        },
        {
            'name': 'ConvLSTM_NoPatch',
            'num_layers': 4,
            'num_hidden': [64, 64, 64, 64],
            'patch_size': 1,
            'description': '无Patch的ConvLSTM'
        }
    ]

    print("🧪 开始批量ConvLSTM实验...")

    for exp in experiments:
        print(f"\n{'=' * 50}")
        print(f"🚀 运行实验: {exp['name']}")
        print(f"📝 描述: {exp['description']}")
        print(f"{'=' * 50}")

        # 构建命令
        cmd = [
                  'python', 'main_convlstm.py',
                  '--ex_name', exp['name'],
                  '--num_layers', str(exp['num_layers']),
                  '--num_hidden'] + [str(h) for h in exp['num_hidden']] + [
                  '--patch_size', str(exp['patch_size']),
                  '--epochs', '50',
                  '--mode', 'train'
              ]

        try:
            subprocess.run(cmd, check=True)
            print(f"✅ 实验 {exp['name']} 成功完成")
        except subprocess.CalledProcessError:
            print(f"❌ 实验 {exp['name']} 失败")

    print("\n📊 所有实验完成！")


if __name__ == '__main__' and '--run_experiments' in __import__('sys').argv:
    run_convlstm_experiments()