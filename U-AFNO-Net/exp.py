import os
import os.path as osp
import h5py
import json
import torch
import pickle
import logging
import numpy as np
from model import create_model  # 使用新的统一模型工厂函数
from tqdm import tqdm
from API import *
from utils import *


class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()
        self.mask = self._load_mask(self.args.mask_path)

        self._preparation()
        print_log(output_namespace(self.args))

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        
        # prepare data
        self._get_data()
        # build the model
        self._build_model()
        # setup optimizer and criterion
        self._select_optimizer()
        self._select_criterion()

    def _build_model(self):
        args = self.args

        # 验证参数合理性
        self._validate_model_params()

        # 准备模型参数字典
        model_params = self._prepare_model_params()

        # 输入形状
        shape_in = tuple(args.in_shape)

        try:
            # 使用统一的模型工厂函数创建模型
            self.model = create_model(
                model_name=args.model_name,
                shape_in=shape_in,
                **model_params
            ).to(self.device)
            
            print_log(f"✓ 成功创建模型: {args.model_name}")
                
        except Exception as e:
            print_log(f"✗ 模型创建失败: {str(e)}")
            print_log(f"模型名称: {args.model_name}")
            print_log(f"输入形状: {shape_in}")
            raise e

        # 打印模型详细信息
        self._print_model_info()

    def _prepare_model_params(self):
        """准备模型参数字典"""
        args = self.args
        
        # 基础参数（所有模型通用）
        base_params = {}
        
        # 根据模型类型添加特定参数
        if 'compact_unet_afno' in args.model_name:
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
                'drop_rate': args.drop_rate,
                'drop_path_rate': args.drop_path_rate,
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
                'mask_path': args.mask_path,
                'patch_size': tuple(args.patch_size),
                'embed_dim': args.embed_dim if args.embed_dim > 0 else 256,
                'depth': args.N_T,
                'mlp_ratio': args.mlp_ratio,
                'drop_rate': args.drop_rate,
                'drop_path_rate': args.drop_path_rate,
                'num_blocks': args.num_blocks,
                'sparsity_threshold': args.sparsity_threshold,
                'hard_thresholding_fraction': args.hard_thresholding_fraction,
            })
            # 注意：pure_afno不需要hid_S, N_S, bilinear等参数
            
        elif args.model_name == 'pure_unet':
            base_params.update({
                'hid_S': args.hid_S,  # 对应紧凑UNet的hid_S
                'input_steps': args.input_steps,
                'output_steps': args.output_steps,
                'mask_path': args.mask_path,
                'bilinear': args.bilinear,
            })
        
        return base_params

    def _validate_model_params(self):
        """验证模型参数的合理性"""
        args = self.args

        # 验证模型名称
        valid_models = [
            'unet_afno', 'compact_unet_afno',
            'pure_afno', 'pure_unet'
        ]
        
        if args.model_name not in valid_models:
            print_log(f"Warning: 未知的模型名称 {args.model_name}，支持的模型: {valid_models}")

        # 对于包含AFNO的模型，检查img_size和patch_size的兼容性
        if 'afno' in args.model_name:
            if args.img_size[0] % args.patch_size[0] != 0 or args.img_size[1] % args.patch_size[1] != 0:
                print_log(f"Warning: img_size {args.img_size} 不能被 patch_size {args.patch_size} 整除")

            # 检查embed_dim的合理性
            if args.embed_dim > 0 and args.embed_dim < 32:
                print_log(f"Warning: embed_dim {args.embed_dim} 可能过小，建议>=32")

        # 根据模型类型推算合理的参数范围
        T, C, H, W = args.in_shape
        
        # 为不同模型类型提供参数建议
        if args.model_name.startswith('compact_unet_afno'):
            suggested_img_size = [H // 8, W // 8]
            print_log(f"紧凑UNet模型建议img_size: {suggested_img_size}")
        elif args.model_name.startswith('unet_afno'):
            suggested_img_size = [H // 16, W // 16]
            print_log(f"标准UNet模型建议img_size: {suggested_img_size}")
        elif args.model_name == 'pure_afno':
            suggested_img_size = [H, W]  # 纯AFNO直接处理原始尺寸
            print_log(f"纯AFNO模型建议img_size: {suggested_img_size}")
        elif args.model_name == 'pure_unet':
            print_log(f"纯UNet模型配置: hid_S={args.hid_S},(与紧凑UNet一致)")

        # 打印当前配置
        print_log(f"当前模型: {args.model_name}")
        print_log(f"输入形状: {args.in_shape}")
        if hasattr(args, 'img_size') and 'afno' in args.model_name:
            print_log(f"设置的img_size: {list(args.img_size)}")
        if args.model_name == 'pure_unet':
            print_log(f"PureUNet参数: hid_channels={args.hid_S}, depth=3")

    def _print_model_info(self):
        """打印模型详细信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print_log("=" * 50)
        print_log("模型信息汇总:")
        print_log(f"模型类型: {self.args.model_name}")
        print_log(f"总参数量: {total_params:,}")
        print_log(f"可训练参数量: {trainable_params:,}")
        print_log(f"模型大小: {total_params * 4 / 1024**2:.2f} MB")

        # 计算FLOPs
        try:
          from thop import profile
          from thop import clever_format
        
          # 创建一个dummy输入
          dummy_input = torch.randn(1, *self.args.in_shape).to(self.device)
        
          # 计算FLOPs和参数量
          flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)
        
          # 格式化输出
          flops, params = clever_format([flops, params], "%.3f")
          print_log(f"FLOPs: {flops}")
          print_log(f"参数量(thop): {params}")
        
        except ImportError:
          print_log("提示: 安装 thop 库以计算FLOPs (pip install thop)")
        except Exception as e:
          print_log(f"计算FLOPs时出错: {str(e)}")
        
        
        
        
        # 根据模型类型打印详细的模块参数统计
        self._print_module_params()
        
        print_log("=" * 50)

    def _print_module_params(self):
        """打印各模块的参数量"""
        args = self.args
        
        if args.model_name.startswith('simvp'):
            # SimVP系列模型的模块统计
            if hasattr(self.model, 'enc'):
                enc_params = sum(p.numel() for p in self.model.enc.parameters())
                print_log(f"编码器参数量: {enc_params:,}")
            
            if hasattr(self.model, 'hid'):
                hid_params = sum(p.numel() for p in self.model.hid.parameters())
                print_log(f"中间网络参数量: {hid_params:,}")
            
            if hasattr(self.model, 'dec'):
                dec_params = sum(p.numel() for p in self.model.dec.parameters())
                print_log(f"解码器参数量: {dec_params:,}")
        
        elif args.model_name == 'pure_afno':
            # 纯AFNO模型的模块统计
            if hasattr(self.model, 'input_proj'):
                input_params = sum(p.numel() for p in self.model.input_proj.parameters())
                print_log(f"输入投影参数量: {input_params:,}")
            
            if hasattr(self.model, 'afno_net'):
                afno_params = sum(p.numel() for p in self.model.afno_net.parameters())
                print_log(f"AFNO网络参数量: {afno_params:,}")
            
            if hasattr(self.model, 'output_proj'):
                output_params = sum(p.numel() for p in self.model.output_proj.parameters())
                print_log(f"输出投影参数量: {output_params:,}")
        
        elif args.model_name == 'pure_unet':
            # 纯UNet模型的模块统计
            if hasattr(self.model, 'input_conv'):
                input_params = sum(p.numel() for p in self.model.input_conv.parameters())
                print_log(f"输入层参数量: {input_params:,}")
            
            if hasattr(self.model, 'downs'):
                down_params = sum(p.numel() for module in self.model.downs for p in module.parameters())
                print_log(f"下采样层参数量: {down_params:,}")
            
            if hasattr(self.model, 'ups'):
                up_params = sum(p.numel() for module in self.model.ups for p in module.parameters())
                print_log(f"上采样层参数量: {up_params:,}")

    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader = load_data(**config)
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer

    def _load_mask(self, mask_path):
        try:
          with h5py.File(mask_path, 'r') as f:
            mask = f['fields'][:]
            mask_tensor = torch.from_numpy(mask.astype(np.float32))
          if mask_tensor.shape[0] == 1:
            return mask_tensor.to(self.device)
          else:
            return mask_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"掩码文件 {mask_path} 不存在")
        except KeyError:
            available_keys = list(f.keys())
            raise KeyError(f"数据集中未找到'fields，可用键：{available_keys}")
    '''
    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()
    
    

    def _select_criterion(self):
      def masked_relative_loss(pred, target):
        # 平方相对误差
        relative_error = ((pred - target) / (torch.abs(target) + 1e-8)) ** 2
        return torch.mean(relative_error * self.mask)
      self.criterion = masked_relative_loss
    '''
    
    

    def _select_criterion(self):
      def masked_mse_loss(pred, target):
        return torch.mean((pred - target) ** 2 * self.mask)
      self.criterion = masked_mse_loss
    

    def _save(self, name=''):
        """保存完整模型参数"""
        model_path = os.path.join(self.checkpoints_path, name + '.pth')
        torch.save(self.model.state_dict(), model_path)
        
        # 保存scheduler状态
        scheduler_path = os.path.join(self.checkpoints_path, name + '.pkl')
        with open(scheduler_path, 'wb') as fw:
            pickle.dump(self.scheduler.state_dict(), fw)

    def load_model(self, model_path):
        """加载预训练模型参数"""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print_log(f'✓ Successfully loaded model from {model_path}')
        except Exception as e:
            print_log(f'✗ Failed to load model from {model_path}: {str(e)}')
            raise e
    

    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)

        for epoch in range(config['epochs']):
            train_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_loader)

            for batch_x, batch_y in train_pbar:
                self.optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                try:
                    pred_y = self.model(batch_x)
                except Exception as e:
                    print_log(f"✗ 前向传播错误: {str(e)}")
                    print_log(f"输入形状: {batch_x.shape}")
                    print_log(f"期望输出形状: {batch_y.shape}")
                    print_log(f"模型类型: {args.model_name}")
                    raise e

                loss = self.criterion(pred_y, batch_y)
                train_loss.append(loss.item())
                train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            train_loss = np.average(train_loss)

            if epoch % args.log_step == 0:
                with torch.no_grad():
                    vali_loss = self.vali(self.vali_loader)
                    if epoch % (args.log_step * 100) == 0:
                        self._save(name=str(epoch + 1))
                print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f}\n".format(
                    epoch + 1, train_loss, vali_loss))
                recorder(vali_loss, self.model, self.path)  # recorder保存vali最佳模型参数

        # 加载训练过程中的最佳模型
        best_model_path = osp.join(self.path, 'checkpoint.pth')
        if osp.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            print_log('✓ Loaded best model from training')
        
        return self.model

    def vali(self, vali_loader):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader, desc='Validation')
        
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            if i * batch_x.shape[0] > 1000:
                break

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            try:
                pred_y = self.model(batch_x)
            except Exception as e:
                print_log(f"✗ 验证阶段前向传播错误: {str(e)}")
                print_log(f"输入形状: {batch_x.shape}")
                print_log(f"模型类型: {self.args.model_name}")
                raise e
                
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                pred_y, batch_y], [preds_lst, trues_lst]))

            loss = self.criterion(pred_y, batch_y)
            vali_pbar.set_description('vali loss: {:.4f}'.format(loss.item()))
            total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        
        mse, mae = metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std, False)
        print_log('vali mse:{:.4f}, mae:{:.4f}'.format(mse, mae))
        
        self.model.train()
        return total_loss

    # 只需要修改 Exp 类中的 train 方法：

    '''

    def train(self, args):
      config = args.__dict__
      recorder = Recorder(verbose=True)

      for epoch in range(config['epochs']):
          train_loss = []
          self.model.train()
          train_pbar = tqdm(self.train_loader)

          for batch_x, batch_y in train_pbar:
              self.optimizer.zero_grad()
              batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
              # 判断是否是多步目标
              if len(batch_y.shape) == 5 and batch_y.shape[1] > 1:
                  # 多步循环预测
                  n_steps = batch_y.shape[1]
                  total_loss = 0
                  current_input = batch_x
                
                  for step in range(n_steps):
                      # 预测一步
                      pred_y = self.model(current_input)  # (B, 1, C, H, W)
                    
                      # 获取当前步的目标
                      if len(pred_y.shape) == 4:  # 如果模型输出是 (B, C, H, W)
                          pred_y = pred_y.unsqueeze(1)  # 变成 (B, 1, C, H, W)
                    
                      target_step = batch_y[:, step:step+1, :, :, :]
                    
                      # 计算损失
                      step_loss = self.criterion(pred_y, target_step)
                      total_loss += step_loss
                    
                      # 准备下一步输入：移除最早的帧，添加预测帧
                      if step < n_steps - 1:
                          current_input = torch.cat([
                              current_input[:, 1:, :, :, :],  # 去掉第一帧
                              pred_y.detach()                  # 添加预测帧
                          ], dim=1)
                
                  loss = total_loss / n_steps  # 平均损失
                
              else:
                  # 单步预测（保持原来的逻辑）
                  pred_y = self.model(batch_x)
                  loss = self.criterion(pred_y, batch_y)
            
              train_loss.append(loss.item())
              train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

              loss.backward()
              self.optimizer.step()
              self.scheduler.step()

          train_loss = np.average(train_loss)

          if epoch % args.log_step == 0:
              with torch.no_grad():
                  vali_loss = self.vali(self.vali_loader)
                  if epoch % (args.log_step * 100) == 0:
                      self._save(name=str(epoch + 1))
              print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f}\n".format(
                  epoch + 1, train_loss, vali_loss))
              recorder(vali_loss, self.model, self.path)

      best_model_path = osp.join(self.path, 'checkpoint.pth')
      if osp.exists(best_model_path):
          self.model.load_state_dict(torch.load(best_model_path))
          print_log('✓ Loaded best model from training')
    
      return self.model


# 同样修改 vali 方法：
    def vali(self, vali_loader):
      self.model.eval()
      preds_lst, trues_lst, total_loss = [], [], []
      vali_pbar = tqdm(vali_loader, desc='Validation')
    
      for i, (batch_x, batch_y) in enumerate(vali_pbar):
          if i * batch_x.shape[0] > 1000:
              break

          batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        
          # 判断是否是多步目标
          if len(batch_y.shape) == 5 and batch_y.shape[1] > 1:
              # 多步预测
              n_steps = batch_y.shape[1]
              current_input = batch_x
              predictions = []
            
              for step in range(n_steps):
                  pred_y = self.model(current_input)
                  if len(pred_y.shape) == 4:
                      pred_y = pred_y.unsqueeze(1)
                
                  predictions.append(pred_y)
                
                  if step < n_steps - 1:
                      current_input = torch.cat([
                          current_input[:, 1:, :, :, :],
                          pred_y.detach()
                      ], dim=1)
            
              pred_y = torch.cat(predictions, dim=1)  # (B, N, C, H, W)
            
          else:
              # 单步预测
              pred_y = self.model(batch_x)
            
          list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
              pred_y, batch_y], [preds_lst, trues_lst]))

          loss = self.criterion(pred_y, batch_y)
          vali_pbar.set_description('vali loss: {:.4f}'.format(loss.item()))
          total_loss.append(loss.item())

      total_loss = np.average(total_loss)
      preds = np.concatenate(preds_lst, axis=0)
      trues = np.concatenate(trues_lst, axis=0)
    
      mse, mae = metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std, False)
      print_log('vali mse:{:.4f}, mae:{:.4f}'.format(mse, mae))
    
      self.model.train()
      return total_loss

    '''

    def test(self, args):
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        
        test_pbar = tqdm(self.test_loader, desc='Testing')
        for batch_x, batch_y in test_pbar:
            batch_x = batch_x.to(self.device)
            
            try:
                pred_y = self.model(batch_x)
            except Exception as e:
                print_log(f"✗ 测试阶段前向传播错误: {str(e)}")
                print_log(f"输入形状: {batch_x.shape}")
                print_log(f"模型类型: {args.model_name}")
                raise e
                
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))

        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])

        folder_path = osp.join(self.path, 'results', args.ex_name, 'sv')
        os.makedirs(folder_path, exist_ok=True)

        mse, mae = metric(preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std, False)
        print_log('Test Results - MSE:{:.4f}, MAE:{:.4f}'.format(mse, mae))

        # 保存测试结果
        for np_data in ['inputs', 'trues', 'preds']:
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        
        return mse