import os
import os.path as osp
import h5py
import json
import torch
import pickle
import logging
import numpy as np
from convlstm_model import create_model
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

        # 使用模型工厂函数创建模型
        self.model = create_model(
            model_name=args.model,
            num_layers=args.num_layers, 
            num_hidden=args.num_hidden, 
            configs=args
        ).to(self.device)

        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print_log("=" * 50)
        print_log("ConvLSTM模型信息:")
        print_log(f"层数: {args.num_layers}")
        print_log(f"隐藏单元: {args.num_hidden}")
        print_log(f"总参数量: {total_params:,}")
        print_log(f"可训练参数量: {trainable_params:,}")
        print_log(f"模型大小: {total_params * 4 / 1024 ** 2:.2f} MB")
        print_log("=" * 50)
        self._estimate_flops()

        

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
        except:
            return None

    def _select_criterion(self):
        def masked_mse_loss(pred, target):
            if self.mask is not None:
                return torch.mean((pred - target) ** 2 * self.mask)
            else:
                return torch.mean((pred - target) ** 2)

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

        # 用于计算eta的参数
        eta = 1.0  # 初始eta值

        for epoch in range(config['epochs']):
            train_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_loader)

            for itr, (batch_x, batch_y) in enumerate(train_pbar):
                self.optimizer.zero_grad()

                # 合并输入和目标为5帧数据
            # batch_x: [B, 4, C, H, W], batch_y: [B, 1, C, H, W]
                batch_combined = torch.cat([batch_x, batch_y], dim=1).to(self.device)  # [B, 5, C, H, W]
                #batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # 计算当前iteration的总数
                iter_num = epoch * len(self.train_loader) + itr

                # 获取scheduled sampling mask
                if args.reverse_scheduled_sampling == 1:
                    # 使用反向计划采样
                    mask_true = reserve_schedule_sampling_exp(iter_num, batch_combined.shape[0], args)
                else:
                    # 使用标准计划采样
                    eta, mask_true = schedule_sampling(eta, iter_num, batch_combined.shape[0], args)

                # 前向传播
                pred_y, loss = self.model(batch_combined, mask_true, return_loss=False)
                target_y = batch_combined[:, 1:]
                loss = self.criterion(pred_y, target_y)

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
            if i >= 10:
                break

            # 合并输入和目标为5帧数据
            # batch_x: [B, 4, C, H, W], batch_y: [B, 1, C, H, W]
            batch_combined = torch.cat([batch_x, batch_y], dim=1).to(self.device)  # [B, 5, C, H, W]

            #batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            # 验证时不使用scheduled sampling
            batch_size = batch_combined.shape[0]
            mask_true = torch.zeros(
                (batch_size, self.args.total_length - 1,
                 self.args.in_shape[2] // self.args.patch_size,
                 self.args.in_shape[3] // self.args.patch_size,
                 self.args.patch_size ** 2 * self.args.in_shape[1])
            ).to(self.device)

            pred_y, _ = self.model(batch_combined, mask_true, return_loss=False)

            # ConvLSTM输出的是frames[:, 1:]，所以需要调整target
            target_y = batch_combined[:, 1:]

            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()),
                     [pred_y, target_y], [preds_lst, trues_lst]))

            loss = self.criterion(pred_y, target_y)
            vali_pbar.set_description('vali loss: {:.4f}'.format(loss.item()))
            total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)

        mse, mae = metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std, False)
        print_log('vali mse:{:.4f}, mae:{:.4f}'.format(mse, mae))

        self.model.train()
        return total_loss

    def test(self, args):
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []

        test_pbar = tqdm(self.test_loader, desc='Testing')
        for batch_x, batch_y in test_pbar:
            # 合并输入和目标为5帧数据
            # batch_x: [B, 4, C, H, W], batch_y: [B, 1, C, H, W]
            batch_combined = torch.cat([batch_x, batch_y], dim=1).to(self.device)  # [B, 5, C, H, W]
            #batch_x = batch_x.to(self.device)

            # 测试时不使用scheduled sampling
            batch_size = batch_combined.shape[0]
            mask_true = torch.zeros(
                (batch_size, self.args.total_length - 1,
                 self.args.in_shape[2] // self.args.patch_size,
                 self.args.in_shape[3] // self.args.patch_size,
                 self.args.patch_size ** 2 * self.args.in_shape[1])
            ).to(self.device)

            pred_y, _ = self.model(batch_combined, mask_true, return_loss=False)

            # 调整batch_y以匹配pred_y
            batch_y_adjusted = batch_y[:, 1:]

            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()),
                     [batch_x, batch_y_adjusted, pred_y], [inputs_lst, trues_lst, preds_lst]))

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



    def _estimate_flops(self):
      """当thop失败时，使用估算方法计算FLOPs"""
      args = self.args
    
      # 基本参数
      batch_size = 1
      time_steps = args.pre_seq_length + args.aft_seq_length - 1
      patch_height = args.in_shape[2] // args.patch_size
      patch_width = args.in_shape[3] // args.patch_size
      frame_channel = args.patch_size * args.patch_size * args.in_shape[1]
    
      total_flops = 0
    
      # 根据模型类型估算
      model_name = args.model.lower()
    
      if model_name == 'convlstm':
          # ConvLSTM: 每层4个门，每个门2个卷积
          for i in range(args.num_layers):
              in_ch = frame_channel if i == 0 else args.num_hidden[i-1]
              out_ch = args.num_hidden[i]
            
              # 每个时间步的计算
              # 输入卷积: 4 gates * (filter_size^2 * in_ch * out_ch * H * W)
              input_conv_flops = 4 * args.filter_size**2 * in_ch * out_ch * patch_height * patch_width
              # 隐藏状态卷积: 4 gates * (filter_size^2 * out_ch * out_ch * H * W)
              hidden_conv_flops = 4 * args.filter_size**2 * out_ch * out_ch * patch_height * patch_width
            
              layer_flops = (input_conv_flops + hidden_conv_flops) * time_steps
              total_flops += layer_flops
        
          # 输出层 1x1 卷积
          output_flops = args.num_hidden[-1] * frame_channel * patch_height * patch_width * time_steps
          total_flops += output_flops
        
      elif model_name == 'predrnn':
          # PredRNN: 类似ConvLSTM但有额外的memory流
          for i in range(args.num_layers):
              in_ch = frame_channel if i == 0 else args.num_hidden[i-1]
              out_ch = args.num_hidden[i]
            
              # 基础ConvLSTM计算
              input_conv_flops = 4 * args.filter_size**2 * in_ch * out_ch * patch_height * patch_width
              hidden_conv_flops = 4 * args.filter_size**2 * out_ch * out_ch * patch_height * patch_width
            
              # Memory更新的额外计算 (约增加50%)
              memory_flops = 2 * args.filter_size**2 * out_ch * out_ch * patch_height * patch_width
            
              layer_flops = (input_conv_flops + hidden_conv_flops + memory_flops) * time_steps
              total_flops += layer_flops
        
          # 输出层
          output_flops = args.num_hidden[-1] * frame_channel * patch_height * patch_width * time_steps
          total_flops += output_flops
        
      elif model_name == 'mim':
          # MIM: 第一层是SpatioTemporalLSTM，其他层是MIMBlock + MIMN
          # 第一层
          in_ch = frame_channel
          out_ch = args.num_hidden[0]
          first_layer_flops = 6 * args.filter_size**2 * in_ch * out_ch * patch_height * patch_width * time_steps
          first_layer_flops += 6 * args.filter_size**2 * out_ch * out_ch * patch_height * patch_width * time_steps
          total_flops += first_layer_flops
        
          # 其他层
          for i in range(1, args.num_layers):
              in_ch = args.num_hidden[i-1]
              out_ch = args.num_hidden[i]
            
              # MIMBlock (类似SpatioTemporalLSTM + 差分处理)
              mim_block_flops = 7 * args.filter_size**2 * in_ch * out_ch * patch_height * patch_width * time_steps
              mim_block_flops += 7 * args.filter_size**2 * out_ch * out_ch * patch_height * patch_width * time_steps
            
              # MIMN (3个门)
              if i < args.num_layers - 1:
                  mimn_flops = 3 * args.filter_size**2 * in_ch * args.num_hidden[i] * patch_height * patch_width * time_steps
                  total_flops += mimn_flops
            
              total_flops += mim_block_flops
        
          # 输出层
          output_flops = args.num_hidden[-1] * frame_channel * patch_height * patch_width * time_steps
          total_flops += output_flops
        
      elif model_name == 'predrnnv2':
          # PredRNNv2: 类似PredRNN但有自适应模块
          for i in range(args.num_layers):
              in_ch = frame_channel if i == 0 else args.num_hidden[i-1]
              out_ch = args.num_hidden[i]
            
              # 基础计算
              layer_flops = 6 * args.filter_size**2 * in_ch * out_ch * patch_height * patch_width * time_steps
              layer_flops += 6 * args.filter_size**2 * out_ch * out_ch * patch_height * patch_width * time_steps
            
              # Adapter的额外计算
              adapter_flops = 2 * args.num_hidden[0]**2 * patch_height * patch_width * time_steps
            
              total_flops += layer_flops + adapter_flops
        
          # 输出层
          output_flops = args.num_hidden[-1] * frame_channel * patch_height * patch_width * time_steps
          total_flops += output_flops
    
      # 格式化输出
      def format_flops(flops):
          if flops >= 1e12:
              return f"{flops/1e12:.2f} TFLOPs"
          elif flops >= 1e9:
              return f"{flops/1e9:.2f} GFLOPs"
          elif flops >= 1e6:
              return f"{flops/1e6:.2f} MFLOPs"
          else:
              return f"{flops:.0f} FLOPs"
    
      print_log(f"\nFLOPs分析 (估算):")
      print_log(f"总FLOPs: {format_flops(total_flops)}")
      print_log(f"平均每帧FLOPs: {format_flops(total_flops / time_steps)}")
    
      # 与参数量对比
      total_params = sum(p.numel() for p in self.model.parameters())
      if total_params > 0:
          flops_per_param = total_flops / total_params
          print_log(f"FLOPs/参数比: {flops_per_param:.2f}")