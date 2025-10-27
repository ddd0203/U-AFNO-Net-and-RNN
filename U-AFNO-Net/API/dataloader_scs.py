import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
import h5py
import math
import os


def reshape_fields(img):
    """
    转换数据格式：
    - input: (C,T,H,W) -> (T,C,H,W)
    - target: (C,H,W) -> (1,C,H,W)
    """
    if len(np.shape(img)) == 3:  # target case: (C,H,W)
        img = np.expand_dims(img, axis=1)  # (C,H,W) -> (C,1,H,W)
        img = np.transpose(img, (1, 0, 2, 3))  # (C,1,H,W) -> (1,C,H,W)
    else:  # input case: (C,T,H,W)
        img = np.transpose(img, (1, 0, 2, 3))  # (C,T,H,W) -> (T,C,H,W)

    return torch.as_tensor(img)


def load_data(batch_size, val_batch_size, data_root, num_workers, dataname='custom'):
    train_data_root = os.path.join(data_root, "train")
    valid_data_root = os.path.join(data_root, "valid")
    test_data_root = os.path.join(data_root, "test")
    train_set = GetDataset(train_data_root)
    valid_set = GetDataset(valid_data_root)
    test_set = GetDataset(test_data_root)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_validation = torch.utils.data.DataLoader(
        valid_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    print(len(train_set), len(test_set))
    print(len(dataloader_train), len(dataloader_validation), len(dataloader_test))

    return dataloader_train, dataloader_validation, dataloader_test




class GetDataset(Dataset):
    def __init__(self, location, data_name='custom', n_step_finetune=False, n_steps=2):
        self.location = location
        self.data_name = data_name
        self.dt = 2
        self.n_history = 3
        self.mean = 0
        self.std = 1
        self.n_step_finetune = n_step_finetune
        self.n_steps = n_steps if n_step_finetune else 1
        self._get_files_stats()

    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/*.h5")
        self.files_paths.sort()
        self.n_years = len(self.files_paths)
        with h5py.File(self.files_paths[0], 'r') as _f:
            logging.info("Getting file stats from {}".format(self.files_paths[0]))
            self.n_in_channels = _f['fields'].shape[0]
            self.n_samples_per_year = _f['fields'].shape[1]
            # original image shape (before padding)
            self.img_shape_x = _f['fields'].shape[2]  # just get rid of one of the pixels
            self.img_shape_y = _f['fields'].shape[3]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]
        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location,
                                                                                                       self.n_samples_total,
                                                                                                       self.img_shape_x,
                                                                                                       self.img_shape_y,
                                                                                                       self.n_in_channels))
        logging.info("Delta t: {} hours".format(self.dt * 3))
    


    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], 'r')
        self.files[year_idx] = _file
    

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        year_idx = int(global_idx / self.n_samples_per_year)  # which year we are on
        local_idx = int(global_idx % self.n_samples_per_year)
        # open image file
        
        if self.files[year_idx] is None:
            self._open_file(year_idx)
        
        #file_path = self.files_paths[year_idx]

         # if we are not at least self.dt*n_history timesteps into the prediction
        if local_idx < self.dt * self.n_history:
            local_idx += self.dt * self.n_history

        if self.n_step_finetune:
          max_allowed_idx = self.n_samples_per_year - self.dt * self.n_steps
          if local_idx >= max_allowed_idx:
            local_idx = max_allowed_idx - self.dt
          inp = reshape_fields(self.files[year_idx]['fields'][:, (local_idx - self.dt * self.n_history):(local_idx + 1):self.dt])
         
          target_indices = []
          for step in range(1, self.n_steps + 1):
              target_idx = local_idx + step * self.dt
              if target_idx < self.n_samples_per_year:
                  target_indices.append(target_idx)
              else:
                  # 如果超出范围，使用最后一个可用的时间步
                  target_indices.append(self.n_samples_per_year - self.dt) 
          targets = []
          for idx in target_indices:
              targets.append(self.files[year_idx]['fields'][:, idx])
          # 将targets堆叠成 (C, N, H, W) 然后转换为 (N, C, H, W)
          targets = np.stack(targets, axis=1)  # (C, N, H, W)
          tar = reshape_fields(targets)  # (N, C, H, W) 

        else:
         # if we are on the last image in a year predict identity, else predict next timestep
          step = 0 if local_idx >= self.n_samples_per_year - self.dt else self.dt
          inp = reshape_fields(self.files[year_idx]['fields'][:, (local_idx - self.dt * self.n_history):(local_idx + 1):self.dt])
          tar = reshape_fields(self.files[year_idx]['fields'][:, local_idx + step])
        return inp, tar

   



def test_dataloader():
   
    data_root = "/content/drive/MyDrive/UAFNO/data/256x256_h5new/"  #训练数据路径
    
    print("测试开始...")
    
    try:
        # 1.测试单个Dataset
        print("测试Dataset...")
        train_path = os.path.join(data_root, "valid")
        print(train_path)
        dataset = GetDataset(train_path)
        print(f"✓ 数据集大小: {len(dataset)}")
        
        # 2.测试获取样本
        print("测试样本...")
        inp, tar = dataset[0]
        print(f"✓ Input shape: {inp.shape}")
        print(f"✓ Target shape: {tar.shape}")
        
        # 3.测试DataLoader
        print("测试DataLoader...")
        train_loader, val_loader, test_loader = load_data(
            batch_size=16, 
            val_batch_size=16,
            data_root=data_root,
            num_workers=0,
            dataname="custom"
        )
        
        # 4. 测试一个batch
        print("测试batch...")
        inputs, targets = next(iter(train_loader))
        print(f"✓ Batch inputs shape: {inputs.shape}")   # 应该是 (B,T,C,H,W)
        print(f"✓ Batch targets shape: {targets.shape}")  # 应该是 (B,1,C,H,W)
        
        print("测试通过!")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

#test_dataloader()