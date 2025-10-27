from .dataloader_scs import load_data as load_scs

def load_data(batch_size, val_batch_size, data_root, num_workers, dataname, **kwargs):
    if dataname == 'custom':
      return load_scs(batch_size, val_batch_size, data_root, num_workers)