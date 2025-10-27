from .dataloader_taxibj import load_data as load_taxibj
from .dataloader_moving_mnist import load_data as load_mmnist
from .dataloader_scs import load_data as load_scs

def load_data(batch_size, val_batch_size, data_root, num_workers, dataname, **kwargs):
    if dataname == 'taxibj':
        return load_taxibj(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'mmnist':
        return load_mmnist(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'custom':
        return load_scs(batch_size, val_batch_size, data_root, num_workers)