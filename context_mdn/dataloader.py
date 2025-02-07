import numpy as np
import torch
import random
import sys
import copy

sys.path.append("/workspace/repos/datasets")

from dataset_utils import MOTION_DICT
from torch.utils.data import Dataset, Sampler
from termcolor import colored


class Context_DataLoader(Dataset):
    
    def __init__(self, cfg, path, typ, size):
        
        super(Context_DataLoader, self).__init__()
        
        self.num_epochs = cfg.train_params['num_total_epochs']
        self.size = size
        self.data_splits = {motion: [] for motion in MOTION_DICT.values()}
        self.useable_data = []
        self.upper_bound = cfg.train_params['auto_increment_upper_bound']
        self.lower_bound = cfg.train_params['auto_increment_lower_bound']
        
        # Load data
        print(colored(f"Loading {typ} dataset", 'green'))
        self.raw_data = np.load(file=path, allow_pickle=True)
        self.data_size = len(self.raw_data)
        
        if self.size < 1.0:
            
            subset_size = int(self.size * self.data_size)
            self.data_storage= random.sample(population=self.raw_data, k=subset_size)
            
        else:
            
            # if typ == 'test': self.data_storage = sorted(self.raw_data, key=lambda x: x["source"])
            # else: self.data_storage = self.raw_data
            
            self.data_storage = self.raw_data
            
        if cfg.train_params['auto_increment'] and typ == 'train':
            
            # Split by motion label
            self.split_by_motion()
            self.update(epoch=1)
            
        else:
        
            self.useable_data = copy.deepcopy(self.data_storage)
            self.data_size = len(self.useable_data)
            
        print(colored(f"Loading {typ} dataset completed: {self.data_size } samples", 'green'))
        return
    
    
    def update(self, epoch):
        
        # Reset
        self.useable_data = []
        scale = (epoch*1.5)/(self.num_epochs)
        
        for _, samples in self.data_splits.items():
            
            # Calc new size
            new_size = int(len(samples) * scale)
            
            if len(samples) < self.lower_bound: new_size = len(samples)
            elif new_size < self.lower_bound: new_size = int(self.lower_bound)
            elif new_size > self.upper_bound: new_size = int(self.upper_bound)
            elif new_size >= len(samples): new_size = len(samples)
            
            # Extract subset from source
            self.useable_data.extend(random.sample(samples, new_size))
        
        # Shuffle data
        random.shuffle(self.useable_data)
        self.data_size = len(self.useable_data)
        return
    
    
    def split_by_motion(self):
        """
        Splits the dataset into subsets based on the motion state label.
        
        Returns:
            A dictionary where keys are motion names (strings) and
            values are lists of sample dicts with that motion label.
        """
        
        for sample in self.data_storage:
            
            motion_label = sample['target']['motion_state']
            self.data_splits[motion_label].append(sample)
            
        return
    
    
    def __len__(self):
        
        return len(self.useable_data)
    
    
    def __getitem__(self, idx):
        
        return self.useable_data[idx]
    
    
class RandomSubsetSampler(Sampler):
    
    def __init__(self, dataset, subset_size):
        
        self.size = subset_size
        self.dataset = dataset
        self.num_samples = len(dataset)
        
    def __iter__(self):
        
        subset_size = int(self.num_samples * self.size)
        return iter(random.sample(range(self.num_samples), subset_size))
    
    def __len__(self):
        
        return int(self.num_samples * self.size)


def collate_fn(batch, cfg, cord_convs_batched):
    
    obs_len = cfg.model_params['input_horizon'] 
    output_horizons = cfg.train_params['output_horizons']
    input_features = cfg.model_params['lstm_input_dim']
    output_features = cfg.model_params['gt_feature_dim']
    
    B = len(batch)
    N = torch.tensor(np.array([len(obj['others']) for obj in batch])).max()
    T = obs_len
    
    # Target track data
    target_past_traj = torch.tensor(np.array([obj['target']['track'][:T] for obj in batch]), dtype=torch.float32)
    target_future_traj = torch.tensor(np.array([obj['target']['track'][[x + T for x in output_horizons], :2] for obj in batch]), dtype=torch.float32)
    target_info = [{"motion_state": obj['target']['motion_state'], "source": obj['source'], "rotation": obj['rotation'], "translation": obj['translation'], "lsa": obj['lsa']} for obj in batch]
    
    # Cost grid
    # Add cord-convs
    occupancy_grid = torch.tensor(np.array([obj['cost_map'] for obj in batch]), dtype=torch.float32).unsqueeze(1)
    cost_grid = torch.cat([occupancy_grid, cord_convs_batched], dim=1)
    
    # Vehicle track data
    vehicle_data = [obj['vehicles'] for obj in batch]
    
    # Others track data and masking
    # At least one other in scenes of current batch
    T_others = obs_len
    if N != 0: 
        
        others_past_traj = torch.zeros(B, N, T_others, input_features)
        others_future_traj = torch.zeros(B, N, len(output_horizons), output_features)
        
        # Initialize the mask with True (agent does not exist) (n+1) for target
        others_padding_mask = torch.ones((B, N+1), dtype=torch.bool)
        
        # Fill the tensor with the "tracks"
        for i, element in enumerate(batch):
            
            # Set to False for existing agents
            others_padding_mask[i, :len(element['others'])+1] = False
            
            for j, other in enumerate(element['others']):
                
                others_past_traj[i, j] = torch.nan_to_num(torch.tensor(other['track'][:T]), nan=0.0)
                others_future_traj[i, j] = torch.tensor(other['track'][[x + T for x in output_horizons], :2])
                
    # No other in scenes of current batch
    else:
        
        others_past_traj = torch.zeros(B, 1, T_others, input_features)
        others_future_traj = torch.zeros(B, 1, len(output_horizons), output_features)
        
        # Initialize the mask with True (agent does not exist) (n+1) for target
        others_padding_mask = torch.ones((B, 2), dtype=torch.bool)
        others_padding_mask[:,0] = False
        
    # past_traj: (B, seq_len, 4)
    # cost_grid: (B, 3, H, W)
    # future_traj: (B, future_len, 2)
    return target_past_traj, target_future_traj, others_past_traj, others_future_traj, others_padding_mask, vehicle_data, cost_grid, target_info