import torch
import os
import json
import argparse
import logging
import numpy as np
import seaborn as sns
import shutil
from termcolor import colored


class ConfigLoader:
    """ Class loading parameters and and paths from external config file
    """
    
    def __init__(self, args, work_dir, regular):
        """ Init and setup 
        """
        
        # load config from file
        print(colored(f"Loading config data", 'green'))
        
        if regular: config_path = os.path.join(work_dir, 'configs', args.target, args.cfg)
        else: config_path = os.path.join(work_dir, args.cfg)
            
        with open(config_path) as f:
            cfg = json.load(f)
        
        # variables and paths
        self.target = args.target
        self.data_target = os.path.join(args.target, args.cfg.split('.')[0])
        self.with_print = args.print
        self.paths = cfg['paths']
        self.grid_params = cfg['grid_params']
        self.model_params = cfg['model_params']
        self.train_params = cfg['train_params']
        self.test_params = cfg['test_params']
        
        # Grid variables
        self.target_grid_resolution = (2 * self.grid_params['grid_size_meter']) / self.grid_params['target_grid_size_cells']
        
        # eval and plotting
        self.reliability_bins = [k for k in np.arange(0.0, 1.01, 0.01)]
        self.colors_rgb = np.array(sns.color_palette(palette='colorblind', n_colors=self.model_params['forecast_horizon']))[::-1,:]
        #self.colors_rgb = np.array(sns.color_palette(palette='YlOrRd', n_colors=self.model_params['forecast_horizon']))[::-1,:]
        
        # paths
        self.result_path = os.path.join(self.paths['result_path'], self.data_target)
        self.checkpoint_path = os.path.join(self.result_path, 'checkpoints')
        self.evaluation_path = os.path.join(self.result_path, 'evaluation')
        self.eval_ego_examples_path = os.path.join(self.result_path, 'evaluation', 'examples', 'ego')
        self.eval_world_examples_path = os.path.join(self.result_path, 'evaluation', 'examples', 'world')
        self.testing_path = os.path.join(self.result_path, 'testing')
        self.test_ego_examples_path = os.path.join(self.result_path, 'testing', 'examples', 'ego')
        self.test_world_examples_path = os.path.join(self.result_path, 'testing', 'examples', 'world')
        
        # create result directory structure
        if not os.path.exists(self.result_path): os.makedirs(self.result_path) 
        if not os.path.exists(self.checkpoint_path): os.makedirs(self.checkpoint_path)
        if not os.path.exists(self.evaluation_path): os.makedirs(self.evaluation_path)
        if not os.path.exists(self.eval_ego_examples_path): os.makedirs(self.eval_ego_examples_path)
        if not os.path.exists(self.eval_world_examples_path): os.makedirs(self.eval_world_examples_path)
        if not os.path.exists(self.testing_path): os.makedirs(self.testing_path)
        if not os.path.exists(self.test_ego_examples_path): os.makedirs(self.test_ego_examples_path)
        if not os.path.exists(self.test_world_examples_path): os.makedirs(self.test_world_examples_path)
        
        # Save config file to dest dir
        if regular: shutil.copyfile(src=config_path, dst=os.path.join(self.result_path, args.cfg))
        
        # Logging
        log_file_path = os.path.join(self.result_path, f'{args.mode}.log')
        os.remove(log_file_path) if os.path.exists(log_file_path) else None        
        self.logger = logging.getLogger(f'{args.mode}')
        self.logger.setLevel(logging.INFO)
        self.log_file_handler = logging.FileHandler(log_file_path)
        self.log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(self.log_file_handler)
        
        print(colored(f"Loading config data completed", 'green'))
        return


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', help='Mode: train or test')
    parser.add_argument('-t', '--target', type=str, default='imptc', help='Target dataset to use')
    parser.add_argument('-c', '--cfg', type=str, default='imptc.json', help='Config file to use')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU ID to use')
    parser.add_argument('-p', '--print', action='store_true')
    return parser.parse_args()


def build_mesh_grid(mesh_range_x, mesh_range_y, mesh_resolution):
    
    # build grid
    steps = int(((mesh_range_x + mesh_range_y) / mesh_resolution))
    xs = torch.linspace(-mesh_range_x, mesh_range_x, steps=steps)
    ys = torch.linspace(-mesh_range_y, mesh_range_y, steps=steps)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    grid = torch.stack([x, y],dim=-1)
    grid = grid.reshape((-1,2))[:,None,:]
    return grid, x, y


def save_model(model, optimizer, epoch, dst_dir, prefix=None):
    """Save model
    """
    
    # create checkpoint
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    
    if prefix is not None:
        
        torch.save(obj=checkpoint, f=os.path.join(dst_dir, prefix))
        
    else:
    
        torch.save(obj=checkpoint, f=os.path.join(dst_dir, f"model_weights.pt"))
    
    return
