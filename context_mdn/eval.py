import torch
import sys
import os
import numpy as np

from tqdm import tqdm
from termcolor import colored
from functools import partial, reduce
from dataloader import Context_DataLoader, collate_fn
from torch.utils.data import DataLoader
from models import ContextMDN
from prediction_utils import ConfigLoader, parse_args, build_mesh_grid
from prob import build_mixture_distribution, build_confidence_set, compute_reliability, compute_sharpness_sets, compute_sharpness, build_forecast
from vis import visualize_forecast
from metrics import calc_ade_fde_mode, calc_ade_fde_k, calc_ade_fde_k_with_probs, calc_ade_fde_k_for_cls, get_final_results
from voronoi import Confidence_Level


torch.backends.cudnn.benchmark = True


def test_process(cfg, test_loader, model, target_mesh_grid, device):
    
    # Eval
    model.eval()
    reliability_confidence_sets = []
    sharpness_sets = []
    deterministic_scores = []
    
    print(colored(f"Start Testing", 'green'))
    cfg.logger.info(f"Start Testing")
    
    with torch.no_grad():
        
        for batch_idx, (target_past_traj, target_future_traj, others_past_traj, others_future_traj, others_padding_mask, _, cost_grid, target_info) in enumerate(tqdm(test_loader, desc="Test_Batches")):
        
            # Load data to device
            cost_grid = cost_grid.to(device)
            target_past_traj = target_past_traj.to(device)
            target_future_traj = target_future_traj.to(device)
            others_past_traj = others_past_traj.to(device)
            others_padding_mask = others_padding_mask.to(device)
            
            # Forward pass
            output = model(
                target_past_traj,
                others_past_traj, 
                others_padding_mask, 
                cost_grid
            )
            
            # # Create mixture distribution
            # mixture_dists, _ = build_mixture_distribution(
            #     cfg=cfg,
            #     output=output, 
            #     num_gaussians=cfg.model_params['mdn_num_gaussians']
            # )
            
            # # Compute target confidence sets
            # S, _ = mixture_dists.mixture_distribution.logits.shape
            # target_conf_maps, _ = build_confidence_set(
            #     gmm=mixture_dists, 
            #     target=target_future_traj.view(S,2), 
            #     n_samples=cfg.train_params['num_samples']
            # )
            
            # # Compute grid confidence sets
            # mesh_conf_maps, means = build_confidence_set(
            #     gmm=mixture_dists, 
            #     target=target_mesh_grid, 
            #     n_samples=cfg.train_params['num_samples']
            # )
            
            # reliability_confidence_sets.append(target_conf_maps.cpu().numpy())
            
            # # Get modes, fix undefined modes due to empty conf map
            # modes = target_mesh_grid[torch.argmin(mesh_conf_maps, dim=0, keepdim=True)]
            # modes = modes[0,:,0,:]     
            # mask = (modes == torch.tensor([-10.0, -10.0]).to(modes.device)).all(dim=1)
            # modes[mask] = means[mask]
            
            #confidence_levels = []
            
            # # Compute defined confidence levels
            # for kappa in cfg.train_params["confidence_levels"]:
                
            #     conf_level = Confidence_Level(
            #         cfg=cfg,
            #         kappa=kappa,
            #         conf_maps=mesh_conf_maps,
            #         modes=modes,
            #         device=device
            #     )
                
            #     confidence_levels.append(conf_level)
                
            #     samples = conf_level.voronoi_sampling()
                
            #     a = 5
            
            # # Compute sharpness sets
            # sharpness_sets.append(compute_sharpness_sets(
            #     cfg=cfg,
            #     gmm=mixture_dist,
            #     target=target_mesh_grid,
            #     n_samples=cfg.train_params['num_samples']
            # ).cpu().numpy())
            
            # ade_mode, fde_mode = calc_ade_fde_mode(
            #     gt=target_future_traj,
            #     gmm=mixture_dist,
            #     batch_size=cfg.test_params["batch_size"],
            #     n_forecast_horizons=cfg.model_params["forecast_horizon"],
            #     n_features=cfg.model_params["mdn_num_gaussians"]
            # )
            
            # ade_k_100, fde_k_100 = calc_ade_fde_k(
            #     gt=target_future_traj,
            #     gmm=mixture_dist,
            #     k=20,
            #     batch_size=cfg.test_params["batch_size"],
            #     n_forecast_horizons=cfg.model_params["forecast_horizon"],
            #     n_features=2
            # )
            
            # ade_k_95, fde_k_95 = calc_ade_fde_k_for_cls(
            #     gt=target_future_traj,
            #     gmm=mixture_dist,
            #     k=20,
            #     n_sample_points=cfg.train_params['num_samples'],
            #     batch_size=cfg.test_params["batch_size"],
            #     n_forecast_horizons=cfg.model_params["forecast_horizon"],
            #     n_features=2,
            #     confidence_level=0.95
            # )
            
            # ade_k_68, fde_k_68 = calc_ade_fde_k_for_cls(
            #     gt=target_future_traj,
            #     gmm=mixture_dist,
            #     k=20,
            #     n_sample_points=cfg.train_params['num_samples'],
            #     batch_size=cfg.test_params["batch_size"],
            #     n_forecast_horizons=cfg.model_params["forecast_horizon"],
            #     n_features=2,
            #     confidence_level=0.68
            # )
            
            # ade_kp, ade_probs, fde_kp, fde_probs = calc_ade_fde_k_with_probs(
            #     gt=target_future_traj,
            #     gmm=mixture_dist,
            #     k=20,
            #     batch_size=cfg.test_params["batch_size"],
            #     n_forecast_horizons=cfg.model_params["forecast_horizon"],
            #     n_features=2
            # )
            
            # deterministic_scores.append(torch.stack([ade_mode, fde_mode, ade_k_100, fde_k_100, ade_k_95, fde_k_95, ade_k_68, fde_k_68]))
            
            # Plot test dataset example forecasts
            if cfg.test_params['plot_examples'] and batch_idx % cfg.test_params['plot_batch_step'] == 0:
                
                visualize_forecast(
                        cfg=cfg,
                        output=output[0],
                        past_traj=target_past_traj[0], 
                        future_traj=target_future_traj[0], 
                        target_info=target_info[0],
                        past_others=others_past_traj[0], 
                        future_others=others_future_traj[0], 
                        others_mask=others_padding_mask[0], 
                        cost_grid=cost_grid[0,0], 
                        grid=target_mesh_grid, 
                        sample_id=batch_idx, 
                        dst_dir=cfg.test_ego_examples_path,
                        epoch=0
                    )
        
        # ADE/FDE
        ade_fde_scores = get_final_results(
            scores=deterministic_scores, 
            k=20, 
            cls=cfg.train_params["confidence_levels"]
        )
        
        # Reliability
        reliability_scores = compute_reliability(
            confidence_sets=np.vstack(reliability_confidence_sets),
            batch_size=cfg.test_params["batch_size"], 
            forecast_horizon=cfg.model_params["forecast_horizon"],
            epoch=0,
            with_plot=True,
            cfg=cfg,
            dst_dir=cfg.testing_path
        )
        
        # Sharpness
        sharpness_scores = compute_sharpness(
            cfg=cfg, 
            sharpness_sets=np.vstack(sharpness_sets)
        )
        
        scores_dict = reduce(lambda a, b: {**a, **b}, [ade_fde_scores, reliability_scores, sharpness_scores])
        print(colored(scores_dict, "green"))
        cfg.logger.info(f"{scores_dict}")
        
    return



def run_test(args):
    
    # set gpu and load config
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = ConfigLoader(args=args, work_dir=os.getcwd(), regular=True)
    print(colored(f"Testing for: {args.target} using config: {args.cfg} on gpu: {args.gpu}", 'green'))
    cfg.logger.info(f"Testing for: {args.target} using config: {args.cfg} on gpu: {args.gpu}")
    
    # Build mesh grids
    target_mesh_grid, target_x_layer, target_y_layer = build_mesh_grid(
        mesh_range_x=cfg.grid_params['grid_size_meter'],
        mesh_range_y=cfg.grid_params['grid_size_meter'], 
        mesh_resolution=cfg.target_grid_resolution
    )
    
    # Create cord conv representation
    cord_convs = torch.stack([target_x_layer, target_y_layer], dim=0).unsqueeze(0)
    cord_convs_batched = cord_convs.expand(cfg.test_params['batch_size'], cord_convs.shape[1], cfg.grid_params['target_grid_size_cells'], cfg.grid_params['target_grid_size_cells'])
    target_mesh_grid = target_mesh_grid.to(device)
    
    
    # Dataset specific test data dataloader
    test_dataset = Context_DataLoader(
        cfg=cfg,
        path=cfg.paths['test_data_path'], 
        typ='test',
        size=1.0,
    )
    
    # Dataset collate function
    collate_fn_with_params = partial(
        collate_fn, 
        cfg=cfg,
        cord_convs_batched=cord_convs_batched,
    )
    
    # Test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test_params['batch_size'],
        shuffle=False,
        collate_fn=collate_fn_with_params,
        drop_last=True,
        pin_memory=True,
        # num_workers=cfg.train_params['dataloader_num_workers'],
        # prefetch_factor=cfg.train_params['dataloader_prefetch'],
        # persistent_workers=True
    )
    
    print(colored(f"Loaded test data: {len(test_dataset)} samples"), 'green')
    cfg.logger.info(f"Loaded test data: {len(test_dataset)} samples")
    
    # Model
    model = ContextMDN(cfg=cfg)
    
    # Load model weights
    model.load_state_dict(torch.load(f=os.path.join(cfg.checkpoint_path, "model_weights_0060.pt"), map_location=device)["model"])
    model.to(device)
    
    test_process(
        cfg=cfg,
        test_loader=test_loader,
        model=model,
        target_mesh_grid=target_mesh_grid,
        device=device
        )
    
    # Close logging
    cfg.log_file_handler.close()
    cfg.logger.removeHandler(cfg.log_file_handler)
    
    print(colored(f'End testing, shutting down...', 'green'))
    
    return
