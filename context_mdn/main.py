import torch
import sys
import os
import numpy as np

sys.path.append("/workspace/repos/datasets")
sys.path.append("/workspace/repos/prediction")

from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
from termcolor import colored
from functools import partial
from dataloader import Context_DataLoader, collate_fn
from torch.utils.data import DataLoader
from models import ContextMDN
from prediction_utils import ConfigLoader, parse_args, build_mesh_grid, save_model
from prob import compute_loss, build_mixture_distribution, build_confidence_set, compute_reliability
from vis import visualize_forecast, plot_loss
from eval import run_test


torch.backends.cudnn.benchmark = True


def training_process(cfg, train_dataset, train_loader, eval_loader, model, optimizer, scheduler, target_mesh_grid, train_size, eval_size, device):
    
    train_loss_list = []
    train_reliability_list = []
    eval_loss_list = []
    eval_reliability_list = []
    eval_score_list = []
    best_epoch = 0
    best_eval_score = float('-inf')
    
    print(colored(f"Start Training", 'green'))
    cfg.logger.info(f"Start Training")
    
    for epoch in range(0, cfg.train_params['num_total_epochs']+1, 1):
        
        model.train()
        train_loss = 0.0
        train_nll_loss = 0.0
        train_cost_loss = 0.0
        train_reliability_confidence_sets = []
        
        print(colored(f"Epoch [{epoch}/{cfg.train_params['num_total_epochs']}]:", 'green'))
        cfg.logger.info(f"Epoch [{epoch}/{cfg.train_params['num_total_epochs']}]:")
        
        # With auto train data increment
        if cfg.train_params['auto_increment']: 
            
            train_dataset.update(epoch=epoch)
        
        # past_traj: (batch_size, input_horizon, input_dim)
        # occupancy_grid: (batch_size, 1, grid_size_cells, grid_size_cells)
        # future_traj: (batch_size, forecast_horizon, 2)
        for batch_idx, (target_past_traj, target_future_traj, others_past_traj, others_future_traj, others_padding_mask, _, cost_grid, target_info) in enumerate(tqdm(train_loader, desc="Training Batches")):
            
            # Load data to device
            cost_grid = cost_grid.to(device)
            target_past_traj = target_past_traj.to(device)
            target_future_traj = target_future_traj.to(device)
            others_past_traj = others_past_traj.to(device)
            others_padding_mask = others_padding_mask.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(
                target_past_traj,
                others_past_traj, 
                others_padding_mask, 
                cost_grid,
            )
            
            # Create mixture distribution
            mixture_dist, sucess = build_mixture_distribution(
                cfg=cfg,
                output=output, 
                num_gaussians=cfg.model_params['mdn_num_gaussians']
                )
            
            # Check if the output contains invalids or mixture cant be build due to divergence
            if not sucess:
                
                print(colored(f"Training diverged at epoch: {epoch}, exit training...", 'red'))
                cfg.logger.info(f"Training diverged at epoch: {epoch}, exit training...")
                sys.exit()
            
            # Compute reliability sets
            s, _ = mixture_dist.mixture_distribution.logits.shape
            train_reliability_confidence_sets.append(build_confidence_set(
                gmm=mixture_dist, 
                target=target_future_traj.view(s,2), 
                n_samples=cfg.train_params['num_samples']
                )[0].cpu().numpy())
            
            # Train loss
            loss, nll, cost_error = compute_loss(
                cfg=cfg,
                mixture_dist=mixture_dist, 
                target=target_future_traj,
                mesh_grid=target_mesh_grid,
                cost_grid=cost_grid[:,0].unsqueeze(1), 
            )
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping
            if cfg.train_params['clip_grad_norm'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train_params['clip_grad_norm'])
            
            # Forward
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item()
            train_nll_loss += nll
            train_cost_loss += cost_error
        
        # Compute epoch loss values
        train_loss /= train_size
        train_nll_loss /= train_size
        train_cost_loss /= train_size
        
        # Compute train data reliability
        train_reliability_scores = compute_reliability(
            confidence_sets=np.vstack(train_reliability_confidence_sets), 
            batch_size=cfg.train_params['batch_size'], 
            forecast_horizon=cfg.model_params['forecast_horizon'],
            with_plot=False,
            cfg=cfg
            )
        
        train_loss_list.append([train_loss, train_nll_loss, train_cost_loss])
        train_reliability_list.append([train_reliability_scores['avg_RLS'], train_reliability_scores['min_RLS']])
        
        # Save current weights
        save_model(
            model=model, 
            optimizer=optimizer,
            epoch=epoch, 
            dst_dir=cfg.checkpoint_path, 
            prefix=f"model_weights_latest.pt"
            )
        
        
        # Eval
        model.eval()
        eval_loss = 0.0
        eval_nll_loss = 0.0
        eval_cost_loss = 0.0
        eval_reliability_confidence_sets = []
        
        with torch.no_grad():
            
            for batch_idx, (target_past_traj, target_future_traj, others_past_traj, others_future_traj, others_padding_mask, _, cost_grid, target_info) in enumerate(tqdm(eval_loader, desc="Eval_Batches")):
            
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
                
                # Create mixture distribution
                mixture_dist, sucess = build_mixture_distribution(
                    cfg=cfg,
                    output=output, 
                    num_gaussians=cfg.model_params['mdn_num_gaussians']
                    )
                
                # Compute reliability sets
                s, _ = mixture_dist.mixture_distribution.logits.shape
                eval_reliability_confidence_sets.append(build_confidence_set(
                    gmm=mixture_dist, 
                    target=target_future_traj.view(s,2), 
                    n_samples=cfg.train_params['num_samples']
                )[0].cpu().numpy())
                
                # Eval loss
                loss, nll, cost_error = compute_loss(
                    cfg=cfg,
                    mixture_dist=mixture_dist, 
                    target=target_future_traj,
                    mesh_grid=target_mesh_grid,
                    cost_grid=cost_grid[:,0].unsqueeze(1),
                )
                
                # Accumulate loss
                eval_loss += loss.item()
                eval_nll_loss += nll
                eval_cost_loss += cost_error
                
                # Plot eval dataset example forecasts
                if cfg.train_params['plot_examples'] and \
                    epoch % cfg.train_params['plot_epoch_step'] == 0 and \
                    batch_idx % cfg.train_params['plot_batch_step'] == 0 and \
                    epoch != 0:
                    
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
                        dst_dir=cfg.eval_ego_examples_path,
                        epoch=epoch
                    )
                
                
            # Average loss for the epoch
            eval_loss /= eval_size
            eval_nll_loss /= eval_size
            eval_cost_loss /= eval_size
            
            # Schedular step
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            
            # Compute eval data reliability
            eval_reliability_scores = compute_reliability(
                confidence_sets=np.vstack(eval_reliability_confidence_sets), 
                batch_size=cfg.train_params['batch_size'], 
                forecast_horizon=cfg.model_params['forecast_horizon'],
                epoch=epoch,
                with_plot=True,
                cfg=cfg,
                dst_dir=cfg.evaluation_path
                )
            
            eval_loss_list.append([eval_loss, eval_nll_loss, eval_cost_loss])
            eval_reliability_list.append([eval_reliability_scores['avg_RLS'], eval_reliability_scores['min_RLS']])
            
            # Save model if new best reliability score exists
            eval_score = (eval_reliability_scores['min_RLS'] - (50*eval_loss) - (25*train_loss)) / 3
            eval_score_list.append(eval_score)
            if eval_score >= best_eval_score:
                
                save_model(
                    model=model, 
                    optimizer=optimizer, 
                    epoch=epoch, 
                    dst_dir=cfg.checkpoint_path,
                    prefix=f"model_weights_best.pt"
                    )
                
                print(colored(f"Saving model: Eval Score: {eval_score:.2f} % > {best_eval_score:.2f} %", 'green'))
                cfg.logger.info(f"Saving model: Eval Score: {eval_score:.2f} % > {best_eval_score:.2f} %")
                best_eval_score = eval_score
                best_epoch = epoch
                
            # Auto save
            if epoch % cfg.train_params['auto_save_step'] == 0 and epoch != 1:
                
                save_model(
                    model=model, 
                    optimizer=optimizer, 
                    epoch=epoch, 
                    dst_dir=cfg.checkpoint_path,
                    prefix=f"model_weights_{str(epoch).zfill(4)}.pt"
                    )
                
                print(colored(f"Saving model: Autosave", 'green'))
                cfg.logger.info(f"Saving model: Autosave")
                
        # Vis losses and reliabilities
        plot_loss(
            train_loss_list=train_loss_list, 
            eval_loss_list=eval_loss_list, 
            train_rel_list=train_reliability_list, 
            val_rel_list=eval_reliability_list, 
            eval_score_list=eval_score_list,
            dst_dir=cfg.result_path)
        
        print(colored(f"Train Loss: {train_loss:.4f} - NLL: {train_nll_loss:.4f} - Cost: {train_cost_loss:.4f} - RLS: {train_reliability_scores['avg_RLS']:.2f} % - RLS_min: {train_reliability_scores['min_RLS']:.2f} % - LR: {current_lr:.8f}", 'green'))
        print(colored(f"Eval Loss: {eval_loss:.4f} - NLL: {eval_nll_loss:.4f} - Cost: {eval_cost_loss:.4f} - RLS: {eval_reliability_scores['avg_RLS']:.2f} % - RLS_min: {eval_reliability_scores['min_RLS']:.2f} % - Best Eval Score: {best_eval_score:.2f} %- @ Epoch: {best_epoch}", 'green'))
        cfg.logger.info(f"Train Loss: {train_loss:.4f} - NLL: {train_nll_loss:.4f} - Cost: {train_cost_loss:.4f} - RLS: {train_reliability_scores['avg_RLS']:.2f} % - RLS_min: {train_reliability_scores['min_RLS']:.2f} % - LR: {current_lr:.8f}")
        cfg.logger.info(f"Eval Loss: {eval_loss:.4f} - NLL: {eval_nll_loss:.4f} - Cost: {eval_cost_loss:.4f} - RLS: {eval_reliability_scores['avg_RLS']:.2f} % - RLS_min: {eval_reliability_scores['min_RLS']:.2f} % - Best Eval Score: {best_eval_score:.2f} %- @ Epoch: {best_epoch}")
        
    return


def train(args):
    
    # Setup gpu and load config
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = ConfigLoader(args=args, work_dir=os.getcwd(), regular=True)
    print(colored(f"Training for: {args.target} using config: {args.cfg} on gpu: {args.gpu}", 'green'))
    cfg.logger.info(f"Training for: {args.target} using config: {args.cfg} on gpu: {args.gpu}")
    
    # Build mesh grids
    target_mesh_grid, target_x_layer, target_y_layer = build_mesh_grid(
        mesh_range_x=cfg.grid_params['grid_size_meter'],
        mesh_range_y=cfg.grid_params['grid_size_meter'], 
        mesh_resolution=cfg.target_grid_resolution
    )
    
    # Create cord conv representation
    cord_convs = torch.stack([target_x_layer, target_y_layer], dim=0).unsqueeze(0)
    cord_convs_batched = cord_convs.expand(cfg.train_params['batch_size'], cord_convs.shape[1], cfg.grid_params['target_grid_size_cells'], cfg.grid_params['target_grid_size_cells'])
    target_mesh_grid = target_mesh_grid.to(device)
    
    # Dataset specific train data dataloader
    train_dataset = Context_DataLoader(
        cfg=cfg,
        path=cfg.paths['train_data_path'],
        typ='train',
        size=cfg.train_params['train_data_size']
    )
    
    # Dataset specific eval data dataloader
    eval_dataset = Context_DataLoader(
        cfg=cfg,
        path=cfg.paths['eval_data_path'], 
        typ='eval',
        size=cfg.train_params['eval_data_size']
    )
    
    print(colored(f"Loaded train data: {len(train_dataset)} samples"), 'green')
    print(colored(f"Loaded eval data: {len(eval_dataset)} samples"), 'green')
    cfg.logger.info(f"Loaded train data: {len(train_dataset)} samples")
    cfg.logger.info(f"Loaded eval data: {len(eval_dataset)} samples")
    
    # Dataset collate function
    collate_fn_with_params = partial(
        collate_fn, 
        cfg=cfg,
        cord_convs_batched=cord_convs_batched,
    )
    
    # Train dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_params['batch_size'],
        shuffle=(not cfg.train_params['auto_increment']),
        collate_fn=collate_fn_with_params,
        drop_last=True,
        pin_memory=True,
        num_workers=cfg.train_params['dataloader_num_workers'],
        prefetch_factor=cfg.train_params['dataloader_prefetch'],
        persistent_workers=(not cfg.train_params['auto_increment'])
    )
    
    # Eval dataloader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.train_params['batch_size'],
        shuffle=True,
        collate_fn=collate_fn_with_params,
        drop_last=True,
        pin_memory=True,
        num_workers=cfg.train_params['dataloader_num_workers'],
        prefetch_factor=cfg.train_params['dataloader_prefetch'],
        persistent_workers=True
    )
    
    # Model
    model = ContextMDN(cfg=cfg)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.train_params['start_learning_rate'], 
        weight_decay=cfg.train_params['weight_decay_val']
    )
    
    # Learnign rate scheduler
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=(cfg.train_params['final_learning_rate']/cfg.train_params['start_learning_rate']), total_iters=cfg.train_params['num_total_epochs'])
    
    # Resume a training
    if cfg.train_params["with_resume_train"]:
        
        checkpoint = torch.load(f=os.path.join(cfg.checkpoint_path, cfg.train_params["pre_train_weights"]))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    
    # Model to gpu
    model.to(device)
    train_size = len(train_loader)
    eval_size = len(eval_loader)
    
    # Start training
    training_process(
        cfg=cfg,
        train_dataset=train_dataset,
        train_loader=train_loader,
        eval_loader=eval_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler, 
        target_mesh_grid=target_mesh_grid,
        train_size=train_size, 
        eval_size=eval_size, 
        device=device
        )
    
    # Close logging
    cfg.log_file_handler.close()
    cfg.logger.removeHandler(cfg.log_file_handler)
    
    print(colored(f'End training, shutting down...', 'green'))
    cfg.logger.info(f'End training, shutting down...')
    
    return


if __name__ == "__main__":
    
    # Get args, set gpu id
    args = parse_args()
    
    # Mode
    if args.mode == "train":
        
        train(args=args)
        
    elif args.mode == "test":
        
        run_test(args=args)
        
    else:
        
        print(colored(f"Unknown mode: {args.mode}, must be 'train' or 'test'", 'red'))
    
    sys.exit(0)