import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical
from skimage import measure

GRID_COST_CL_LEVEL = 0.95


def compute_loss(cfg, mixture_dist, target, mesh_grid, cost_grid):
    # mixture_dist: (batch_size * forecast_horizon)
    # target: (batch_size, forecast_horizon, 2)
    
    S, _ = mixture_dist.mixture_distribution.logits.shape
    B = cfg.train_params['batch_size']
    T = cfg.model_params['forecast_horizon']
    C = cfg.grid_params['target_grid_size_cells']
    
    # 1: NLL against ground truth
    # Goal: Encourage the model to assign high probability to the ground truth, minimize this term
    target_log_prob = mixture_dist.log_prob(target.view(S,2))
    nll_score = -torch.mean(target_log_prob)
    
    # 2: Occupancy grid failure cost
    if cfg.model_params["with_grid"]:
        
        # Occupancy grid failure cost
        grid_log_probs = mixture_dist.log_prob(mesh_grid).permute(1,0)  
        grid_probs = torch.exp(grid_log_probs)
        
        # Sort probabilities in descending order, Compute cumulative sums along the spatial dimension and Compute the total sum per row and find the 95% threshold
        sorted_probs, sorted_indices = torch.sort(grid_probs, dim=1, descending=True)
        cumulative = sorted_probs.cumsum(dim=1)
        threshold = cumulative[:, -1] * GRID_COST_CL_LEVEL
        
        # Create a boolean mask identifying elements <= 95% threshold, since cumulative is sorted in descending order, cumulative <= threshold
        # Will be True for all elements contributing to the densest 95% mass
        mask_sorted = cumulative <= threshold.unsqueeze(1)
        
        # We have the mask in the sorted order. We need to map it back to original order
        mask = torch.zeros_like(mask_sorted, dtype=torch.bool)
        mask.scatter_(dim=1, index=sorted_indices, src=mask_sorted)
        
        # Mask now is a binary mask in the original ordering, marking cells in the top 95% mass as True (1), others as False (0).
        mask = mask.view(B, T, C, C) 
        
        # Get occupancy grid errors
        grid = cost_grid[:,0].unsqueeze(1)
        grid_score = (mask * grid).sum() / (B * T)
        
        # Combine and weight the losses
        loss = nll_score + cfg.train_params['cost_weight'] * grid_score
        return loss, nll_score.item(), cfg.train_params['cost_weight'] * grid_score.item()
    
    else:
        
        # Combine and weight the losses
        loss = nll_score
        return loss, nll_score.item(), 0.0


def build_mixture_distribution(cfg, output, num_gaussians):
    
    # Model output check
    if torch.isnan(output).any():
        
        return None, False
    
    # Get shapes
    batch_size, forecast_horizon, _ = output.shape
    
    # Split the output into the parameters for each Gaussian
    mu_x = output[..., :num_gaussians]
    mu_y = output[..., num_gaussians:2*num_gaussians]
    sigma_x = F.softplus(output[..., 2*num_gaussians:3*num_gaussians]) + cfg.train_params["build_mixture_variance_epsilon"]
    sigma_y = F.softplus(output[..., 3*num_gaussians:4*num_gaussians]) + cfg.train_params["build_mixture_variance_epsilon"]
    rho = torch.tanh(output[..., 4*num_gaussians:5*num_gaussians])
    alpha = torch.softmax(output[..., 5*num_gaussians:], dim=-1)
    
    # Build covariance
    covs = torch.zeros(batch_size, forecast_horizon, num_gaussians, 2, 2).to(output.device)
    covs[..., 0, 0] = sigma_x ** 2
    covs[..., 0, 1] = rho * sigma_x * sigma_y
    covs[..., 1, 0] = rho * sigma_x * sigma_y
    covs[..., 1, 1] = sigma_y ** 2
    
    # Build means and weights
    mu = torch.stack([mu_x, mu_y], dim=-1).view(batch_size*forecast_horizon, num_gaussians, 2)
    covs = covs.view(batch_size*forecast_horizon, num_gaussians, 2, 2)
    weights = Categorical(alpha.view(batch_size*forecast_horizon, num_gaussians))
    
    # Create mixture object
    try:
        
        gaussians = MultivariateNormal(mu, covs)
        
    # No sucess, to due to divergence error
    except:
        
        return None, False
        
    # Sucess
    mixture = MixtureSameFamily(weights, gaussians)
    return mixture, True


def build_confidence_set(gmm, target, n_samples):
        
        gt_log_prob = gmm.log_prob(target)
        samples = gmm.sample(sample_shape=torch.Size([n_samples]))
        samples_log_prob = gmm.log_prob(samples)
        idx_mask = (samples_log_prob.unsqueeze(1) > gt_log_prob).float()
        conf = torch.sum(idx_mask, 0)/samples.shape[0]
        samples_mean = samples.mean(dim=0)
        return conf, samples_mean
    

def build_forecast(cfg, output, grid):
    
    # Create mixture
    gmm, _ = build_mixture_distribution(
        cfg=cfg,
        output=output.unsqueeze(0), 
        num_gaussians=cfg.model_params['mdn_num_gaussians']
        )
    
    # Build confidence sets
    conf_map, samples_mean = build_confidence_set(
        gmm=gmm, 
        target=grid, 
        n_samples=cfg.train_params['num_samples']
        )
    
    # Get modes
    modes = grid[torch.argmin(conf_map, dim=0, keepdim=True)]
    modes = modes[0,:,0,:]
    
    # Fix undefined modes due to empty conf map
    mask = (modes == torch.tensor([-10.0, -10.0]).to(modes.device)).all(dim=1)
    modes[mask] = samples_mean[mask]
    
    # Build confidence levels
    conf_levels = build_confidence_levels(
        cfg=cfg,
        conf_map=conf_map, 
        modes=modes 
        )
    
    return conf_levels, modes


def build_confidence_levels(cfg, conf_map, modes):
    
    conf_areas = []
    conf_contours_ego = []
    confidence_levels = cfg.train_params['confidence_levels']
    test_horizons = len(cfg.train_params['output_horizons'])
    
    for k in confidence_levels:
                    
        conf_areas.append(torch.reshape(torch.where(conf_map <= k, 1, 0), (cfg.grid_params['target_grid_size_cells'], cfg.grid_params['target_grid_size_cells'], test_horizons)).cpu().numpy()) 
        
    for idx, _ in enumerate(confidence_levels):
        
        contours_ego = []
        
        for h in range(0, test_horizons):
            
            # Get contour(s) of confidence level and time step
            c = measure.find_contours(conf_areas[idx][:, :, h], 0.5)
            
            # Uni modal dist
            if len(c) == 1:
                
                cont_ego = [np.flip(m=np.squeeze(np.array(c, dtype=np.float32) * cfg.target_grid_resolution - cfg.grid_params['grid_size_meter']))]
                contours_ego.append([np.squeeze(a=cont_ego, axis=0)])
                
            # Multi modal dist
            elif len(c) >= 2:
                
                cont_ego = [np.array(c[i], dtype=np.float32) * cfg.target_grid_resolution - cfg.grid_params['grid_size_meter'] for i in range(0, len(c))]
                cont_ego = [np.flip(m=ct) for ct in cont_ego]
                contours_ego.append(cont_ego)
                
            # Dist area smaller or equal to single point or grid size resolution, i.e mode of this dist
            else:
                
                cont_ego = [np.array(modes.cpu().numpy()[h], dtype=np.float32)[None, ...]]
                contours_ego.append([np.squeeze(a=cont_ego, axis=0)])
            
        conf_contours_ego.append(contours_ego)
        
    return conf_contours_ego


def compute_reliability(confidence_sets, batch_size, forecast_horizon, epoch=0, with_plot=False, cfg=None, dst_dir=None):
    
    res_dict = {}
    bins = cfg.reliability_bins
    batches, _ = confidence_sets.shape
    confidence_sets = confidence_sets.reshape(batches * batch_size, forecast_horizon)
    
    if with_plot:
        
        plt.figure()
        plt.plot(bins, bins, 'k--', linewidth=3, label=f"ideal")
        
    # place/sort values into bins
    # attention!: digitize() returns indexes, with first index starting at 1 not 0
    bin_data = np.digitize(confidence_sets, bins=bins)
    reliability_errors = []
    
    for idx in range(0, forecast_horizon):
        
        # build calibration curve
        # attention!: bincount() returns amount of each bin, first bin to count is bin at 0,
        # due to digitize behavior must increment len(bins) by 1 and later ignore the zero bin count
        f0 = np.array(np.bincount(bin_data[:,idx], minlength=len(bins)+1)).T
        
        # f0[1:]: because of the different start values of digitize and bincount, we remove/ignore the first value of f0
        acc_f0 = np.cumsum(f0[1:],axis=0)/confidence_sets.shape[0]
        
        # get differences for current step
        r = abs(acc_f0 - bins)
        reliability_errors.append(r)
        
        if with_plot: 
            color = cfg.colors_rgb[idx]
            plt.plot(bins, acc_f0, color=color, linewidth=3, label=f"{round((cfg.train_params['output_horizons'][idx]+1)*cfg.train_params['delta_t'], 1)} sec @ avg: {(1 - np.mean(r))*100:.1f} %, min: {(1 - np.max(r))*100:.1f} %")
        
    # get reliability scores
    reliability_avg_score = (1 - np.mean(reliability_errors)) * 100
    reliability_min_score = (1- np.max(reliability_errors)) * 100
    res_dict[f"avg_RLS"] = reliability_avg_score
    res_dict[f"min_RLS"] = reliability_min_score
    
    if with_plot:
        
        plt.grid()
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.title(f'Reliability Calibration: - Avg: {reliability_avg_score:.1f} % - Min: {reliability_min_score:.1f} %')
        plt.savefig(os.path.join(dst_dir, f'reliability_plain_{str(epoch).zfill(3)}.png'))
        plt.legend(fontsize = 10)
        plt.savefig(os.path.join(dst_dir, f'reliability_legend_{str(epoch).zfill(3)}.png'))
        plt.close()
    
    return res_dict


def compute_sharpness_sets(cfg, gmm, target, n_samples):
    
    # Build confidence maps
    conf_maps, _ = build_confidence_set(
        gmm=gmm, 
        target=target, 
        n_samples=n_samples
    )
    
    conf_maps = conf_maps.permute(1,0)
    conf_maps = conf_maps.view(cfg.test_params['batch_size'], cfg.model_params['forecast_horizon'] ,-1)
    
    sharpness_scores = []
    
    for kappa in cfg.train_params['confidence_levels']:
        
        sharpness_scores.append(estimate_sharpness(conf_map=conf_maps, kappa=kappa)*(cfg.grid_params['grid_size_meter']*cfg.grid_params['grid_size_meter']))
    
    return torch.stack(sharpness_scores, 1)


def compute_sharpness(cfg, sharpness_sets):
    
    sharpness_scores = {}
    percentiles = cfg.reliability_bins
    
    for idx, cl in enumerate(cfg.train_params['confidence_levels']):
        
        s = sharpness_sets[:,idx,:].T
        SDist=np.zeros((cfg.model_params['forecast_horizon'], len(percentiles)))
        
        for k, p in enumerate(percentiles):
            
            for t in range(cfg.model_params['forecast_horizon']):
                
                SDist[t,k]=np.percentile(s[t,:], p*100, axis=-1)
        
        # calc mean sharpness score (i.e 50%) for current confidence level
        sharpness_score = sum([np.mean(SDist[idx,:] / ((step+1)*cfg.train_params['delta_t'])) for idx, step in enumerate(range(cfg.model_params['forecast_horizon']))]) * (1/(cfg.model_params['forecast_horizon']*cfg.train_params['delta_t']))
        sharpness_scores[f"SS @ {cl}"] = sharpness_score
    
    return sharpness_scores


def estimate_sharpness(conf_map, kappa):
    """get sharpness area
    Args:
        confidence_map (torch.tensor): confidence map for single track [n_grid_points, n_horizons]
        kappa (float): confidence level
    Returns:
        torch.tensor: area [n_horizon]
    """
    
    area = torch.where(conf_map <= kappa, 1.0, 0.0)
    area = area.mean(dim=2)
    return area