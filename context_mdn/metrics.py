import torch


from termcolor import colored
from prob import build_forecast


def calc_ade_fde_mode(gt, gmm, batch_size, n_forecast_horizons, n_features):

    # Extract the categorical probabilities and component means and find the most probable mode for each distribution
    categorical_probs = gmm.mixture_distribution.probs.view(batch_size, n_forecast_horizons, n_features)
    component_means = gmm.component_distribution.mean.view(batch_size, n_forecast_horizons, n_features, 2)
    most_probable_mode_idx = categorical_probs.argmax(dim=-1)
    
    # Gather the means of the most probable mode
    modes = torch.gather(component_means, dim=2, index=most_probable_mode_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 2)).squeeze(2)
    error = torch.norm(modes - gt, dim=-1)
    ade_mode = error.mean()
    fde_mode = error[:,-1].mean()
    
    return ade_mode, fde_mode


def calc_ade_fde_k(gt, gmm, k, batch_size, n_forecast_horizons, n_features):
    """calculate best of k min ade/fde scores from distribution samples
    Args:
        gt (torch.tensor): batch of ground truth data [batch_size, n_horizons, n_features]
        gmm (torch.distribution.MixtureSameFamily): batch of gaussian mixture models [batch_size, n_horizons]
        k (int): number k of samples to get from distributions
    Returns:
        float: min_ade, min_fde scores
    """
    
    # get samples from distributions
    k_predictions = get_samples(gmm=gmm, n_sample_points=k)
    
    # reshape from [batch_size * n_horizons, k, n_features] to [batch_size, n_horizons, k, n_features]
    k_predictions = k_predictions.permute(1,0,2)
    k_predictions = k_predictions.view(batch_size, n_forecast_horizons, k, n_features)
    
    # predictions: [batch_size, n_horizons, k, n_features]
    min_ade = compute_ade(predictions=k_predictions, gt=gt)
    min_fde = compute_fde(predictions=k_predictions, gt=gt)
    
    return min_ade, min_fde


def calc_ade_fde_k_with_probs(gt, gmm, k, batch_size, n_forecast_horizons, n_features):
    """calculate best of k min ade/fde scores from distribution samples
    Args:
        gt (torch.tensor): batch of ground truth data [batch_size, n_horizons, n_features]
        gmm (torch.distribution.MixtureSameFamily): batch of gaussian mixture models [batch_size, n_horizons]
        k (int): number k of samples to get from distributions
    Returns:
        float: min_ade, min_fde scores
    """
    # Get prediction samples with probs
    k_predictions, k_probs = get_samples_with_probs(gmm=gmm, n_sample_points=k)
    
    # Re-arrange
    k_predictions = k_predictions.permute(1,0,2)
    k_predictions = k_predictions.view(batch_size, n_forecast_horizons, k, n_features)
    k_probs = k_probs.permute(1,0)
    k_probs = k_probs.view(batch_size, n_forecast_horizons, k)
    
    # Sort k samples by probabilities
    k_sorted_probs, k_sorted_indices = torch.topk(k_probs, k=k, dim=-1)
    k_sorted_predictions = k_predictions.gather(dim=2, index=k_sorted_indices.unsqueeze(-1).expand(-1, -1, -1, 2))
    
    # Get euclidean errors for k samples over defined future time steps of batch
    k_sorted_errors = torch.linalg.norm(k_sorted_predictions - gt.unsqueeze(2), dim=-1)
    
    # Compute ADE by averaging over the timesteps
    k_sorted_ade = torch.mean(k_sorted_errors, dim=1)  # [batch_size, k]
    k_sorted_fde = k_sorted_errors[:,-1,:]
    
    # Compute the average probability for each sorted sample
    k_sorted_ade_probs = torch.mean(k_sorted_probs, dim=1)  # [batch_size, k]
    k_sorted_fde_probs = k_sorted_probs[:,-1,:] # [batch_size, k]
    
    return k_sorted_ade[:,0], k_sorted_ade_probs[:,0], k_sorted_fde[:,0], k_sorted_fde_probs[:,0]


def calc_ade_fde_k_for_cls(gt, gmm, k, n_sample_points, batch_size, n_forecast_horizons, n_features, confidence_level):
    """calculate best of k min ade/fde scores from distribution samples at defined confidence levels
    Args:
        gt (torch.tensor): batch of ground truth data [batch_size, n_horizons, n_features]
        gmm (torch.distribution.MixtureSameFamily): batch of gaussian mixture models [batch_size, n_horizons]
        k (int): number k of samples to get from distributions
        n_sample_points (int): number of samples to get from each distribution
        n_forecast_horizons (int): number of discrete forecast horizons
        n_features (int): number of features
        confidence_level (list): list object containing a number of user defined confidence levels
    Returns:
        dict: dictionary containing min_ade, min_fde scores at each defined confidence level
    """
    
    # get k samples from within defined confidence level of distributions
    k_predictions = get_samples_from_confidence_level(
        gmm=gmm, 
        level=confidence_level, 
        k=k, 
        n_sample_points=n_sample_points, 
        batch_size=batch_size, 
        n_forecast_horizons=n_forecast_horizons, 
        n_features=n_features
    )
    
    # reshape from [batch_size * n_horizons, k, n_features] to [batch_size, n_horizons, k, n_features]
    k_predictions = k_predictions.view(batch_size, n_forecast_horizons, k, -1)
    
    # predictions: [batch_size, n_horizons, k, n_features]
    # gt: [batch_size, n_horizons, n_features]
    min_ade = compute_ade(predictions=k_predictions, gt=gt)
    min_fde = compute_fde(predictions=k_predictions, gt=gt)
    
    return min_ade, min_fde


# def calc_ade_fde_k_voronoi(cfg, gt, conf_levels):
#     """calculate best of k min ade/fde scores from distribution samples at defined confidence levels
#     Args:
#         gt (torch.tensor): batch of ground truth data [batch_size, n_horizons, n_features]
#         gmm (torch.distribution.MixtureSameFamily): batch of gaussian mixture models [batch_size, n_horizons]
#         k (int): number k of samples to get from distributions
#         n_sample_points (int): number of samples to get from each distribution
#         n_forecast_horizons (int): number of discrete forecast horizons
#         n_features (int): number of features
#         confidence_level (list): list object containing a number of user defined confidence levels
#     Returns:
#         dict: dictionary containing min_ade, min_fde scores at each defined confidence level
#     """
    
#     # get k samples from within defined confidence level of distributions
#     k_predictions = get_samples_from_voronoi(
#         confidence_areas=conf_levels, 
#         k=cfg.test_params['k_predictions'], 
#         iters=cfg.test_params['voronoi_iters'], 
#         grid_size=(cfg.test_params['voronoi_grid_size'],cfg.test_params['voronoi_grid_size']),
#         device=gt.device
#         )
    
#     # reshape from [batch_size * n_horizons, k, n_features] to [batch_size, n_horizons, k, n_features]
#     k_predictions = k_predictions.view(cfg.test_params['batch_size'], cfg.model_params['forecast_horizon'], cfg.test_params['k_predictions'], -1)
    
#     # predictions: [batch_size, n_horizons, k, n_features]
#     # gt: [batch_size, n_horizons, n_features]
#     min_ade = compute_ade(predictions=k_predictions, gt=gt)
#     min_fde = compute_fde(predictions=k_predictions, gt=gt)
    
#     return min_ade, min_fde


def compute_ade(predictions, gt):
    """ batched min ade calculation
    Args:
        predictions (torch.tensor): k predicted samples [batch_size, n_horizons, k, n_features]
        gt (torch.tensor): batch of ground truth data [batch_size, n_horizons, n_features]
    Returns:
        float: min ade result
    """
    
    error = torch.linalg.norm(predictions - gt.unsqueeze(2), axis=-1)
    ade = torch.mean(error, axis=1)
    min_ade = torch.min(ade, dim=1).values.mean()
    return min_ade


def compute_fde(predictions, gt):
    """ batched min fde calculation
    Args:
        predictions (torch.tensor): k predicted samples [batch_size, n_horizons, k, n_features]
        gt (torch.tensor): batch of ground truth data [batch_size, n_horizons, n_features]
    Returns:
        float: min fde result
    """
    
    fde = torch.linalg.norm(predictions - gt.unsqueeze(2), axis=-1)[:,-1,:]
    min_fde = torch.min(fde, dim=1).values.mean()
    return min_fde


def get_samples(gmm, n_sample_points):
    """get n samples from batched distributions
    Args:
        gmm (torch.distribution.MixtureSameFamily): batch of gaussian mixture models [batch_size, n_horizons]
        n_sample_points (int): number of samples to get from each distribution
    Returns:
        torch.tensor: batched samples
    """
    
    samples = gmm.sample(sample_shape=torch.Size([n_sample_points]))
    return samples


def get_samples_with_probs(gmm, n_sample_points):
    
    # get samples and compute log probabilities
    tau = 1
    samples = gmm.sample(sample_shape=torch.Size([n_sample_points]))
    samples_log_prob = gmm.log_prob(samples)
    probs = torch.exp(samples_log_prob/tau) / torch.sum(torch.exp(samples_log_prob/tau), dim=0)
    return samples, probs


def get_samples_from_confidence_level(gmm, level, k, n_sample_points, batch_size, n_forecast_horizons, n_features):
    """get k samples from batched distributions at defined confidence level
    Args:
        gmm (torch.distribution.MixtureSameFamily): batch of gaussian mixture models [batch_size, n_horizons]
        level (float): confidence level
        k (int): number k of samples to get from distributions
        n_sample_points (int): number of samples to get from each distribution
        n_tracks (int): number of trackes / batch size
        n_forecast_horizons (int): number of discrete forecast horizons
        n_features (int): number of features
    Returns:
        torch.tensor: batched k random samples of defined confidence level
    """
    
    # compute log probabilities for each sample
    samples, probs = get_samples_with_probs(gmm=gmm, n_sample_points=n_sample_points)
    
    # filter samples by confidence/density level using probabilities
    thres = int(level * n_sample_points)
    _, indices = probs.topk(k=thres, dim=0, largest=True, sorted=True)
    filtered_samples = torch.gather(samples, 0, torch.stack((indices, indices), dim=2))
    
    # randomly extract k samples for each track and timestep
    random_k_samples = torch.stack([filtered_samples[torch.randperm(thres)[:k], i] for i in range(batch_size*n_forecast_horizons)])
    
    # random_k_samples: [batch_size, n_horizons, k, n_features]
    return random_k_samples


def get_final_results(scores, k, cls):
    
    res = torch.mean(torch.stack(scores), axis=0).cpu().numpy()
    
    res_dict = {
        f"ADE mode": res[0], 
        f"FDE mode": res[1],
        f"min_ADE @ k={k}": res[2], 
        f"min_FDE @ k={k}": res[3]
    }
    
    for i, cl in enumerate(cls):
        
        res_dict[f"min_ADE @ k={k} @ {cl*100} % Cl"] = res[i*2+4]
        res_dict[f"min_FDE @ k={k} @ {cl*100} % Cl"] = res[i*2+5]
    
    return res_dict