import torch
import torch.nn as nn
import torch.nn.functional as F


from layers import TrajectoryEncoder, SocialEncoder, GridEncoder, MixtureDensityNetwork, initialize_weights


# Full Context MDN
class ContextMDN(nn.Module):
    
    def __init__(self, cfg):
        
        super(ContextMDN, self).__init__()
        
        # Internal params
        self.lstm_input_dim=cfg.model_params['lstm_input_dim']
        self.lstm_hidden_size=cfg.model_params['lstm_hidden_size'] 
        self.lstm_layer_size=cfg.model_params['lstm_layer_size'] 
        self.grid_feature_size=cfg.model_params['grid_feature_size'] 
        self.grid_input_dim=cfg.model_params['grid_input_dim'] 
        self.grid_head_size=cfg.model_params['grid_head_size']
        self.tf_model_dim=cfg.model_params['tf_model_dim'] 
        self.tf_head_size=cfg.model_params['tf_head_size'] 
        self.tf_layer_size=cfg.model_params['tf_layer_size'] 
        self.tf_forward_dim=cfg.model_params['tf_forward_dim']
        self.mdn_hidden_size=cfg.model_params['mdn_hidden_size']
        self.mdn_output_dim=cfg.model_params['mdn_output_dim']
        self.mdn_num_gaussians=cfg.model_params['mdn_num_gaussians']
        self.forecast_horizon=cfg.model_params['forecast_horizon']
        self.dropout_prob=cfg.model_params['dropout_prob']
        
        # Trajectory encoding (LSTM) with social attention (TF)
        self.target_encoder_lstm = TrajectoryEncoder(input_dim=self.lstm_input_dim, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_layer_size, batch_first=True, dropout_prob=self.dropout_prob)
        self.others_encoder_lstm = TrajectoryEncoder(input_dim=self.lstm_input_dim, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_layer_size, batch_first=True, dropout_prob=self.dropout_prob)
        self.social_encoder = SocialEncoder(model_dim=self.tf_model_dim, head_size=self.tf_head_size, num_layers=self.tf_layer_size, dropout_prob=self.dropout_prob, dim_feedforward=self.tf_forward_dim)
        
        # Grid encoder with multi head attention (CNN+MHA) 
        self.grid_encoder = GridEncoder(input_channels=self.grid_input_dim, output_size=self.grid_feature_size, nheads=self.grid_head_size, dropout_prob=self.dropout_prob)
        
        # MDN Network (MLP)
        self.mdn = MixtureDensityNetwork(input_dim=self.tf_model_dim + self.grid_feature_size, hidden_size=self.mdn_hidden_size, num_gaussians=self.mdn_num_gaussians, output_dim=self.mdn_output_dim, forecast_horizon=self.forecast_horizon, dropout_prob=self.dropout_prob)
        
        # Weight initialization
        self.apply(initialize_weights)
        self.target_encoder_lstm.apply(initialize_weights)
        self.others_encoder_lstm.apply(initialize_weights)
        self.social_encoder.apply(initialize_weights)
        self.grid_encoder.apply(initialize_weights)
        self.mdn.apply(initialize_weights)
        return
    
    
    def forward(self, target_traj, other_trajs, others_padding_mask, occupancy_grid):
        # past_traj: (batch_size, input_horizon, input_dim)
        # occupancy_grid: (batch_size, 3, grid_size_cells, grid_size_cells)
        
        # Get shapes
        _, A, _, _ = other_trajs.shape
        
        # Encode grid with attention
        grid_features = self.grid_encoder(occupancy_grid)  # (batch_size, grid_feature_size)
        
        # Encode trajectories
        target_embedding = self.target_encoder_lstm(x=target_traj)
        other_embeddings = torch.stack([self.others_encoder_lstm(other_trajs[:, i]) for i in range(A)], dim=1)  # [B, A, d_model]
        all_embeddings = torch.cat([target_embedding.unsqueeze(1), other_embeddings], dim=1)
        
        # Apply multi-head attention for social encoding and extract the socially-aware target embedding (first agent = our target agent)
        social_out = self.social_encoder(x=all_embeddings, m=others_padding_mask) # [B, A+1, d_model]
        social_target_embedding = social_out[:, 0, :]  # [B, d_model]
        
        # Apply Mixture density network
        params = self.mdn(x=torch.cat([social_target_embedding, grid_features], dim=1))  # (batch_size, forecast_horizon * num_gaussians * output_dim)
        
        
        return params
    

# Without Occupancy Grid
class ContextMDN_wo_Grid(nn.Module):
    
    def __init__(self, cfg):
        
        super(ContextMDN_wo_Grid, self).__init__()
        
        # Internal params
        self.lstm_input_dim=cfg.model_params['lstm_input_dim']
        self.lstm_hidden_size=cfg.model_params['lstm_hidden_size'] 
        self.lstm_layer_size=cfg.model_params['lstm_layer_size'] 
        self.tf_model_dim=cfg.model_params['tf_model_dim'] 
        self.tf_head_size=cfg.model_params['tf_head_size'] 
        self.tf_layer_size=cfg.model_params['tf_layer_size'] 
        self.tf_forward_dim=cfg.model_params['tf_forward_dim']
        self.mdn_hidden_size=cfg.model_params['mdn_hidden_size']
        self.mdn_output_dim=cfg.model_params['mdn_output_dim']
        self.mdn_num_gaussians=cfg.model_params['mdn_num_gaussians']
        self.forecast_horizon=cfg.model_params['forecast_horizon']
        self.dropout_prob=cfg.model_params['dropout_prob']
        self.mdn_dropout_prob=cfg.model_params['mdn_dropout_prob']
        
        # Trajectory encoding (LSTM) with social attention (TF)
        self.target_encoder_lstm = TrajectoryEncoder(input_dim=self.lstm_input_dim, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_layer_size, batch_first=True, dropout_prob=self.dropout_prob)
        self.others_encoder_lstm = TrajectoryEncoder(input_dim=self.lstm_input_dim, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_layer_size, batch_first=True, dropout_prob=self.dropout_prob)
        self.social_encoder = SocialEncoder(model_dim=self.tf_model_dim, head_size=self.tf_head_size, num_layers=self.tf_layer_size, dropout_prob=self.dropout_prob, dim_feedforward=self.tf_forward_dim)
        
        # MDN Network (MLP)
        self.mdn = MixtureDensityNetwork(input_dim=self.tf_model_dim, hidden_size=self.mdn_hidden_size, num_gaussians=self.mdn_num_gaussians, output_dim=self.mdn_output_dim, forecast_horizon=self.forecast_horizon, dropout_prob=self.mdn_dropout_prob)
        
        # Weight initialization
        self.apply(initialize_weights)
        self.target_encoder_lstm.apply(initialize_weights)
        self.others_encoder_lstm.apply(initialize_weights)
        self.social_encoder.apply(initialize_weights)
        self.mdn.apply(initialize_weights)
        return
    
    
    def forward(self, target_traj, other_trajs, others_padding_mask):
        # past_traj: (batch_size, input_horizon, input_dim)
        # occupancy_grid: (batch_size, 3, grid_size_cells, grid_size_cells)
        
        # Get shapes
        _, A, _, _ = other_trajs.shape
        
        # Encode trajectories
        target_embedding = self.target_encoder_lstm(x=target_traj)
        other_embeddings = torch.stack([self.others_encoder_lstm(other_trajs[:, i]) for i in range(A)], dim=1)  # [B, A, d_model]
        all_embeddings = torch.cat([target_embedding.unsqueeze(1), other_embeddings], dim=1)
        
        # Apply multi-head attention for social encoding and extract the socially-aware target embedding (first agent = our target agent)
        social_out = self.social_encoder(x=all_embeddings, m=others_padding_mask) # [B, A+1, d_model]
        social_target_embedding = social_out[:, 0, :]  # [B, d_model]
        
        # Apply Mixture density network
        params = self.mdn(x=social_target_embedding)  # (batch_size, forecast_horizon * num_gaussians * output_dim)
        
        return params
    
    
# Without Social Attention
class ContextMDN_wo_Social(nn.Module):
    
    def __init__(self, cfg):
        
        super(ContextMDN_wo_Social, self).__init__()
        
        # Internal params
        self.lstm_input_dim=cfg.model_params['lstm_input_dim']
        self.lstm_hidden_size=cfg.model_params['lstm_hidden_size'] 
        self.lstm_layer_size=cfg.model_params['lstm_layer_size'] 
        self.grid_feature_size=cfg.model_params['grid_feature_size'] 
        self.grid_input_dim=cfg.model_params['grid_input_dim'] 
        self.grid_head_size=cfg.model_params['grid_head_size']
        self.mdn_hidden_size=cfg.model_params['mdn_hidden_size']
        self.mdn_output_dim=cfg.model_params['mdn_output_dim']
        self.mdn_num_gaussians=cfg.model_params['mdn_num_gaussians']
        self.forecast_horizon=cfg.model_params['forecast_horizon']
        self.dropout_prob=cfg.model_params['dropout_prob']
        
        # Trajectory encoding (LSTM) with social attention (TF)
        self.target_encoder_lstm = TrajectoryEncoder(input_dim=self.lstm_input_dim, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_layer_size, batch_first=True, dropout_prob=self.dropout_prob)
        
        # Grid encoder with multi head attention (CNN+MHA) 
        self.grid_encoder = GridEncoder(input_channels=self.grid_input_dim, output_size=self.grid_feature_size, nheads=self.grid_head_size, dropout_prob=self.dropout_prob)
        
        # MDN Network (MLP)
        self.mdn = MixtureDensityNetwork(input_dim=self.lstm_hidden_size + self.grid_feature_size, hidden_size=self.mdn_hidden_size, num_gaussians=self.mdn_num_gaussians, output_dim=self.mdn_output_dim, forecast_horizon=self.forecast_horizon, dropout_prob=self.dropout_prob)
        
        # Weight initialization
        self.apply(initialize_weights)
        self.target_encoder_lstm.apply(initialize_weights)
        self.grid_encoder.apply(initialize_weights)
        self.mdn.apply(initialize_weights)
        return
    
    
    def forward(self, target_traj, occupancy_grid):
        # past_traj: (batch_size, input_horizon, input_dim)
        # occupancy_grid: (batch_size, 3, grid_size_cells, grid_size_cells)
        
        # Encode grid with attention
        grid_features = self.grid_encoder(occupancy_grid)  # (batch_size, grid_feature_size)
        
        # Encode trajectories
        target_embedding = self.target_encoder_lstm(x=target_traj)
        
        # Apply Mixture density network
        params = self.mdn(x=torch.cat([target_embedding, grid_features], dim=1))  # (batch_size, forecast_horizon * num_gaussians * output_dim)
        
        return params
    
    
# Full Context MDN
class ContextMDN_wo_Context(nn.Module):
    
    def __init__(self, cfg):
        
        super(ContextMDN_wo_Context, self).__init__()
        
        # Internal params
        self.lstm_input_dim=cfg.model_params['lstm_input_dim']
        self.lstm_hidden_size=cfg.model_params['lstm_hidden_size'] 
        self.lstm_layer_size=cfg.model_params['lstm_layer_size'] 
        self.mdn_hidden_size=cfg.model_params['mdn_hidden_size']
        self.mdn_output_dim=cfg.model_params['mdn_output_dim']
        self.mdn_num_gaussians=cfg.model_params['mdn_num_gaussians']
        self.forecast_horizon=cfg.model_params['forecast_horizon']
        self.dropout_prob=cfg.model_params['dropout_prob']
        
        # Trajectory encoding (LSTM) with social attention (TF)
        self.target_encoder_lstm = TrajectoryEncoder(input_dim=self.lstm_input_dim, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_layer_size, batch_first=True, dropout_prob=self.dropout_prob)
        
        # MDN Network (MLP)
        self.mdn = MixtureDensityNetwork(input_dim=self.lstm_hidden_size, hidden_size=self.mdn_hidden_size, num_gaussians=self.mdn_num_gaussians, output_dim=self.mdn_output_dim, forecast_horizon=self.forecast_horizon, dropout_prob=self.dropout_prob)
        
        # Weight initialization
        self.apply(initialize_weights)
        self.target_encoder_lstm.apply(initialize_weights)
        self.mdn.apply(initialize_weights)
        return
    
    
    def forward(self, target_traj):
        # past_traj: (batch_size, input_horizon, input_dim)
        # occupancy_grid: (batch_size, 3, grid_size_cells, grid_size_cells)
        
        # Encode trajectories
        target_embedding = self.target_encoder_lstm(x=target_traj)
        
        # Apply Mixture density network
        params = self.mdn(x=target_embedding)  # (batch_size, forecast_horizon * num_gaussians * output_dim)
        
        return params