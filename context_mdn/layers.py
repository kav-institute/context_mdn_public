import torch
import torch.nn as nn
import torch.nn.functional as F


class GridMultiheadAttention(nn.Module):
    
    def __init__(self, embed_dim, num_heads):
        
        super(GridMultiheadAttention, self).__init__()
        
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # A learnable query vector for attention pooling
        self.global_query = nn.Parameter(torch.randn(embed_dim))
        self.layer_norm = nn.LayerNorm(embed_dim)
        return
        
        
    def forward(self, features):
        """
        features: [B, N, E]  (N = H'*W', E = embed_dim)
        
        Returns:
            context:    [B, E] - the pooled context vector
            attn_weights: [B, N, N] from the multi-head attention
            pool_weights: [B, N] attention pooling weights
        """
        # Multi-head self-attention: query, key, and value are the same
        attended, attn_weights = self.mha(
            query=features,
            key=features,
            value=features,
            key_padding_mask=None
        )
        
        # Residual connection and layer normalization
        out = attended + features
        out = self.layer_norm(out)
        
        B, N, E = out.shape
        
        # Expand global_query for each batch instance: [B, E]
        query = self.global_query.unsqueeze(0).expand(B, E)
        
        # Compute dot product between global query and each spatial location in out: [B, N]
        scores = (out * query.unsqueeze(1)).sum(dim=-1)
        pool_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum to compute context vector: [B, E]
        context = torch.bmm(pool_weights.unsqueeze(1), out).squeeze(1)
        
        return context, attn_weights, pool_weights
    
    
class SocialEncoder(nn.Module):
    """
    Applies a transformer encoder layer (multi-head attention) to a set of agent embeddings.
    The first embedding is assumed to be the target agent, and the rest are neighbors.
    """
    def __init__(self, model_dim, head_size, num_layers, dropout_prob, dim_feedforward):
        
        super(SocialEncoder, self).__init__()
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=head_size, dim_feedforward=dim_feedforward, dropout=dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        return
        
        
    def forward(self, x, m):
        """
        agent_embeddings: [B, N_agents, d_model]
                        N_agents = 1 (target) + N_neighbors
        """
        
        # Transformer expects [S, B, E]: S=sequence length, B=batch size, E=embedding dim
        x = x.transpose(0, 1)  # [N_agents, B, d_model]
        
        # src_key_padding_mask: [B, N_agents], True=ignore, matches PyTorch convention directly
        out = self.transformer_encoder(x, src_key_padding_mask=m)      # [N_agents, B, d_model]
        out = out.transpose(0, 1)                             # [B, N_agents, d_model]
        return out
    
    
class TrajectoryEncoder(nn.Module):
    """
    Encodes a sequence of (x,y, vx, vy, ax, ay) coordinates into a fixed-dimensional embedding using an LSTM.
    """
    def __init__(self, input_dim, hidden_size, num_layers, batch_first, dropout_prob):
        
        super(TrajectoryEncoder, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.dropout = nn.Dropout(p=dropout_prob)
        return
    
    
    def forward(self, x):
        """
        traj: Tensor of shape [B, T, input_dim] 
            B = batch size
            T = number of timesteps in the trajectory
        """
        # LSTM returns (output, (h, c))
        # output: [B, T, hidden_size]
        lstm_out, _ = self.lstm(x)
        output = self.dropout(lstm_out[:,-1,:])
        
        return output


class GridEncoder(nn.Module):
    
    def __init__(self, input_channels, output_size, nheads, dropout_prob):
        
        super(GridEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            
            # First Convolutional Block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64
            
            # Second Convolutional Block
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            
            # Third Convolutional Block
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            
            # Fourth Convolutional Block
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
        )
        
        # Convert spatial feature map to a sequence for multi-head attention: [B, N, E]
        self.dropout = nn.Dropout(p=dropout_prob)
        self.flatten = nn.Flatten(2)  # flattens H x W into N, keeping [B, E, N]
        self.transpose = lambda x: x.transpose(1, 2)  # convert to [B, N, E]
        self.grid_attn = GridMultiheadAttention(embed_dim=256, num_heads=nheads)
        
        return
        
        
    def forward(self, x):
        
        features = self.encoder(x)
        features = self.flatten(features)   # [B, embed_dim, N]
        features = self.transpose(features)  # [B, N, embed_dim]
        
        context, attn_weights, pool_weights = self.grid_attn(features)  # context: [B, embed_dim]
        output = self.dropout(context)
        return output


class MixtureDensityNetwork(nn.Module):
    
    def __init__(self, input_dim, hidden_size, num_gaussians, output_dim, forecast_horizon, dropout_prob):
        
        super(MixtureDensityNetwork, self).__init__()
        
        # Variables
        self.num_gaussians = num_gaussians
        self.output_dim = output_dim
        self.forecast_horizon = forecast_horizon
        
        # Operations
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        
        # # Layers
        # self.fc1 = nn.Linear(input_dim, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Fully connected layer
        self.fc = nn.Linear(in_features=hidden_size, out_features=num_gaussians * output_dim * forecast_horizon)
        
        return
        
        
    def forward(self, x):
        
        # # x: (batch_size, input_dim)
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        
        # params: (batch_size, num_gaussians * output_dim * forecast_horizon)
        params = self.fc(x)
        
        # Reshape
        params = params.view(-1, self.forecast_horizon, self.num_gaussians * self.output_dim)  # (batch_size, forecast_horizon * num_gaussians * output_dim)
        return params


def initialize_weights(m):
    
    if isinstance(m, nn.Linear):
        
        nn.init.xavier_uniform_(m.weight)
        
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
            
    elif isinstance(m, nn.LSTM):
        
        for name, param in m.named_parameters():
            
            if "weight" in name:
                nn.init.xavier_uniform_(param)
                
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
    return