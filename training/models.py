import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class ActorCriticPolicy(nn.Module):
    """Advanced structured Actor-Critic for host scheduling with attention mechanism."""

    def __init__(self, obs_dim, action_dim, num_hosts, exploration_noise_decay=0.995, min_exploration_noise=0.01):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_hosts = num_hosts
        
        # Exploration noise scheduling
        self.exploration_noise_decay = exploration_noise_decay
        self.min_exploration_noise = min_exploration_noise
        self.current_exploration_noise = 1.0

        # Host feature processing
        self.host_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )
        
        # Job feature processing
        self.job_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
        )
        
        # Job-to-host projection for attention
        self.job_to_host_proj = nn.Linear(16, 16)
        
        # Multi-head attention for host-job interaction
        self.attention = nn.MultiheadAttention(
            embed_dim=16, 
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Global context processing
        host_features_dim = 16
        job_features_dim = 16
        global_context_dim = host_features_dim + job_features_dim
        
        self.shared_net = nn.Sequential(
            nn.Linear(global_context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Policy head (actor)
        self.policy_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # Transform to per-host outputs via host-aware projection
        self.host_policy_proj = nn.Linear(64 + 16, 1)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)

        # Value head (critic)
        self.value_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(64, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        single_input = obs.dim() == 1
        if single_input:
            obs = obs.unsqueeze(0)
        
        batch_size = obs.shape[0]
        
        # Parse structured input: [host1_cores, host1_mem, host2_cores, host2_mem, ..., job_cores, job_mem]
        host_data = obs[:, :-2].reshape(batch_size, self.num_hosts, 2)  # [batch, num_hosts, 2]
        job_data = obs[:, -2:]  # [batch, 2]
        
        # Process host features individually
        host_features = self.host_encoder(host_data)  # [batch, num_hosts, 16]
        
        # Process job features
        job_features = self.job_encoder(job_data)  # [batch, 12]
        
        # Attention: let hosts attend to job requirements
        job_query = job_features.unsqueeze(1).expand(-1, self.num_hosts, -1)  # [batch, num_hosts, 12]
        # Project job features to match host embedding dimension for attention
        job_projected = self.job_to_host_proj(job_query)  # [batch, num_hosts, 16]
        
        # Multi-head attention: hosts as keys/values, job as query
        attended_hosts, _ = self.attention(
            query=job_projected,  # [batch, num_hosts, 16] 
            key=host_features,    # [batch, num_hosts, 16]
            value=host_features   # [batch, num_hosts, 16]
        )
        
        # Global context: aggregate attended host features + job features
        global_host_context = attended_hosts.mean(dim=1)  # [batch, 16]
        global_context = torch.cat([global_host_context, job_features], dim=-1)  # [batch, 32]
        
        # Shared processing
        shared_features = self.shared_net(global_context)  # [batch, 128]
        
        # Policy: generate per-host priorities
        policy_features = self.policy_layers(shared_features)  # [batch, 64]
        
        # Combine shared policy features with individual host features for per-host decisions
        policy_features_expanded = policy_features.unsqueeze(1).expand(-1, self.num_hosts, -1)  # [batch, num_hosts, 64]
        host_policy_input = torch.cat([policy_features_expanded, attended_hosts], dim=-1)  # [batch, num_hosts, 64+16]
        
        # Generate per-host priorities
        action_mean = torch.sigmoid(self.host_policy_proj(host_policy_input).squeeze(-1))  # [batch, num_hosts]
        action_std = torch.exp(self.policy_log_std.clamp(-5, 2)).unsqueeze(0).expand(batch_size, -1)
        
        # Value estimation
        value_features = self.value_layers(shared_features)
        value = self.value_head(value_features)
        
        if single_input:
            action_mean = action_mean.squeeze(0)
            action_std = action_std.squeeze(0)
            value = value.squeeze(0)
        return action_mean, action_std, value

    def get_action_and_value(self, obs, action=None, deterministic=False):
        action_mean, action_std, value = self.forward(obs)
        
        # Apply exploration noise decay
        if not deterministic and self.training:
            exploration_factor = max(self.current_exploration_noise, self.min_exploration_noise)
            action_std = action_std * exploration_factor
        
        dist = Normal(action_mean, action_std)
        if action is None:
            if deterministic:
                action = action_mean
            else:
                action = dist.sample()
        action = torch.clamp(action, 0, 1)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        if value.dim() > 0:
            value = value.squeeze(-1)
        return action, log_prob, entropy, value
    
    def decay_exploration_noise(self):
        """Decay exploration noise for better exploitation over time"""
        self.current_exploration_noise = max(
            self.current_exploration_noise * self.exploration_noise_decay,
            self.min_exploration_noise
        )

    def get_value(self, obs):
        _, _, value = self.forward(obs)
        if value.dim() > 0:
            return value.squeeze(-1)
        return value