import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class VariableHostPolicy(nn.Module):
    """Variable Host Policy that processes hosts individually using attention mechanism."""
    
    def __init__(self, obs_dim, action_dim, num_hosts=30, exploration_noise_decay=0.995, min_exploration_noise=0.01):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_hosts = num_hosts
        
        self.hidden_size = 16
        
        # Exploration noise scheduling
        self.exploration_noise_decay = exploration_noise_decay
        self.min_exploration_noise = min_exploration_noise
        self.current_exploration_noise = 1.0
        
        self.host_encoder = nn.Linear(4, self.hidden_size)
        
        self.job_encoder = nn.Linear(2, self.hidden_size)
        
        self.attention = nn.MultiheadAttention(self.hidden_size, num_heads=4, batch_first=True)
        
        self.priority_head = nn.Linear(self.hidden_size, 1)
        
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Action standard deviation parameter
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)
        
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
        num_hosts = (obs.shape[1] - 2) // 4
        
        # Parse observation: [host1_core_util, host1_mem_util, host1_cores_norm, host1_mem_norm, ..., job_core, job_mem]
        host_features = obs[:, :num_hosts * 4].view(batch_size, num_hosts, 4)  # [batch, num_hosts, 4]
        job_features = obs[:, -2:]  # [batch, 2]
        
        # Encode features with activation
        host_embeds = F.relu(self.host_encoder(host_features))  # [batch, num_hosts, hidden_size]
        job_embed = F.relu(self.job_encoder(job_features)).unsqueeze(1)  # [batch, 1, hidden_size]
        
        # Attention: job queries host capabilities
        attended_hosts, attention_weights = self.attention(job_embed, host_embeds, host_embeds)  # [batch, 1, hidden_size]
        
        # Generate priorities for each host using attention-weighted host features
        # Apply attention weights to original host embeddings
        weighted_host_embeds = attention_weights.squeeze(1).unsqueeze(-1) * host_embeds  # [batch, num_hosts, hidden_size]
        action_mean = torch.sigmoid(self.priority_head(weighted_host_embeds).squeeze(-1))  # [batch, num_hosts]
        
        # Action standard deviation should match actual number of hosts
        actual_num_hosts = action_mean.shape[1]
        if actual_num_hosts <= self.policy_log_std.shape[0]:
            action_std = torch.exp(self.policy_log_std[:actual_num_hosts].clamp(-5, 2))
        else:
            # Extend log_std if we have more hosts than originally configured
            extended_log_std = torch.cat([
                self.policy_log_std,
                torch.full((actual_num_hosts - self.policy_log_std.shape[0],), -1.0, device=self.policy_log_std.device)
            ])
            action_std = torch.exp(extended_log_std.clamp(-5, 2))
        
        action_std = action_std.unsqueeze(0).expand(batch_size, -1)
        
        # Value estimation using attended job-host representation
        value = self.value_head(attended_hosts.squeeze(1))  # [batch, 1]
        
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