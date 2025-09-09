import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


class VariableHostMLPPolicy(nn.Module):
    """Simple MLP-based Variable Host Policy without encoders or attention."""
    
    def __init__(self, obs_dim, action_dim, num_hosts=30, exploration_noise_decay=0.995, min_exploration_noise=0.01):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_hosts = num_hosts
        
        # Exploration noise scheduling
        self.exploration_noise_decay = exploration_noise_decay
        self.min_exploration_noise = min_exploration_noise
        self.current_exploration_noise = 1.0
        
        # Simple direct mapping: host features to priorities
        self.host_priority_net = nn.Sequential(
            nn.Linear(4, 8),   # Just host features → small hidden
            nn.ReLU(),
            nn.Linear(8, 1)    # → priority
        )
        
        # Job influence on priorities
        self.job_influence_net = nn.Sequential(
            nn.Linear(2, 4),   # Job features → influence vector
            nn.ReLU(),
            nn.Linear(4, 1)    # → global influence scalar
        )
        
        # Smaller value head
        self.value_head = nn.Sequential(
            nn.Linear(obs_dim, 64),
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
    
    def forward(self, obs, deterministic=False):
        single_input = obs.dim() == 1
        if single_input:
            obs = obs.unsqueeze(0)
        
        batch_size = obs.shape[0]
        num_hosts = (obs.shape[1] - 2) // 4
        
        # Parse observation: [host1_core_util, host1_mem_util, host1_cores_norm, host1_mem_norm, ..., job_core, job_mem]
        host_features = obs[:, :num_hosts * 4].view(batch_size, num_hosts, 4)  # [batch, num_hosts, 4]
        job_features = obs[:, -2:]  # [batch, 2]
        
        # Process host features to get base priorities
        host_flat = host_features.view(-1, 4)  # [batch * num_hosts, 4]
        base_priorities = self.host_priority_net(host_flat)  # [batch * num_hosts, 1]
        base_priorities = base_priorities.view(batch_size, num_hosts)  # [batch, num_hosts]
        
        # Get job influence (global modifier)
        job_influence = self.job_influence_net(job_features)  # [batch, 1]
        
        # Apply job influence to base priorities
        action_mean = torch.sigmoid(base_priorities + job_influence)  # [batch, num_hosts]
        
        # Value estimation using global features
        value = self.value_head(obs)  # [batch, 1]
        
        # Skip expensive action_std computation for deterministic inference
        if deterministic:
            if single_input:
                action_mean = action_mean.squeeze(0)
                value = value.squeeze(0)
            return action_mean, None, value
        
        # Action standard deviation computation (only for training/stochastic)
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
        
        if single_input:
            action_mean = action_mean.squeeze(0)
            action_std = action_std.squeeze(0)
            value = value.squeeze(0)
        
        return action_mean, action_std, value
    
    def get_action_and_value(self, obs, action=None, deterministic=False):
        # Fast path for deterministic inference (testing)  
        if deterministic and action is None:
            action_mean, _, value = self.forward(obs, deterministic=True)
            action = torch.clamp(action_mean, 0, 1)
            # Create zero scalar/tensor for log_prob and entropy (shape should match expected output)
            if action.dim() == 1:
                zero_tensor = torch.tensor(0.0, device=action.device, dtype=action.dtype)
            else:
                zero_tensor = torch.zeros(action.shape[0], device=action.device, dtype=action.dtype)
            return action, zero_tensor, zero_tensor, value.squeeze(-1) if value.dim() > 0 else value
        
        # Full path for training and stochastic inference
        action_mean, action_std, value = self.forward(obs, deterministic=False)
        
        if not deterministic and self.training:
            exploration_factor = max(self.current_exploration_noise, self.min_exploration_noise)
            action_std = action_std * exploration_factor
        
        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample() if not deterministic else action_mean
        
        action = torch.clamp(action, 0, 1)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value.squeeze(-1) if value.dim() > 0 else value
    
    def decay_exploration_noise(self):
        """Decay exploration noise for better exploitation over time"""
        self.current_exploration_noise = max(
            self.current_exploration_noise * self.exploration_noise_decay,
            self.min_exploration_noise
        )

    def get_value(self, obs):
        _, _, value = self.forward(obs, deterministic=True)
        if value.dim() > 0:
            return value.squeeze(-1)
        return value