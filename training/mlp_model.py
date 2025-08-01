import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class MLPPolicy(nn.Module):
    """Simple Multi-Layer Perceptron Actor-Critic for host scheduling."""

    def __init__(self, obs_dim, action_dim, num_hosts=10, exploration_noise_decay=0.995, min_exploration_noise=0.01):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_hosts = num_hosts
        
        # Exploration noise scheduling
        self.exploration_noise_decay = exploration_noise_decay
        self.min_exploration_noise = min_exploration_noise
        self.current_exploration_noise = 1.0

        # Shared feature extraction
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Policy head (actor)
        self.policy_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)

        # Value head (critic)
        self.value_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

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
        
        # Shared processing
        shared_features = self.shared_net(obs)
        
        # Policy: generate action probabilities
        action_mean = torch.sigmoid(self.policy_layers(shared_features))
        action_std = torch.exp(self.policy_log_std.clamp(-5, 2)).unsqueeze(0).expand(batch_size, -1)
        
        # Value estimation
        value = self.value_layers(shared_features)
        
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