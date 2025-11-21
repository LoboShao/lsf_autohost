import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


class SimpleMlpPolicy(nn.Module):
    """Simple MLP policy for continuous control tasks.

    A straightforward actor-critic architecture that can work with any environment.
    Useful as a baseline or fallback when specialized models aren't available.
    """

    def __init__(self, obs_dim, action_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Build actor network
        actor_layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            actor_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size
        actor_layers.append(nn.Linear(prev_size, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        # Build critic network
        critic_layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            critic_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size
        critic_layers.append(nn.Linear(prev_size, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Log standard deviation (learnable)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Initialize weights
        self._initialize_weights()

        # For compatibility with models that have exploration decay
        self.current_exploration_noise = 1.0

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Special initialization for output layers
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, obs):
        """Forward pass through the network.

        Args:
            obs: Observation tensor of shape (batch_size, obs_dim) or (obs_dim,)

        Returns:
            action_mean: Mean of action distribution (batch_size, action_dim)
            action_logstd: Log standard deviation (batch_size, action_dim)
            value: Value estimate (batch_size, 1)
        """
        # Handle both batched and single observations
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False

        # Actor output
        action_mean = self.actor(obs)

        # Apply sigmoid for bounded action space [0, 1]
        action_mean = torch.sigmoid(action_mean)

        # Expand log_std to match batch size
        action_logstd = self.log_std.expand_as(action_mean)

        # Critic output
        value = self.critic(obs)

        # Remove batch dimension if input was single observation
        if squeeze_output:
            action_mean = action_mean.squeeze(0)
            action_logstd = action_logstd.squeeze(0)
            value = value.squeeze(0)

        return action_mean, action_logstd, value

    def get_action_and_value(self, obs, action=None, deterministic=False):
        """Get action and value from the policy (PPOTrainer interface).

        Args:
            obs: Observation tensor
            action: Optional action to evaluate (if None, sample new action)
            deterministic: If True, return mean action (no exploration)

        Returns:
            action: Action tensor
            log_prob: Log probability of the action (scalar per batch)
            entropy: Entropy of the distribution (scalar)
            value: Value estimate (scalar per batch)
        """
        action_mean, action_logstd, value = self.forward(obs)

        # Fast path for deterministic inference (testing)
        if deterministic and action is None:
            action = torch.clamp(action_mean, 0, 1)
            # Create zero scalar/tensor for log_prob and entropy
            if action.dim() == 1:
                zero_tensor = torch.tensor(0.0, device=action.device, dtype=action.dtype)
            else:
                zero_tensor = torch.zeros(action.shape[0], device=action.device, dtype=action.dtype)
            return action, zero_tensor, zero_tensor, value.squeeze(-1) if value.dim() > 0 else value

        # Full path for training and stochastic inference
        action_std = action_logstd.exp()
        dist = Normal(action_mean, action_std)

        if action is None:
            # Sample new action
            action = dist.sample()
            action = torch.clamp(action, 0.0, 1.0)

        # Calculate log probability and entropy
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)  # Keep per-sample entropy, don't take mean

        # Ensure value is properly shaped (scalar per batch)
        value = value.squeeze(-1) if value.dim() > 1 else value

        return action, log_prob, entropy, value

    def get_value(self, obs):
        """Get value estimate for observations."""
        _, _, value = self.forward(obs)
        return value.squeeze(-1) if value.dim() > 1 else value

    def decay_exploration_noise(self):
        """Placeholder for compatibility with other models."""
        pass