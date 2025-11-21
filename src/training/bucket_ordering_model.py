import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


class BucketOrderingPolicy(nn.Module):
    """Policy for ordering job buckets based on their resource requirements.

    This model is specifically designed for the job_ordering environment where:
    - State: [bucket_cores, bucket_count] × max_buckets + [avail_cores_ratio, avail_mem_ratio]
    - Action: Priority values [0, 1] for each bucket
    - Strategy: RL decides bucket priorities, heuristic (first-available) selects hosts
    """

    def __init__(self, obs_dim, action_dim, max_buckets=100,
                 exploration_noise_decay=0.995, min_exploration_noise=0.01):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_buckets = max_buckets

        # Exploration noise scheduling
        self.exploration_noise_decay = exploration_noise_decay
        self.min_exploration_noise = min_exploration_noise
        self.current_exploration_noise = 1.0

        # Hidden sizes
        self.bucket_hidden = 64
        self.global_hidden = 32
        self.combined_hidden = 128

        # ========== Bucket Processing ==========
        # Each bucket has 2 features: normalized cores and job count
        self.bucket_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, self.bucket_hidden),
            nn.LayerNorm(self.bucket_hidden),
            nn.GELU()
        )

        # Attention mechanism to aggregate bucket information
        self.bucket_attention = nn.MultiheadAttention(
            self.bucket_hidden,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )

        # ========== Global State Processing ==========
        # Global features: available cores ratio, available memory ratio
        self.global_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, self.global_hidden),
            nn.LayerNorm(self.global_hidden),
            nn.GELU()
        )

        # ========== Combined Processing ==========
        # Combine bucket and global information
        combined_input_size = self.bucket_hidden + self.global_hidden

        self.combiner = nn.Sequential(
            nn.Linear(combined_input_size, self.combined_hidden),
            nn.LayerNorm(self.combined_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.combined_hidden, self.combined_hidden),
            nn.LayerNorm(self.combined_hidden),
            nn.GELU()
        )

        # ========== Actor (Policy) Head ==========
        # Outputs priority for each bucket
        self.actor_head = nn.Sequential(
            nn.Linear(self.combined_hidden, self.combined_hidden),
            nn.GELU(),
            nn.Linear(self.combined_hidden, action_dim)
        )

        # Log standard deviation for exploration (learnable)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # ========== Critic (Value) Head ==========
        self.critic_head = nn.Sequential(
            nn.Linear(self.combined_hidden, self.combined_hidden),
            nn.GELU(),
            nn.Linear(self.combined_hidden, 1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Special initialization for output layers
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)

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

        batch_size = obs.shape[0]

        # Split observation into bucket features and global features
        # obs format: [bucket_features..., global_features]
        bucket_features_flat = obs[:, :-2]  # All except last 2 features
        global_features = obs[:, -2:]  # Last 2 features

        # Reshape bucket features: (batch, max_buckets * 2) -> (batch, max_buckets, 2)
        bucket_features = bucket_features_flat.reshape(batch_size, self.max_buckets, 2)

        # ========== Process Buckets ==========
        # Encode each bucket
        bucket_encoded = self.bucket_encoder(bucket_features)  # (batch, max_buckets, bucket_hidden)

        # Apply self-attention across buckets
        bucket_attended, _ = self.bucket_attention(
            bucket_encoded, bucket_encoded, bucket_encoded
        )  # (batch, max_buckets, bucket_hidden)

        # Aggregate bucket information (weighted mean based on job counts)
        # Use job counts (second feature) as weights
        job_counts = bucket_features[:, :, 1:2]  # (batch, max_buckets, 1)
        weights = torch.softmax(job_counts, dim=1)  # Normalize across buckets
        bucket_aggregated = (bucket_attended * weights).sum(dim=1)  # (batch, bucket_hidden)

        # ========== Process Global State ==========
        global_encoded = self.global_encoder(global_features)  # (batch, global_hidden)

        # ========== Combine Information ==========
        combined = torch.cat([bucket_aggregated, global_encoded], dim=-1)
        combined_features = self.combiner(combined)  # (batch, combined_hidden)

        # ========== Generate Outputs ==========
        # Actor output: priorities for each bucket
        action_mean = self.actor_head(combined_features)  # (batch, action_dim)

        # Apply sigmoid to ensure priorities are in [0, 1]
        action_mean = torch.sigmoid(action_mean)

        # Expand log_std to match batch size
        action_logstd = self.log_std.expand_as(action_mean)

        # Apply exploration noise decay
        action_logstd = action_logstd * self.current_exploration_noise

        # Critic output: value estimate
        value = self.critic_head(combined_features)  # (batch, 1)

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
        if not deterministic and self.training:
            exploration_factor = max(self.current_exploration_noise, self.min_exploration_noise)
            action_logstd = action_logstd * exploration_factor

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
        """Decay exploration noise over time."""
        self.current_exploration_noise = max(
            self.current_exploration_noise * self.exploration_noise_decay,
            self.min_exploration_noise
        )