"""
CNN-based Policy Network for Imitation Learning
Compatible with highway-env observation types including GrayscaleObservation and KinematicObservation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Union, Optional
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class CNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    CNN features extractor for image-based observations from highway-env
    Supports GrayscaleObservation and RGB observations
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        # Assume observation_space is either:
        # - (C, H, W) for stacked grayscale images
        # - (H, W, C) for RGB images 
        # - (H, W) for single grayscale image
        
        n_input_channels = self._get_input_channels(observation_space.shape)
        
        # CNN layers for feature extraction (adaptive to input size)
        cnn_input_shape = self._get_cnn_input_shape(observation_space.shape)
        if min(cnn_input_shape[1:]) >= 64:  # Large images
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
        elif min(cnn_input_shape[1:]) >= 32:  # Medium images
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
        else:  # Small images or vector data
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),  # Adaptive pooling to fixed size
                nn.Flatten(),
            )
        
        # Compute the output dimension after CNN
        with torch.no_grad():
            sample_input = torch.zeros(1, *self._get_cnn_input_shape(observation_space.shape))
            cnn_output = self.cnn(sample_input)
            cnn_output_dim = cnn_output.shape[1]
        
        # Final fully connected layer
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU(),
        )
    
    def _get_input_channels(self, obs_shape: Tuple[int, ...]) -> int:
        """Determine number of input channels based on observation shape"""
        if len(obs_shape) == 3:
            # Either (C, H, W) or (H, W, C)
            # Assume (C, H, W) if first dimension is small (channels)
            if obs_shape[0] <= 4:  # Stack size or RGB channels
                return obs_shape[0]
            else:
                return obs_shape[2]  # RGB format (H, W, C)
        elif len(obs_shape) == 2:
            return 1  # Single grayscale image
        else:
            raise ValueError(f"Unsupported observation shape: {obs_shape}")
    
    def _get_cnn_input_shape(self, obs_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Get the input shape for CNN (C, H, W format)"""
        if len(obs_shape) == 3:
            if obs_shape[0] <= 4:  # Already in (C, H, W) format
                return obs_shape
            else:  # Convert from (H, W, C) to (C, H, W)
                return (obs_shape[2], obs_shape[0], obs_shape[1])
        elif len(obs_shape) == 2:
            return (1, obs_shape[0], obs_shape[1])
        else:
            raise ValueError(f"Unsupported observation shape: {obs_shape}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Handle different input formats
        if len(observations.shape) == 3:  # Single observation (H, W, C) or (C, H, W)
            observations = observations.unsqueeze(0)  # Add batch dimension
        
        # Ensure proper format (B, C, H, W)
        if observations.shape[-1] <= 4 and len(observations.shape) == 4:
            # Likely (B, H, W, C) format, convert to (B, C, H, W)
            if observations.shape[1] > 4:  # Height > 4, so it's (B, H, W, C)
                observations = observations.permute(0, 3, 1, 2)
        elif len(observations.shape) == 3:  # (B, H, W) grayscale
            observations = observations.unsqueeze(1)  # Add channel dimension
        
        # Normalize to [0, 1] if needed (assuming uint8 input)
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
        
        return self.linear(self.cnn(observations))


class MLPFeaturesExtractor(BaseFeaturesExtractor):
    """
    MLP features extractor for vector-based observations (KinematicObservation)
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        input_dim = np.prod(observation_space.shape)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations.flatten(start_dim=1))


class ImitationCNNPolicy(nn.Module):
    """
    CNN-based policy for imitation learning that can handle multiple observation types
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_dim: int = 512,
        use_cnn: bool = True,
    ):
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.use_cnn = use_cnn
        
        # Choose appropriate features extractor
        if use_cnn and isinstance(observation_space, spaces.Box) and len(observation_space.shape) >= 2:
            self.features_extractor = CNNFeaturesExtractor(observation_space, features_dim)
        else:
            self.features_extractor = MLPFeaturesExtractor(observation_space, features_dim)
        
        # Action head
        if isinstance(action_space, spaces.Discrete):
            self.action_head = nn.Linear(features_dim, action_space.n)
            self.is_discrete = True
        elif isinstance(action_space, spaces.Box):
            self.action_head = nn.Linear(features_dim, np.prod(action_space.shape))
            self.is_discrete = False
        else:
            raise ValueError(f"Unsupported action space: {action_space}")
        
        # Value head for advantage estimation (optional)
        self.value_head = nn.Linear(features_dim, 1)
    
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action logits/values and state values
        """
        features = self.features_extractor(observations)
        actions = self.action_head(features)
        values = self.value_head(features)
        
        return actions, values
    
    def predict(self, observations: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Predict actions from observations
        """
        actions, _ = self.forward(observations)
        
        if self.is_discrete:
            if deterministic:
                return torch.argmax(actions, dim=-1)
            else:
                probs = F.softmax(actions, dim=-1)
                return torch.multinomial(probs, 1).squeeze(-1)
        else:
            if deterministic:
                return torch.tanh(actions)  # Bounded continuous actions
            else:
                # Add noise for exploration during training
                noise = torch.randn_like(actions) * 0.1
                return torch.tanh(actions + noise)
    
    def get_action_probabilities(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities for discrete actions
        """
        actions, _ = self.forward(observations)
        if self.is_discrete:
            return F.softmax(actions, dim=-1)
        else:
            return actions  # For continuous actions, return raw logits


class HybridPolicy(nn.Module):
    """
    Hybrid policy that can handle both image and vector observations
    Useful for environments that provide multiple observation types
    """
    
    def __init__(
        self,
        image_observation_space: Optional[spaces.Box] = None,
        vector_observation_space: Optional[spaces.Box] = None,
        action_space: spaces.Space = None,
        features_dim: int = 512,
    ):
        super().__init__()
        
        self.action_space = action_space
        total_features_dim = 0
        
        # Image features extractor
        if image_observation_space is not None:
            self.image_extractor = CNNFeaturesExtractor(image_observation_space, features_dim)
            total_features_dim += features_dim
            self.has_image = True
        else:
            self.has_image = False
        
        # Vector features extractor
        if vector_observation_space is not None:
            self.vector_extractor = MLPFeaturesExtractor(vector_observation_space, features_dim // 2)
            total_features_dim += features_dim // 2
            self.has_vector = True
        else:
            self.has_vector = False
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(total_features_dim, features_dim),
            nn.ReLU(),
        )
        
        # Action and value heads
        if isinstance(action_space, spaces.Discrete):
            self.action_head = nn.Linear(features_dim, action_space.n)
            self.is_discrete = True
        elif isinstance(action_space, spaces.Box):
            self.action_head = nn.Linear(features_dim, np.prod(action_space.shape))
            self.is_discrete = False
        
        self.value_head = nn.Linear(features_dim, 1)
    
    def forward(
        self, 
        image_obs: Optional[torch.Tensor] = None,
        vector_obs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional image and vector observations
        """
        features_list = []
        
        if self.has_image and image_obs is not None:
            image_features = self.image_extractor(image_obs)
            features_list.append(image_features)
        
        if self.has_vector and vector_obs is not None:
            vector_features = self.vector_extractor(vector_obs)
            features_list.append(vector_features)
        
        if not features_list:
            raise ValueError("At least one observation type must be provided")
        
        # Concatenate features
        combined_features = torch.cat(features_list, dim=-1)
        fused_features = self.fusion(combined_features)
        
        actions = self.action_head(fused_features)
        values = self.value_head(fused_features)
        
        return actions, values


def create_policy_for_env(env_name: str, observation_space: spaces.Space, action_space: spaces.Space) -> nn.Module:
    """
    Factory function to create appropriate policy based on environment and observation space
    """
    # Determine if we should use CNN based on observation space
    use_cnn = False
    if isinstance(observation_space, spaces.Box):
        if len(observation_space.shape) >= 2:  # Image-like observation
            use_cnn = True
    
    # Create policy based on environment characteristics
    policy = ImitationCNNPolicy(
        observation_space=observation_space,
        action_space=action_space,
        features_dim=512 if use_cnn else 256,
        use_cnn=use_cnn,
    )
    
    return policy