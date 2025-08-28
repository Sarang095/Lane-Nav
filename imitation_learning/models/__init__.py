"""
CNN-based Policy Models for Imitation Learning
"""

from .cnn_policy import (
    ImitationCNNPolicy,
    CNNFeaturesExtractor,
    MLPFeaturesExtractor,
    HybridPolicy,
    create_policy_for_env
)

__all__ = [
    'ImitationCNNPolicy',
    'CNNFeaturesExtractor',
    'MLPFeaturesExtractor',
    'HybridPolicy',
    'create_policy_for_env',
]