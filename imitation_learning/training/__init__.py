"""
Training Pipeline for Imitation Learning
"""

from .imitation_trainer import (
    ImitationLearningTrainer,
    train_imitation_learning_pipeline
)

__all__ = [
    'ImitationLearningTrainer',
    'train_imitation_learning_pipeline',
]