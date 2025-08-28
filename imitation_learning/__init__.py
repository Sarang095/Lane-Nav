"""
Imitation Learning Framework for Highway-Env
CNN-based imitation learning with RL integration capabilities
"""

__version__ = "1.0.0"
__author__ = "Highway-Env Imitation Learning Team"

from .models.cnn_policy import ImitationCNNPolicy, create_policy_for_env
from .data_collection.expert_data_collector import ExpertDataCollector, Trajectory
from .training.imitation_trainer import ImitationLearningTrainer, train_imitation_learning_pipeline
from .evaluation.evaluator import ImitationLearningEvaluator, run_comprehensive_evaluation
from .integration.rl_integration import create_warm_start_rl_model, train_rl_with_il_warmstart

__all__ = [
    # Models
    'ImitationCNNPolicy',
    'create_policy_for_env',
    
    # Data Collection
    'ExpertDataCollector',
    'Trajectory',
    
    # Training
    'ImitationLearningTrainer',
    'train_imitation_learning_pipeline',
    
    # Evaluation
    'ImitationLearningEvaluator',
    'run_comprehensive_evaluation',
    
    # RL Integration
    'create_warm_start_rl_model',
    'train_rl_with_il_warmstart',
]