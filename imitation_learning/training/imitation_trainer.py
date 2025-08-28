"""
Imitation Learning Training Pipeline
Supports Behavioral Cloning and advanced imitation learning techniques
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

from ..models.cnn_policy import ImitationCNNPolicy, create_policy_for_env
from ..data_collection.expert_data_collector import Trajectory, DatasetCreator


class ImitationLearningTrainer:
    """
    Main trainer class for imitation learning with CNN policies
    Supports multiple training strategies and evaluation metrics
    """
    
    def __init__(
        self,
        policy: nn.Module,
        device: str = "auto",
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        num_epochs: int = 100,
        validation_split: float = 0.2,
        save_dir: str = "./trained_models",
        log_interval: int = 10,
    ):
        self.policy = policy
        self.device = self._get_device(device)
        self.policy.to(self.device)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.validation_split = validation_split
        self.log_interval = log_interval
        
        # Setup directories
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training components
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5
        )
        
        # Training state
        self.training_history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        
        # Loss functions
        if hasattr(policy, 'is_discrete') and policy.is_discrete:
            self.action_loss_fn = nn.CrossEntropyLoss()
        else:
            self.action_loss_fn = nn.MSELoss()
        
        self.value_loss_fn = nn.MSELoss()
    
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device for training"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def prepare_data(
        self,
        trajectories: List[Trajectory],
        normalize_observations: bool = True,
        augment_data: bool = False,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data loaders from trajectories
        """
        print("Preparing dataset from trajectories...")
        
        # Create dataset
        dataset_creator = DatasetCreator(trajectories)
        observations, actions = dataset_creator.create_torch_dataset(
            normalize_observations=normalize_observations,
            augment_data=augment_data,
        )
        
        print(f"Dataset size: {len(observations)} samples")
        print(f"Observation shape: {observations.shape}")
        print(f"Action shape: {actions.shape}")
        
        # Move to device
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        
        # Create dataset
        dataset = TensorDataset(observations, actions)
        
        # Split into train/validation
        val_size = int(len(dataset) * self.validation_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_behavioral_cloning(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        use_value_loss: bool = False,
        value_loss_weight: float = 0.5,
    ) -> Dict[str, List[float]]:
        """
        Train policy using Behavioral Cloning (BC)
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            use_value_loss: Whether to include value function loss
            value_loss_weight: Weight for value loss term
        """
        print(f"Starting Behavioral Cloning training for {self.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.batch_size}")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, use_value_loss, value_loss_weight)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader, use_value_loss, value_loss_weight)
            
            # Update learning rate
            self.scheduler.step(val_metrics['total_loss'])
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.save_model(f"best_model_epoch_{epoch}.pth")
            
            # Save periodic checkpoints
            if (epoch + 1) % 20 == 0:
                self.save_model(f"checkpoint_epoch_{epoch}.pth")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return dict(self.training_history)
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        use_value_loss: bool,
        value_loss_weight: float,
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.policy.train()
        
        total_loss = 0.0
        action_loss = 0.0
        value_loss = 0.0
        num_batches = 0
        
        for batch_idx, (observations, actions) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.policy, 'forward'):
                pred_actions, pred_values = self.policy(observations)
            else:
                pred_actions = self.policy(observations)
                pred_values = None
            
            # Compute losses
            if self.policy.is_discrete:
                a_loss = self.action_loss_fn(pred_actions, actions)
            else:
                a_loss = self.action_loss_fn(pred_actions, actions)
            
            loss = a_loss
            
            if use_value_loss and pred_values is not None:
                # For BC, we don't have value targets, so we skip value loss
                # or use reward-to-go as targets
                v_loss = torch.zeros_like(a_loss)  # Placeholder
                loss += value_loss_weight * v_loss
                value_loss += v_loss.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            action_loss += a_loss.item()
            num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'action_loss': action_loss / num_batches,
            'value_loss': value_loss / num_batches if use_value_loss else 0.0,
        }
    
    def _validate_epoch(
        self,
        val_loader: DataLoader,
        use_value_loss: bool,
        value_loss_weight: float,
    ) -> Dict[str, float]:
        """Validate for one epoch"""
        self.policy.eval()
        
        total_loss = 0.0
        action_loss = 0.0
        value_loss = 0.0
        num_batches = 0
        accuracy = 0.0
        
        with torch.no_grad():
            for observations, actions in val_loader:
                # Forward pass
                if hasattr(self.policy, 'forward'):
                    pred_actions, pred_values = self.policy(observations)
                else:
                    pred_actions = self.policy(observations)
                    pred_values = None
                
                # Compute losses
                if self.policy.is_discrete:
                    a_loss = self.action_loss_fn(pred_actions, actions)
                    # Compute accuracy
                    pred_labels = torch.argmax(pred_actions, dim=1)
                    accuracy += (pred_labels == actions).float().mean().item()
                else:
                    a_loss = self.action_loss_fn(pred_actions, actions)
                
                loss = a_loss
                
                if use_value_loss and pred_values is not None:
                    v_loss = torch.zeros_like(a_loss)  # Placeholder
                    loss += value_loss_weight * v_loss
                    value_loss += v_loss.item()
                
                total_loss += loss.item()
                action_loss += a_loss.item()
                num_batches += 1
        
        metrics = {
            'total_loss': total_loss / num_batches,
            'action_loss': action_loss / num_batches,
            'value_loss': value_loss / num_batches if use_value_loss else 0.0,
        }
        
        if self.policy.is_discrete:
            metrics['accuracy'] = accuracy / num_batches
        
        return metrics
    
    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ):
        """Log training metrics"""
        # Store metrics
        for key, value in train_metrics.items():
            self.training_history[f'train_{key}'].append(value)
        
        for key, value in val_metrics.items():
            self.training_history[f'val_{key}'].append(value)
        
        # Print metrics
        if (epoch + 1) % self.log_interval == 0:
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f} | "
                  f"Val Loss: {val_metrics['total_loss']:.4f}")
            print(f"  Train Action Loss: {train_metrics['action_loss']:.4f} | "
                  f"Val Action Loss: {val_metrics['action_loss']:.4f}")
            
            if 'accuracy' in val_metrics:
                print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  Learning Rate: {current_lr:.6f}")
            print()
    
    def save_model(self, filename: str) -> str:
        """Save model checkpoint"""
        filepath = self.save_dir / filename
        
        checkpoint = {
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': dict(self.training_history),
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'model_config': {
                'observation_space': str(self.policy.observation_space),
                'action_space': str(self.policy.action_space),
                'use_cnn': getattr(self.policy, 'use_cnn', True),
            }
        }
        
        torch.save(checkpoint, filepath)
        return str(filepath)
    
    def load_model(self, filepath: str) -> Dict[str, Any]:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_history = defaultdict(list, checkpoint['training_history'])
        self.current_epoch = checkpoint['current_epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded model from epoch {self.current_epoch}")
        return checkpoint
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss plots
        axes[0, 0].plot(self.training_history['train_total_loss'], label='Train')
        axes[0, 0].plot(self.training_history['val_total_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.training_history['train_action_loss'], label='Train')
        axes[0, 1].plot(self.training_history['val_action_loss'], label='Validation')
        axes[0, 1].set_title('Action Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Accuracy plot (if available)
        if 'val_accuracy' in self.training_history:
            axes[1, 0].plot(self.training_history['val_accuracy'])
            axes[1, 0].set_title('Validation Accuracy')
            axes[1, 0].grid(True)
        
        # Learning rate plot
        if len(self.training_history['train_total_loss']) > 0:
            lr_history = [self.optimizer.param_groups[0]['lr']] * len(self.training_history['train_total_loss'])
            axes[1, 1].plot(lr_history)
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def export_for_rl(self, filepath: str) -> str:
        """
        Export trained policy weights in format compatible with RL frameworks
        """
        # Extract only the policy weights (without optimizer state)
        policy_weights = {
            'features_extractor': self.policy.features_extractor.state_dict(),
            'action_head': self.policy.action_head.state_dict(),
        }
        
        if hasattr(self.policy, 'value_head'):
            policy_weights['value_head'] = self.policy.value_head.state_dict()
        
        export_data = {
            'policy_weights': policy_weights,
            'model_config': {
                'observation_space': str(self.policy.observation_space),
                'action_space': str(self.policy.action_space),
                'use_cnn': getattr(self.policy, 'use_cnn', True),
                'features_dim': getattr(self.policy.features_extractor, 'features_dim', 512),
            },
            'training_info': {
                'final_epoch': self.current_epoch,
                'best_val_loss': self.best_val_loss,
                'training_complete': True,
            }
        }
        
        torch.save(export_data, filepath)
        print(f"Exported RL-compatible weights to {filepath}")
        return filepath


def train_imitation_learning_pipeline(
    env_name: str,
    trajectories: List[Trajectory],
    env_config: Optional[Dict] = None,
    training_config: Optional[Dict] = None,
) -> Tuple[nn.Module, ImitationLearningTrainer]:
    """
    Complete pipeline for training imitation learning on highway-env
    
    Args:
        env_name: Name of the highway environment
        trajectories: List of expert trajectories
        env_config: Environment configuration
        training_config: Training configuration
    
    Returns:
        Trained policy and trainer instance
    """
    import gymnasium as gym
    import highway_env
    
    # Default configurations
    if env_config is None:
        env_config = {}
    
    if training_config is None:
        training_config = {
            'learning_rate': 1e-3,
            'batch_size': 64,
            'num_epochs': 100,
            'validation_split': 0.2,
        }
    
    # Create environment to get spaces
    temp_env = gym.make(env_name)
    if env_config:
        temp_env.unwrapped.configure(env_config)
    
    observation_space = temp_env.observation_space
    action_space = temp_env.action_space
    temp_env.close()
    
    # Create policy
    policy = create_policy_for_env(env_name, observation_space, action_space)
    
    # Create trainer
    trainer = ImitationLearningTrainer(
        policy=policy,
        **training_config
    )
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(trajectories)
    
    # Train
    training_history = trainer.train_behavioral_cloning(train_loader, val_loader)
    
    return policy, trainer