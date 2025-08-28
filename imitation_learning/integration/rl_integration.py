"""
Integration scripts for loading Imitation Learning weights into RL models
Provides warm-start capabilities for RL training with pre-trained IL policies
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path
import warnings

from stable_baselines3 import DQN, PPO, A2C, SAC
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.policies import ActorCriticPolicy

from ..models.cnn_policy import CNNFeaturesExtractor, MLPFeaturesExtractor


class ImitationToRLAdapter:
    """
    Adapter class to integrate imitation learning weights into RL frameworks
    """
    
    def __init__(self):
        self.supported_algorithms = ['DQN', 'PPO', 'A2C', 'SAC']
    
    def load_il_weights(self, filepath: str) -> Dict[str, Any]:
        """Load imitation learning weights from file"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        if 'policy_weights' in checkpoint:
            # New format (exported from trainer)
            return checkpoint
        else:
            # Old format (direct model checkpoint)
            return {
                'policy_weights': {
                    'features_extractor': checkpoint['model_state_dict'],
                },
                'model_config': checkpoint.get('model_config', {}),
                'training_info': checkpoint.get('training_info', {}),
            }
    
    def create_rl_policy_with_il_weights(
        self,
        algorithm: str,
        env,
        il_weights_path: str,
        rl_config: Optional[Dict] = None,
    ):
        """
        Create RL policy initialized with imitation learning weights
        
        Args:
            algorithm: RL algorithm ('DQN', 'PPO', 'A2C', 'SAC')
            env: Gymnasium environment
            il_weights_path: Path to IL weights file
            rl_config: RL algorithm configuration
        
        Returns:
            RL model with initialized weights
        """
        if algorithm not in self.supported_algorithms:
            raise ValueError(f"Algorithm {algorithm} not supported. "
                           f"Supported: {self.supported_algorithms}")
        
        # Load IL weights
        il_data = self.load_il_weights(il_weights_path)
        policy_weights = il_data['policy_weights']
        model_config = il_data.get('model_config', {})
        
        # Default RL configurations
        if rl_config is None:
            rl_config = self._get_default_rl_config(algorithm)
        
        # Modify policy kwargs to match IL architecture
        policy_kwargs = rl_config.get('policy_kwargs', {})
        policy_kwargs = self._adapt_policy_kwargs(policy_kwargs, model_config)
        rl_config['policy_kwargs'] = policy_kwargs
        
        # Create RL model
        if algorithm == 'DQN':
            model = self._create_dqn_with_il_weights(env, policy_weights, rl_config)
        elif algorithm in ['PPO', 'A2C']:
            model = self._create_actor_critic_with_il_weights(
                algorithm, env, policy_weights, rl_config
            )
        elif algorithm == 'SAC':
            model = self._create_sac_with_il_weights(env, policy_weights, rl_config)
        else:
            raise NotImplementedError(f"Algorithm {algorithm} not implemented yet")
        
        print(f"Created {algorithm} model with IL weights initialization")
        return model
    
    def _get_default_rl_config(self, algorithm: str) -> Dict[str, Any]:
        """Get default configuration for RL algorithms"""
        configs = {
            'DQN': {
                'learning_rate': 1e-4,
                'buffer_size': 50000,
                'learning_starts': 1000,
                'batch_size': 32,
                'gamma': 0.99,
                'target_update_interval': 1000,
                'train_freq': 4,
                'exploration_fraction': 0.1,
                'exploration_final_eps': 0.02,
                'policy_kwargs': dict(net_arch=[256, 256]),
            },
            'PPO': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'policy_kwargs': dict(net_arch=[256, 256]),
            },
            'A2C': {
                'learning_rate': 7e-4,
                'n_steps': 5,
                'gamma': 0.99,
                'gae_lambda': 1.0,
                'policy_kwargs': dict(net_arch=[256, 256]),
            },
            'SAC': {
                'learning_rate': 3e-4,
                'buffer_size': 100000,
                'batch_size': 256,
                'gamma': 0.99,
                'tau': 0.005,
                'policy_kwargs': dict(net_arch=[256, 256]),
            },
        }
        
        return configs.get(algorithm, {})
    
    def _adapt_policy_kwargs(
        self, 
        policy_kwargs: Dict, 
        model_config: Dict
    ) -> Dict:
        """Adapt policy kwargs based on IL model configuration"""
        # Extract features dimension from IL model
        features_dim = model_config.get('features_dim', 512)
        use_cnn = model_config.get('use_cnn', False)
        
        if use_cnn:
            # Use CNN features extractor
            policy_kwargs['features_extractor_class'] = CNNFeaturesExtractor
            policy_kwargs['features_extractor_kwargs'] = {'features_dim': features_dim}
        else:
            # Use MLP features extractor
            policy_kwargs['features_extractor_class'] = MLPFeaturesExtractor
            policy_kwargs['features_extractor_kwargs'] = {'features_dim': features_dim}
        
        # Adapt network architecture
        if 'net_arch' not in policy_kwargs:
            policy_kwargs['net_arch'] = [features_dim, features_dim // 2]
        
        return policy_kwargs
    
    def _create_dqn_with_il_weights(
        self,
        env,
        policy_weights: Dict,
        rl_config: Dict,
    ) -> DQN:
        """Create DQN model with IL weights"""
        # Create DQN model
        model = DQN(
            policy="CnnPolicy" if "cnn" in str(rl_config.get('policy_kwargs', {})) else "MlpPolicy",
            env=env,
            **{k: v for k, v in rl_config.items() if k != 'policy_kwargs'},
            policy_kwargs=rl_config.get('policy_kwargs', {}),
            verbose=1,
        )
        
        # Load IL weights into the policy
        self._load_weights_into_dqn_policy(model.policy, policy_weights)
        
        return model
    
    def _create_actor_critic_with_il_weights(
        self,
        algorithm: str,
        env,
        policy_weights: Dict,
        rl_config: Dict,
    ) -> Union[PPO, A2C]:
        """Create Actor-Critic model (PPO/A2C) with IL weights"""
        # Select algorithm class
        AlgClass = PPO if algorithm == 'PPO' else A2C
        
        # Create model
        model = AlgClass(
            policy="CnnPolicy" if "cnn" in str(rl_config.get('policy_kwargs', {})) else "MlpPolicy",
            env=env,
            **{k: v for k, v in rl_config.items() if k != 'policy_kwargs'},
            policy_kwargs=rl_config.get('policy_kwargs', {}),
            verbose=1,
        )
        
        # Load IL weights into the policy
        self._load_weights_into_actor_critic_policy(model.policy, policy_weights)
        
        return model
    
    def _create_sac_with_il_weights(
        self,
        env,
        policy_weights: Dict,
        rl_config: Dict,
    ) -> SAC:
        """Create SAC model with IL weights"""
        # Create SAC model
        model = SAC(
            policy="CnnPolicy" if "cnn" in str(rl_config.get('policy_kwargs', {})) else "MlpPolicy",
            env=env,
            **{k: v for k, v in rl_config.items() if k != 'policy_kwargs'},
            policy_kwargs=rl_config.get('policy_kwargs', {}),
            verbose=1,
        )
        
        # Load IL weights into the policy
        self._load_weights_into_sac_policy(model.policy, policy_weights)
        
        return model
    
    def _load_weights_into_dqn_policy(
        self,
        policy: DQNPolicy,
        il_weights: Dict,
    ):
        """Load IL weights into DQN policy"""
        try:
            # Load features extractor weights
            if 'features_extractor' in il_weights:
                self._safe_load_state_dict(
                    policy.q_net.features_extractor,
                    il_weights['features_extractor']
                )
            
            # Initialize Q-network head with IL action head weights if available
            if 'action_head' in il_weights:
                self._safe_load_state_dict(
                    policy.q_net.q_net,
                    il_weights['action_head'],
                    partial=True
                )
            
            print("Successfully loaded IL weights into DQN policy")
            
        except Exception as e:
            warnings.warn(f"Could not load all IL weights into DQN policy: {e}")
    
    def _load_weights_into_actor_critic_policy(
        self,
        policy: ActorCriticPolicy,
        il_weights: Dict,
    ):
        """Load IL weights into Actor-Critic policy"""
        try:
            # Load features extractor weights
            if 'features_extractor' in il_weights:
                self._safe_load_state_dict(
                    policy.features_extractor,
                    il_weights['features_extractor']
                )
            
            # Load action head into actor
            if 'action_head' in il_weights:
                self._safe_load_state_dict(
                    policy.action_net,
                    il_weights['action_head'],
                    partial=True
                )
            
            # Load value head into critic
            if 'value_head' in il_weights:
                self._safe_load_state_dict(
                    policy.value_net,
                    il_weights['value_head'],
                    partial=True
                )
            
            print("Successfully loaded IL weights into Actor-Critic policy")
            
        except Exception as e:
            warnings.warn(f"Could not load all IL weights into Actor-Critic policy: {e}")
    
    def _load_weights_into_sac_policy(
        self,
        policy,
        il_weights: Dict,
    ):
        """Load IL weights into SAC policy"""
        try:
            # Load features extractor weights
            if 'features_extractor' in il_weights:
                # SAC has separate features extractors for actor and critic
                self._safe_load_state_dict(
                    policy.actor.features_extractor,
                    il_weights['features_extractor']
                )
                
                if hasattr(policy, 'critic'):
                    self._safe_load_state_dict(
                        policy.critic.features_extractor,
                        il_weights['features_extractor']
                    )
            
            # Load action head into actor
            if 'action_head' in il_weights:
                self._safe_load_state_dict(
                    policy.actor.mu,
                    il_weights['action_head'],
                    partial=True
                )
            
            print("Successfully loaded IL weights into SAC policy")
            
        except Exception as e:
            warnings.warn(f"Could not load all IL weights into SAC policy: {e}")
    
    def _safe_load_state_dict(
        self,
        module: nn.Module,
        state_dict: Dict,
        partial: bool = False,
    ):
        """Safely load state dict with size checking"""
        module_state = module.state_dict()
        
        # Filter state dict to only include compatible keys
        filtered_state = {}
        for key, value in state_dict.items():
            if key in module_state:
                if module_state[key].shape == value.shape:
                    filtered_state[key] = value
                else:
                    if not partial:
                        print(f"Skipping {key}: shape mismatch "
                              f"{module_state[key].shape} vs {value.shape}")
            else:
                if not partial:
                    print(f"Skipping {key}: not found in target module")
        
        # Load filtered state dict
        if filtered_state:
            module.load_state_dict(filtered_state, strict=False)
            print(f"Loaded {len(filtered_state)}/{len(state_dict)} parameters")
        else:
            print("No compatible parameters found to load")


def create_warm_start_rl_model(
    env_name: str,
    algorithm: str,
    il_weights_path: str,
    env_config: Optional[Dict] = None,
    rl_config: Optional[Dict] = None,
):
    """
    Convenience function to create RL model with IL warm start
    
    Args:
        env_name: Highway environment name
        algorithm: RL algorithm ('DQN', 'PPO', 'A2C', 'SAC')
        il_weights_path: Path to IL weights
        env_config: Environment configuration
        rl_config: RL algorithm configuration
    
    Returns:
        RL model ready for training
    """
    import gymnasium as gym
    import highway_env
    
    # Create environment
    env = gym.make(env_name)
    if env_config:
        env.unwrapped.configure(env_config)
    
    # Create adapter and model
    adapter = ImitationToRLAdapter()
    model = adapter.create_rl_policy_with_il_weights(
        algorithm=algorithm,
        env=env,
        il_weights_path=il_weights_path,
        rl_config=rl_config,
    )
    
    return model, env


def train_rl_with_il_warmstart(
    env_name: str,
    algorithm: str,
    il_weights_path: str,
    total_timesteps: int = 100000,
    env_config: Optional[Dict] = None,
    rl_config: Optional[Dict] = None,
    save_path: Optional[str] = None,
) -> Any:
    """
    Complete pipeline for training RL with IL warm start
    
    Args:
        env_name: Highway environment name
        algorithm: RL algorithm
        il_weights_path: Path to IL weights
        total_timesteps: Training timesteps
        env_config: Environment configuration
        rl_config: RL configuration
        save_path: Path to save trained model
    
    Returns:
        Trained RL model
    """
    print(f"Training {algorithm} on {env_name} with IL warm start")
    print(f"IL weights: {il_weights_path}")
    print(f"Total timesteps: {total_timesteps}")
    
    # Create model
    model, env = create_warm_start_rl_model(
        env_name=env_name,
        algorithm=algorithm,
        il_weights_path=il_weights_path,
        env_config=env_config,
        rl_config=rl_config,
    )
    
    # Train model
    print("Starting RL training...")
    model.learn(total_timesteps=total_timesteps)
    
    # Save model
    if save_path:
        model.save(save_path)
        print(f"Saved trained model to {save_path}")
    
    env.close()
    return model


class PerformanceComparator:
    """
    Compare performance between IL-warmstarted RL and standard RL
    """
    
    def __init__(self):
        self.results = {}
    
    def compare_training_performance(
        self,
        env_name: str,
        algorithm: str,
        il_weights_path: str,
        total_timesteps: int = 50000,
        num_trials: int = 3,
        env_config: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Compare IL-warmstarted vs standard RL training
        
        Args:
            env_name: Environment name
            algorithm: RL algorithm
            il_weights_path: Path to IL weights
            total_timesteps: Training timesteps
            num_trials: Number of trials for averaging
            env_config: Environment configuration
        
        Returns:
            Comparison results
        """
        import gymnasium as gym
        import highway_env
        
        print(f"Comparing {algorithm} performance: IL-warmstart vs standard")
        
        warmstart_rewards = []
        standard_rewards = []
        
        for trial in range(num_trials):
            print(f"Trial {trial + 1}/{num_trials}")
            
            # Train with IL warmstart
            print("Training with IL warmstart...")
            model_warmstart, env = create_warm_start_rl_model(
                env_name, algorithm, il_weights_path, env_config
            )
            
            # Monitor training
            warmstart_callback = self._create_monitoring_callback()
            model_warmstart.learn(total_timesteps=total_timesteps, callback=warmstart_callback)
            warmstart_rewards.append(warmstart_callback.episode_rewards)
            
            # Train standard RL
            print("Training standard RL...")
            if algorithm == 'DQN':
                model_standard = DQN("MlpPolicy", env, verbose=0)
            elif algorithm == 'PPO':
                model_standard = PPO("MlpPolicy", env, verbose=0)
            elif algorithm == 'A2C':
                model_standard = A2C("MlpPolicy", env, verbose=0)
            else:
                raise ValueError(f"Algorithm {algorithm} not supported for comparison")
            
            standard_callback = self._create_monitoring_callback()
            model_standard.learn(total_timesteps=total_timesteps, callback=standard_callback)
            standard_rewards.append(standard_callback.episode_rewards)
            
            env.close()
        
        # Analyze results
        results = {
            'warmstart_rewards': warmstart_rewards,
            'standard_rewards': standard_rewards,
            'warmstart_mean': np.mean([np.mean(r) for r in warmstart_rewards]),
            'standard_mean': np.mean([np.mean(r) for r in standard_rewards]),
            'improvement': None,
        }
        
        if results['standard_mean'] != 0:
            improvement = (results['warmstart_mean'] - results['standard_mean']) / abs(results['standard_mean'])
            results['improvement'] = improvement
        
        print(f"IL-Warmstart Mean Reward: {results['warmstart_mean']:.2f}")
        print(f"Standard RL Mean Reward: {results['standard_mean']:.2f}")
        if results['improvement']:
            print(f"Improvement: {results['improvement']*100:.1f}%")
        
        return results
    
    def _create_monitoring_callback(self):
        """Create callback for monitoring training progress"""
        from stable_baselines3.common.callbacks import BaseCallback
        
        class RewardMonitor(BaseCallback):
            def __init__(self):
                super().__init__()
                self.episode_rewards = []
                self.current_episode_reward = 0
            
            def _on_step(self):
                self.current_episode_reward += self.locals['rewards'][0]
                
                if self.locals['dones'][0]:
                    self.episode_rewards.append(self.current_episode_reward)
                    self.current_episode_reward = 0
                
                return True
        
        return RewardMonitor()