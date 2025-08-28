# Highway-Env Imitation Learning Framework

A comprehensive CNN-based imitation learning framework for autonomous vehicle training in highway-env environments. This framework provides seamless integration with reinforcement learning models for warm-start training and cold-start prevention.

## 🎯 Overview

This framework integrates imitation learning capabilities into the highway-env autonomous driving simulation environment. It supports:

- **CNN-based Policy Networks**: Visual and vector observation processing
- **Expert Data Collection**: Rule-based and human expert demonstrations
- **Behavioral Cloning Training**: Supervised learning from expert trajectories
- **RL Integration**: Warm-start RL training with pre-trained IL weights
- **Comprehensive Evaluation**: Multi-environment testing and benchmarking

## 🏗️ Architecture

```
highway-env/
├── imitation_learning/
│   ├── models/              # CNN and MLP policy networks
│   ├── data_collection/     # Expert demonstration collection
│   ├── training/           # Imitation learning training pipeline
│   ├── evaluation/         # Model evaluation and benchmarking
│   └── integration/        # RL framework integration
├── scripts/                # Original RL training scripts
└── test_*.py              # Comprehensive testing suite
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install highway-env with dependencies
pip install -e .

# Install additional ML packages
pip install torch torchvision stable-baselines3[extra] opencv-python scikit-learn
```

### 2. Basic Usage

```python
import gymnasium as gym
import highway_env
from imitation_learning import (
    ExpertDataCollector,
    ImitationLearningTrainer,
    create_policy_for_env,
    create_warm_start_rl_model
)

# 1. Collect Expert Data
env_config = {
    'observation': {'type': 'Kinematics'},
    'action': {'type': 'DiscreteMetaAction'},
    'duration': 40,
}

collector = ExpertDataCollector(env_config=env_config)
trajectories = collector.collect_rule_based_expert_data(
    env_name='highway-v0',
    expert_type='idm',
    num_episodes=50
)

# 2. Train Imitation Learning Model
env = gym.make('highway-v0')
env.unwrapped.configure(env_config)

policy = create_policy_for_env('highway-v0', env.observation_space, env.action_space)
trainer = ImitationLearningTrainer(policy=policy, num_epochs=100)

train_loader, val_loader = trainer.prepare_data(trajectories)
trainer.train_behavioral_cloning(train_loader, val_loader)

# Export weights for RL
trainer.export_for_rl('highway_il_weights.pth')

# 3. Create RL Model with IL Warm Start
rl_model, env = create_warm_start_rl_model(
    env_name='highway-v0',
    algorithm='DQN',
    il_weights_path='highway_il_weights.pth',
    env_config=env_config
)

# Train RL model (with warm start)
rl_model.learn(total_timesteps=100000)
```

## 🌟 Supported Environments

| Environment | Description | Observation Type | Action Type |
|-------------|-------------|------------------|-------------|
| `highway-v0` | Multi-lane highway driving | Kinematics | Discrete |
| `highway-fast-v0` | Highway with CNN observations | Grayscale Images | Discrete |
| `intersection-v0` | Traffic intersection navigation | Kinematics | Discrete |
| `roundabout-v0` | Roundabout navigation | Kinematics | Discrete |
| `parking-v0` | Parking maneuver | Kinematics | Continuous |

## 🔧 Configuration Examples

### Highway Environment with CNN

```python
env_config = {
    'observation': {
        'type': 'GrayscaleObservation',
        'observation_shape': (128, 64),
        'stack_size': 4,
        'weights': [0.2989, 0.5870, 0.1140],
        'scaling': 1.75,
    },
    'action': {'type': 'DiscreteMetaAction'},
    'duration': 40,
    'vehicles_count': 50,
}
```

### Intersection Environment

```python
env_config = {
    'observation': {'type': 'Kinematics'},
    'action': {'type': 'DiscreteMetaAction'},
    'duration': 20,
    'destination': 'o1',
    'collision_reward': -5,
    'reached_goal_reward': 1,
}
```

### Parking Environment (Continuous Control)

```python
env_config = {
    'observation': {'type': 'Kinematics'},
    'action': {'type': 'ContinuousAction'},
    'duration': 25,
    'collision_reward': -5,
    'success_goal_reward': 1,
}
```

## 📊 Expert Data Collection

### Rule-based Experts

```python
# IDM (Intelligent Driver Model) Expert
trajectories = collector.collect_rule_based_expert_data(
    env_name='highway-v0',
    expert_type='idm',  # Conservative, safe driving
)

# Aggressive Expert
trajectories = collector.collect_rule_based_expert_data(
    env_name='highway-v0',
    expert_type='aggressive',  # Higher speeds, more lane changes
)

# Planned Expert
trajectories = collector.collect_rule_based_expert_data(
    env_name='intersection-v0',
    expert_type='planned',  # Goal-directed behavior
)
```

### Data Quality Control

```python
collector = ExpertDataCollector(
    env_config=env_config,
    max_episodes=100,
    min_episode_length=20,        # Minimum trajectory length
    quality_threshold=0.0,        # Minimum reward threshold
)
```

## 🎯 Training Configuration

### Basic Training

```python
trainer = ImitationLearningTrainer(
    policy=policy,
    learning_rate=1e-3,
    batch_size=64,
    num_epochs=100,
    validation_split=0.2,
)
```

### Advanced Training with Monitoring

```python
trainer = ImitationLearningTrainer(
    policy=policy,
    learning_rate=1e-3,
    batch_size=32,
    num_epochs=200,
    validation_split=0.2,
    save_dir="./trained_models",
    log_interval=10,
)

# Train with value loss (if available)
training_history = trainer.train_behavioral_cloning(
    train_loader, 
    val_loader,
    use_value_loss=True,
    value_loss_weight=0.5
)

# Plot training curves
trainer.plot_training_history("training_curves.png")
```

## 🔗 RL Integration

### Supported RL Algorithms

- **DQN**: Deep Q-Network
- **PPO**: Proximal Policy Optimization  
- **A2C**: Advantage Actor-Critic
- **SAC**: Soft Actor-Critic

### Warm Start Example

```python
from imitation_learning.integration import train_rl_with_il_warmstart

# Complete pipeline: IL warmstart → RL training
trained_model = train_rl_with_il_warmstart(
    env_name='highway-v0',
    algorithm='DQN',
    il_weights_path='highway_il_weights.pth',
    total_timesteps=100000,
    env_config=env_config,
    save_path='warmstart_dqn_model'
)
```

### Performance Comparison

```python
from imitation_learning.integration import PerformanceComparator

comparator = PerformanceComparator()
results = comparator.compare_training_performance(
    env_name='highway-v0',
    algorithm='DQN',
    il_weights_path='highway_il_weights.pth',
    total_timesteps=50000,
    num_trials=3,
)

print(f"Improvement: {results['improvement']*100:.1f}%")
```

## 🧪 Testing

### Run Comprehensive Tests

```bash
# Test all components and environments
python test_imitation_learning.py

# Quick test mode
python test_imitation_learning.py --quick

# Test specific environment
python test_imitation_learning.py --env highway-v0
```

### Test Individual Environments

```bash
# Test all environments individually
python test_individual_environments.py --all

# Test specific environment with rendering
python test_individual_environments.py --env parking-v0 --render

# Quick test mode
python test_individual_environments.py --all --quick
```

### Test Results

Test results are saved in structured JSON format:

```json
{
  "environment_creation": {"highway-v0": true, ...},
  "data_collection": {"highway-v0": {"success": true, "num_trajectories": 50}},
  "training_pipeline": {"highway-v0": {"success": true, "final_val_loss": 0.123}},
  "evaluation": {"highway-v0": {"success": true, "mean_reward": 25.4}},
  "rl_integration": {"highway-v0": {"success": true, "warmstart_working": true}}
}
```

## 📈 Evaluation and Benchmarking

### Model Evaluation

```python
from imitation_learning.evaluation import ImitationLearningEvaluator

evaluator = ImitationLearningEvaluator(trained_policy)

# Evaluate on single environment
results = evaluator.evaluate_single_environment(
    env_name='highway-v0',
    num_episodes=20,
    deterministic=True
)

print(f"Mean Reward: {results['mean_reward']:.2f}")
print(f"Success Rate: {results['success_rate']:.2f}")
print(f"Collision Rate: {results['collision_rate']:.2f}")
```

### Baseline Comparison

```python
# Compare against baselines
comparison = evaluator.benchmark_against_baselines(
    env_name='highway-v0',
    baselines=['random', 'idm', 'aggressive']
)

# Plot comparison
evaluator.plot_comparison(comparison, save_path='comparison.png')
```

### Multi-Environment Evaluation

```python
from imitation_learning.evaluation import run_comprehensive_evaluation

results = run_comprehensive_evaluation(
    model=trained_policy,
    save_dir="./evaluation_results"
)
```

## 🛠️ Advanced Features

### Hybrid Policy (Multi-Modal Observations)

```python
from imitation_learning.models import HybridPolicy

# For environments with both image and vector observations
hybrid_policy = HybridPolicy(
    image_observation_space=image_space,
    vector_observation_space=vector_space,
    action_space=action_space,
    features_dim=512
)
```

### Custom Expert Policies

```python
class CustomExpert:
    def get_action(self, env, obs):
        # Implement custom expert behavior
        return action

# Use with data collector
collector.custom_expert = CustomExpert()
```

### Data Augmentation

```python
# Enable data augmentation during training
train_loader, val_loader = trainer.prepare_data(
    trajectories,
    normalize_observations=True,
    augment_data=True  # Adds noise and variations
)
```

## 📝 Logging and Monitoring

### TensorBoard Integration

```python
# Training automatically logs to tensorboard
# View with: tensorboard --logdir ./trained_models
```

### Custom Callbacks

```python
class CustomCallback:
    def on_epoch_end(self, epoch, logs):
        # Custom logging logic
        pass

trainer.add_callback(CustomCallback())
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
```python
trainer = ImitationLearningTrainer(policy, device='cpu', batch_size=16)
```

2. **Observation Shape Mismatch**: Ensure environment config matches training
```python
# Check observation shapes
print(f"Env obs shape: {env.observation_space.shape}")
print(f"Model expects: {policy.observation_space.shape}")
```

3. **Action Space Compatibility**: Verify action spaces match
```python
print(f"Env action space: {env.action_space}")
print(f"Model action space: {policy.action_space}")
```

### Debug Mode

```python
# Enable verbose logging
trainer = ImitationLearningTrainer(policy, verbose=True)

# Use small dataset for debugging
trajectories = trajectories[:5]  # Use only 5 trajectories
```

## 📚 Examples and Tutorials

### Complete Example: Highway Environment

```python
#!/usr/bin/env python3
"""
Complete example: Highway environment with CNN-based imitation learning
"""

import gymnasium as gym
import highway_env
from imitation_learning import *

def main():
    # 1. Environment configuration
    env_config = {
        'observation': {
            'type': 'GrayscaleObservation',
            'observation_shape': (128, 64),
            'stack_size': 4,
            'weights': [0.2989, 0.5870, 0.1140],
            'scaling': 1.75,
        },
        'action': {'type': 'DiscreteMetaAction'},
        'duration': 40,
        'vehicles_count': 30,
    }
    
    # 2. Collect expert data
    print("Collecting expert data...")
    collector = ExpertDataCollector(
        env_config=env_config,
        max_episodes=100,
        min_episode_length=20,
    )
    
    trajectories = collector.collect_rule_based_expert_data(
        env_name='highway-fast-v0',
        expert_type='idm',
        render=False,
    )
    
    collector.save_data('highway_expert_data.pkl')
    
    # 3. Train imitation learning model
    print("Training imitation learning model...")
    env = gym.make('highway-fast-v0')
    env.unwrapped.configure(env_config)
    
    policy = create_policy_for_env(
        'highway-fast-v0', 
        env.observation_space, 
        env.action_space
    )
    
    trainer = ImitationLearningTrainer(
        policy=policy,
        num_epochs=100,
        batch_size=32,
        learning_rate=1e-3,
    )
    
    train_loader, val_loader = trainer.prepare_data(trajectories)
    trainer.train_behavioral_cloning(train_loader, val_loader)
    
    trainer.save_model('highway_il_model.pth')
    trainer.export_for_rl('highway_rl_weights.pth')
    
    # 4. Evaluate model
    print("Evaluating model...")
    evaluator = ImitationLearningEvaluator(policy)
    results = evaluator.evaluate_single_environment(
        'highway-fast-v0',
        env_config,
        num_episodes=20
    )
    
    print(f"IL Mean Reward: {results['mean_reward']:.2f}")
    print(f"Success Rate: {results['success_rate']:.2f}")
    
    # 5. Create RL model with warm start
    print("Creating RL model with warm start...")
    rl_model, env = create_warm_start_rl_model(
        env_name='highway-fast-v0',
        algorithm='DQN',
        il_weights_path='highway_rl_weights.pth',
        env_config=env_config,
    )
    
    # 6. Train RL model
    print("Training RL model...")
    rl_model.learn(total_timesteps=50000)
    rl_model.save('highway_rl_final')
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `python test_imitation_learning.py`
5. Submit a pull request

## 📄 License

This project extends highway-env and follows its MIT license.

## 🔗 References

- [Highway-Env Repository](https://github.com/eleurent/highway-env)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Imitation Learning Papers](https://arxiv.org/abs/1710.11248)

## 📞 Support

- Check the troubleshooting section
- Run test suite to verify installation
- Review example notebooks in `/examples/`
- Open issues for bugs or feature requests