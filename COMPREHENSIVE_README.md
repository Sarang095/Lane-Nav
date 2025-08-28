# Highway-Env: Complete Autonomous Driving Simulation Framework

A comprehensive autonomous driving simulation environment with integrated imitation learning framework for training and evaluating RL agents across multiple driving scenarios.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/highway-env.gif?raw=true"><br/>
    <em>Highway-Env in action across multiple environments</em>
</p>

## 🎯 Project Overview

This project provides a complete framework for autonomous vehicle research with:

- **Multiple Driving Environments**: Highway, intersection, roundabout, parking, merge scenarios
- **Imitation Learning Framework**: CNN and MLP-based behavioral cloning
- **RL Integration**: Seamless warm-start training with pre-trained IL weights
- **Comprehensive Evaluation**: Multi-environment testing and benchmarking
- **Expert Data Collection**: Rule-based and planned expert demonstrations

## 🏗️ Repository Structure

```
highway-env/
├── highway_env/                    # Core environment package
│   ├── envs/                      # Environment implementations
│   │   ├── highway_env.py         # Highway driving
│   │   ├── intersection_env.py    # Traffic intersection
│   │   ├── roundabout_env.py      # Roundabout navigation
│   │   ├── parking_env.py         # Parking maneuvers
│   │   └── merge_env.py           # Highway merging
│   ├── road/                      # Road infrastructure
│   └── vehicle/                   # Vehicle dynamics
├── imitation_learning/             # IL framework
│   ├── models/                    # CNN and MLP policies
│   ├── data_collection/           # Expert data collection
│   ├── training/                  # IL training pipeline
│   ├── evaluation/                # Model evaluation
│   └── integration/               # RL integration
├── scripts/                       # Training examples
│   ├── sb3_highway_dqn.py         # DQN training script
│   ├── sb3_highway_ppo.py         # PPO training script
│   └── parking_her.py             # HER for parking
├── example_usage.py               # Complete workflow examples
├── test_individual_environments.py # Environment testing
└── run_tests.py                   # Comprehensive test suite
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/eleurent/highway-env.git
   cd highway-env
   ```

2. **Install the package**
   ```bash
   # Install in development mode
   pip install -e .
   
   # Install additional ML dependencies
   pip install torch torchvision stable-baselines3[extra] opencv-python scikit-learn
   ```

3. **Verify installation**
   ```bash
   python3 -c "import highway_env; import gymnasium as gym; env = gym.make('highway-v0'); print('✅ Installation successful')"
   ```

## 🌍 Available Environments

### 1. Highway Environment (`highway-v0`)

**Scenario**: Multi-lane highway driving with traffic  
**Objective**: Maintain high speed while avoiding collisions  
**Action Space**: Discrete (5 actions: LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER)  
**Observation**: Kinematics (vehicle positions, speeds, headings)

```python
import gymnasium as gym
import highway_env

env = gym.make("highway-v0", render_mode="human")
obs, info = env.reset()
```

### 2. Highway Fast (`highway-fast-v0`)

**Scenario**: Highway with CNN-based visual observations  
**Objective**: Same as highway-v0 but with image observations  
**Action Space**: Discrete  
**Observation**: Grayscale images (128x64)

### 3. Intersection (`intersection-v0`)

**Scenario**: Traffic intersection with traffic lights  
**Objective**: Navigate intersection safely to reach destination  
**Action Space**: Discrete  
**Observation**: Kinematics

```python
env = gym.make("intersection-v0", render_mode="human")
```

### 4. Roundabout (`roundabout-v0`)

**Scenario**: Roundabout navigation with traffic  
**Objective**: Navigate roundabout efficiently while yielding  
**Action Space**: Discrete  
**Observation**: Kinematics

```python
env = gym.make("roundabout-v0", render_mode="human")
```

### 5. Parking (`parking-v0`)

**Scenario**: Goal-conditioned parking maneuver  
**Objective**: Park in designated spot with correct orientation  
**Action Space**: Continuous (steering, acceleration)  
**Observation**: Kinematics

```python
env = gym.make("parking-v0", render_mode="human")
```

## 🤖 Training Approaches

### Option 1: Direct Reinforcement Learning

Train RL agents directly on environments without imitation learning.

#### DQN Training (Highway)

```bash
python3 scripts/sb3_highway_dqn.py
```

#### PPO Training (Highway)

```bash
python3 scripts/sb3_highway_ppo.py
```

#### HER Training (Parking)

```bash
python3 scripts/parking_her.py
```

### Option 2: Imitation Learning + RL (Recommended)

Use imitation learning for warm-start, then continue with RL.

#### Step 1: Collect Expert Data

```python
from imitation_learning import ExpertDataCollector

# Configure environment
env_config = {
    'observation': {'type': 'Kinematics'},
    'action': {'type': 'DiscreteMetaAction'},
    'duration': 40,
    'vehicles_count': 30,
}

# Collect expert demonstrations
collector = ExpertDataCollector(env_config=env_config)
trajectories = collector.collect_rule_based_expert_data(
    env_name='highway-v0',
    expert_type='idm',  # Intelligent Driver Model
    num_episodes=50,
    render=False
)

# Save data
collector.save_data('highway_expert_data.pkl')
```

#### Step 2: Train Imitation Learning Model

```python
from imitation_learning import ImitationLearningTrainer, create_policy_for_env
import gymnasium as gym

# Create environment and policy
env = gym.make('highway-v0')
env.unwrapped.configure(env_config)
policy = create_policy_for_env('highway-v0', env.observation_space, env.action_space)

# Train behavioral cloning
trainer = ImitationLearningTrainer(policy=policy, num_epochs=100)
train_loader, val_loader = trainer.prepare_data(trajectories)
trainer.train_behavioral_cloning(train_loader, val_loader)

# Save for RL warm-start
trainer.export_for_rl('highway_il_weights.pth')
```

#### Step 3: RL Training with Warm Start

```python
from imitation_learning.integration import create_warm_start_rl_model

# Create RL model with IL initialization
rl_model, env = create_warm_start_rl_model(
    env_name='highway-v0',
    algorithm='DQN',
    il_weights_path='highway_il_weights.pth',
    env_config=env_config
)

# Continue training with RL
rl_model.learn(total_timesteps=100000)
rl_model.save('highway_final_model')
```

## 🏃‍♂️ Complete Workflow Examples

### Highway Environment (Complete Pipeline)

```bash
# Run complete highway example
python3 example_usage.py
```

This script demonstrates:
1. Expert data collection with IDM policy
2. Imitation learning training
3. Model evaluation
4. RL warm-start training

### Individual Environment Testing

```bash
# Test specific environment
python3 test_individual_environments.py --env highway-v0

# Test all environments (quick mode)
python3 test_individual_environments.py --all --quick

# Test with full pipeline
python3 test_individual_environments.py --env intersection-v0 --full
```

## 🧪 Testing and Evaluation

### Basic Functionality Test

```bash
# Test basic installation and imports
python3 run_tests.py --basic
```

### Comprehensive Test Suite

```bash
# Run all tests
python3 run_tests.py --all

# Run specific test category
python3 run_tests.py --comprehensive
python3 run_tests.py --example
```

### Environment-Specific Testing

Each environment can be tested individually with detailed analysis:

```bash
# Highway environment
python3 test_individual_environments.py --env highway-v0 --full

# Intersection environment  
python3 test_individual_environments.py --env intersection-v0 --full

# Roundabout environment
python3 test_individual_environments.py --env roundabout-v0 --full

# Parking environment
python3 test_individual_environments.py --env parking-v0 --full
```

## 📊 Expert Types and Configurations

### Expert Types Available

| Expert Type | Description | Best For | Behavior |
|-------------|-------------|----------|----------|
| `idm` | Intelligent Driver Model | Highway, Merge | Realistic car-following |
| `planned` | Route-based planner | Intersection, Parking | Goal-directed navigation |
| `conservative` | Cautious driver | Roundabout | Safe, yield-heavy behavior |

### Environment Configurations

#### Highway Configuration

```python
highway_config = {
    'observation': {'type': 'Kinematics'},
    'action': {'type': 'DiscreteMetaAction'},
    'duration': 40,
    'vehicles_count': 30,
    'collision_reward': -1,
    'right_lane_reward': 0.1,
    'high_speed_reward': 0.4,
    'reward_speed_range': [20, 30],
}
```

#### Intersection Configuration

```python
intersection_config = {
    'observation': {'type': 'Kinematics'},
    'action': {'type': 'DiscreteMetaAction'},
    'duration': 20,
    'destination': 'o1',
    'collision_reward': -5,
    'reached_goal_reward': 1,
}
```

#### Parking Configuration

```python
parking_config = {
    'observation': {'type': 'KinematicsGoal'},
    'action': {'type': 'ContinuousAction'},
    'duration': 20,
    'success_goal_reward': 1,
    'collision_reward': -5,
}
```

## 🔧 Advanced Usage

### Custom Environment Configuration

```python
import gymnasium as gym
import highway_env

# Create environment with custom config
env = gym.make('highway-v0')
config = {
    'lanes_count': 4,
    'vehicles_count': 50,
    'duration': 60,
    'initial_spacing': 2,
    'collision_reward': -10,
}
env.unwrapped.configure(config)
```

### Custom Policy Architecture

```python
from imitation_learning.models.cnn_policy import CNNPolicy
import torch.nn as nn

# Create custom policy
class CustomPolicy(CNNPolicy):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        # Add custom layers
        self.custom_layer = nn.Linear(256, 128)
```

### Multi-Environment Training

```python
# Train on multiple environments
env_names = ['highway-v0', 'intersection-v0', 'roundabout-v0']
for env_name in env_names:
    # Collect data and train for each environment
    collector = ExpertDataCollector(env_config=configs[env_name])
    trajectories = collector.collect_rule_based_expert_data(
        env_name=env_name,
        expert_type=expert_types[env_name],
        num_episodes=30
    )
    # Train IL model...
```

## 📈 Evaluation and Benchmarking

### Model Evaluation

```python
from imitation_learning import ImitationLearningEvaluator

evaluator = ImitationLearningEvaluator(policy=trained_policy)

# Evaluate on single environment
results = evaluator.evaluate_single_environment(
    env_name='highway-v0',
    env_config=env_config,
    num_episodes=20,
    render=True
)

# Benchmark against baselines
comparison = evaluator.benchmark_against_baselines(
    env_name='highway-v0',
    env_config=env_config,
    num_episodes=10
)
```

### Performance Metrics

The framework provides comprehensive metrics:
- **Success Rate**: Percentage of successful episodes
- **Average Reward**: Mean episode reward
- **Collision Rate**: Percentage of episodes ending in collision
- **Goal Achievement**: For goal-based environments
- **Efficiency**: Speed/time metrics

## 🐛 Troubleshooting

### Common Issues

1. **Import Error**: `ModuleNotFoundError: No module named 'highway_env'`
   ```bash
   pip install -e .
   ```

2. **PyTorch Not Found**: 
   ```bash
   pip install torch torchvision
   ```

3. **Rendering Issues**:
   ```bash
   pip install pygame matplotlib
   ```

4. **Memory Issues**: Reduce `num_episodes` and `batch_size` in configurations

### Debug Mode

```bash
# Run with debug output
python3 run_tests.py --basic --verbose

# Test individual components
python3 -c "from imitation_learning import ExpertDataCollector; print('✅ IL components loaded')"
```

## 📚 Documentation

- **Main Documentation**: [highway-env.farama.org](https://highway-env.farama.org/)
- **API Reference**: See individual module docstrings
- **Examples**: `scripts/` and `example_usage.py`
- **Testing Guide**: `TESTING_INSTRUCTIONS.md`
- **IL Framework**: `IMITATION_LEARNING_README.md`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test: `python3 run_tests.py --all`
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Citation

If you use this project in your research, please cite:

```bibtex
@misc{highway-env,
  author = {Leurent, Edouard},
  title = {An Environment for Autonomous Driving Decision-Making},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/eleurent/highway-env}},
}
```

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/eleurent/highway-env/issues)
- **Discussions**: [GitHub Discussions](https://github.com/eleurent/highway-env/discussions)
- **Documentation**: [Official Docs](https://highway-env.farama.org/)

---

**Quick Start Commands Summary**:

```bash
# Installation
git clone https://github.com/eleurent/highway-env.git
cd highway-env
pip install -e .
pip install torch stable-baselines3[extra]

# Basic test
python3 run_tests.py --basic

# Complete example
python3 example_usage.py

# Individual environment test
python3 test_individual_environments.py --env highway-v0 --quick

# RL training
python3 scripts/sb3_highway_dqn.py
```