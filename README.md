# Lane-Nav: Autonomous Driving RL/IL Framework

A modular framework for training and evaluating autonomous driving agents using both Reinforcement Learning (RL) and Imitation Learning (IL) across multiple highway-env scenarios.

## 🚗 Supported Scenarios

- **Highway** (`highway-v0`): Multi-lane highway driving with lane changes and overtaking
- **Intersection** (`intersection-v0`): Urban intersection navigation with traffic lights
- **Roundabout** (`roundabout-v0`): Roundabout entry, navigation, and exit
- **Parking** (`parking-v0`): Precise parking maneuvers and goal-reaching

## 🧠 Supported Algorithms

### Reinforcement Learning
- **PPO** (Proximal Policy Optimization) - Main RL algorithm

### Imitation Learning  
- **BC** (Behavioral Cloning) - Learn from expert demonstrations
- **DAgger** (Dataset Aggregation) - Interactive imitation learning

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd lane-nav
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Create necessary directories:**
```bash
mkdir -p models logs tensorboard_logs
```

## 🚀 Quick Start

### Training Agents

Train a PPO agent for highway driving:
```bash
python train_highway.py --mode rl
```

Train a Behavioral Cloning agent for intersection:
```bash
python train_intersection.py --mode il --algorithm bc
```

Train with custom parameters:
```bash
python train_parking.py --mode rl --timesteps 200000 --save-path custom_model.zip
```

### Evaluating Agents

Evaluate a trained RL agent:
```bash
python evaluate.py highway --mode rl
```

Evaluate an IL agent with multiple episodes:
```bash
python evaluate.py roundabout --mode il --episodes 5
```

Evaluate without rendering (headless):
```bash
python evaluate.py parking --mode rl --no-render
```

## 📁 Project Structure

```
lane-nav/
├── config.py              # Shared configurations and hyperparameters
├── requirements.txt       # Python dependencies
├── imitation.py           # Imitation learning utilities
├── train_highway.py       # Highway scenario training
├── train_intersection.py  # Intersection scenario training  
├── train_roundabout.py    # Roundabout scenario training
├── train_parking.py       # Parking scenario training
├── evaluate.py            # Universal evaluation script
├── models/                # Saved model files
├── logs/                  # Training logs
└── tensorboard_logs/      # TensorBoard logs
```

## 🎯 Usage Examples

### Complete Training Pipeline

1. **Train an RL expert:**
```bash
python train_highway.py --mode rl --timesteps 100000
```

2. **Use expert to train IL agent:**
```bash
python train_highway.py --mode il --algorithm bc --n-demonstrations 50
```

3. **Evaluate both agents:**
```bash
python evaluate.py highway --mode rl --episodes 10
python evaluate.py highway --mode il --episodes 10
```

### Advanced Training Options

**Custom hyperparameters** (modify `config.py`):
- Adjust `PPO_HYPERPARAMS` for RL training
- Modify `IMITATION_HYPERPARAMS` for IL training
- Change environment configurations in `ENVIRONMENT_CONFIGS`

**DAgger training:**
```bash
python train_intersection.py --mode il --algorithm dagger
```

**Custom expert model:**
```bash
python train_parking.py --mode il --expert-model ./models/custom_expert.zip
```

### Evaluation Options

**Save evaluation results:**
```bash
python evaluate.py highway --mode rl --save-results results.json
```

**Quiet evaluation:**
```bash
python evaluate.py roundabout --mode il --quiet --no-render --episodes 100
```

**Stochastic policy evaluation:**
```bash
python evaluate.py intersection --mode rl --non-deterministic
```

## 📊 Configuration

### Environment Settings
Modify scenario-specific settings in `config.py`:
- Observation types and features
- Action spaces (discrete/continuous)
- Reward functions and weights
- Simulation parameters

### Training Hyperparameters
- **PPO**: Learning rate, batch size, epochs, etc.
- **BC/DAgger**: IL-specific parameters
- **General**: Save frequency, evaluation intervals

### Model Files
Models are automatically saved with descriptive names:
- `ppo_highway_agent.zip` - PPO model for highway
- `bc_intersection_agent.zip` - BC model for intersection
- `dagger_roundabout_agent.zip` - DAgger model for roundabout

## 🔧 Troubleshooting

### Common Issues

**1. Module not found errors:**
```bash
pip install -r requirements.txt
```

**2. OpenGL rendering issues:**
```bash
# For headless systems, use --no-render flag
python evaluate.py highway --mode rl --no-render
```

**3. Model not found:**
```bash
# Train the model first
python train_highway.py --mode rl
# Then evaluate
python evaluate.py highway --mode rl
```

**4. GPU/CUDA issues:**
```bash
# Force CPU usage by modifying config.py:
# Set device="cpu" in hyperparameters
```

### Performance Tips

1. **Use multiple CPU cores:**
   - Modify `n_envs` in vectorized environments
   - Adjust `n_steps` for PPO batch collection

2. **Monitor training:**
   ```bash
   tensorboard --logdir ./tensorboard_logs
   ```

3. **Resume training:**
   ```bash
   # Load existing model and continue training
   python train_highway.py --mode rl --load-model ./models/checkpoint.zip
   ```

## 🎮 Environment Details

### Highway-v0
- **Objective**: Drive on highway, avoid collisions, maintain speed
- **Observation**: Kinematics of nearby vehicles
- **Action**: Discrete meta-actions (lane changes, acceleration)
- **Success**: High speed, no collisions, lane discipline

### Intersection-v0  
- **Objective**: Navigate intersection, reach destination
- **Observation**: Vehicle positions and traffic state
- **Action**: Discrete driving commands
- **Success**: Reach destination without collision

### Roundabout-v0
- **Objective**: Enter, navigate, and exit roundabout
- **Observation**: Vehicle kinematics in roundabout
- **Action**: Discrete meta-actions
- **Success**: Complete roundabout navigation safely

### Parking-v0
- **Objective**: Park vehicle in designated spot
- **Observation**: KinematicsGoal (position relative to target)
- **Action**: Continuous steering and acceleration
- **Success**: Precise positioning in parking spot

## 📈 Monitoring and Logging

### TensorBoard Integration
```bash
# View training progress
tensorboard --logdir ./tensorboard_logs
```

### Log Files
- Training logs: `./logs/`
- Model checkpoints: `./models/`
- Evaluation results: Save with `--save-results`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [highway-env](https://github.com/eleurent/highway-env) for the driving environments
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL algorithms
- [imitation](https://github.com/HumanCompatibleAI/imitation) for IL algorithms