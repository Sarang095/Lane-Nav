# Step-by-Step Terminal Commands for Highway-Env Project

Complete guide with terminal commands to install, run, and test the highway-env autonomous driving simulation project.

## 🚀 **Phase 1: Installation and Setup**

### Step 1: Clone the Repository
```bash
# Clone the project
git clone https://github.com/eleurent/highway-env.git
cd highway-env
```

### Step 2: Install Python Dependencies
```bash
# Install the highway-env package
pip install -e .

# Install additional ML dependencies
pip install torch torchvision stable-baselines3[extra] opencv-python scikit-learn matplotlib pandas scipy pygame

# Verify installation
python3 -c "import highway_env; import gymnasium as gym; env = gym.make('highway-v0'); print('✅ Installation successful')"
```

## 🧪 **Phase 2: Quick Testing**

### Step 3: Run Basic Tests
```bash
# Test basic functionality
python3 run_tests.py --basic

# Run example workflow
python3 run_tests.py --example
```

### Step 4: View All Available Environments
```bash
# Interactive demo of all environments
python3 run_simulation_demo.py
```

## 🎬 **Phase 3: View Simulations (Visual Demos)**

### Step 5: Test Individual Environments

#### Highway Environment
```bash
# Random policy demo
python3 run_simulation_demo.py --env highway-v0 --episodes 3

# With video recording
python3 run_simulation_demo.py --env highway-v0 --episodes 3 --record
```

#### Intersection Environment
```bash
python3 run_simulation_demo.py --env intersection-v0 --episodes 3
```

#### Roundabout Environment
```bash
python3 run_simulation_demo.py --env roundabout-v0 --episodes 3
```

#### Parking Environment
```bash
python3 run_simulation_demo.py --env parking-v0 --episodes 3
```

#### Merge Environment
```bash
python3 run_simulation_demo.py --env merge-v0 --episodes 3
```

## 🤖 **Phase 4: Train Models with Direct RL**

### Step 6: Train DQN on Highway
```bash
# Train DQN model (includes final simulation)
python3 scripts/sb3_highway_dqn.py
```

### Step 7: Train PPO on Highway
```bash
# Train PPO model (includes final simulation)
python3 scripts/sb3_highway_ppo.py
```

### Step 8: Train HER on Parking
```bash
# Train HER model for parking (includes final simulation)
python3 scripts/parking_her.py
```

## 🎓 **Phase 5: Imitation Learning + RL Pipeline**

### Step 9: Complete Imitation Learning Example
```bash
# Run complete IL + RL workflow
python3 example_usage.py
```

### Step 10: Test Individual Environments with IL
```bash
# Test highway with imitation learning
python3 test_individual_environments.py --env highway-v0 --quick

# Test with visualization
python3 test_individual_environments.py --env highway-v0 --render

# Test intersection environment
python3 test_individual_environments.py --env intersection-v0 --quick

# Test parking environment
python3 test_individual_environments.py --env parking-v0 --quick

# Test all environments (quick mode)
python3 test_individual_environments.py --all --quick
```

## 🔬 **Phase 6: Comprehensive Testing**

### Step 11: Run Full Test Suite
```bash
# Run all tests
python3 run_tests.py --all

# Run comprehensive tests
python3 run_tests.py --comprehensive
```

### Step 12: Full Pipeline Testing
```bash
# Test complete pipeline for highway
python3 test_individual_environments.py --env highway-v0 --full

# Test complete pipeline for intersection
python3 test_individual_environments.py --env intersection-v0 --full

# Test complete pipeline for parking
python3 test_individual_environments.py --env parking-v0 --full
```

## 📊 **Phase 7: Custom Training Examples**

### Step 13: Custom Environment Configuration
```bash
# Create a custom training script
cat > custom_highway_training.py << 'EOF'
import gymnasium as gym
import highway_env
from stable_baselines3 import DQN

# Create environment with custom configuration
env = gym.make("highway-v0", render_mode="human")
env.unwrapped.configure({
    'observation': {'type': 'Kinematics'},
    'action': {'type': 'DiscreteMetaAction'},
    'lanes_count': 4,
    'vehicles_count': 30,
    'duration': 40,
    'collision_reward': -1,
    'right_lane_reward': 0.1,
    'high_speed_reward': 0.4,
})

# Train DQN model
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save model
model.save("custom_highway_model")

# Test trained model
for episode in range(3):
    obs, info = env.reset()
    done = truncated = False
    episode_reward = 0
    
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        env.render()
    
    print(f"Episode {episode + 1} reward: {episode_reward}")

env.close()
EOF

# Run custom training
python3 custom_highway_training.py
```

### Step 14: Advanced Imitation Learning
```bash
# Create advanced IL example
cat > advanced_il_example.py << 'EOF'
import gymnasium as gym
import highway_env
from imitation_learning import ExpertDataCollector, ImitationLearningTrainer, create_policy_for_env

# Environment configuration
env_config = {
    'observation': {'type': 'Kinematics'},
    'action': {'type': 'DiscreteMetaAction'},
    'duration': 40,
    'vehicles_count': 30,
}

# Step 1: Collect expert data
print("Collecting expert data...")
collector = ExpertDataCollector(env_config=env_config)
trajectories = collector.collect_rule_based_expert_data(
    env_name='highway-v0',
    expert_type='idm',
    num_episodes=20,
    render=False
)

# Step 2: Train imitation learning model
print("Training imitation learning model...")
env = gym.make('highway-v0')
env.unwrapped.configure(env_config)
policy = create_policy_for_env('highway-v0', env.observation_space, env.action_space)
trainer = ImitationLearningTrainer(policy=policy, num_epochs=50)

train_loader, val_loader = trainer.prepare_data(trajectories)
trainer.train_behavioral_cloning(train_loader, val_loader)

# Step 3: Test the model
print("Testing trained model...")
env = gym.make('highway-v0', render_mode='human')
env.unwrapped.configure(env_config)

for episode in range(3):
    obs, info = env.reset()
    done = truncated = False
    episode_reward = 0
    
    while not (done or truncated):
        action = trainer.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        env.render()
    
    print(f"Episode {episode + 1} reward: {episode_reward}")

env.close()
print("Advanced IL example completed!")
EOF

# Run advanced IL example
python3 advanced_il_example.py
```

## 📹 **Phase 8: Video Recording**

### Step 15: Record Training Videos
```bash
# Create video recording script
cat > record_videos.py << 'EOF'
import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
import os

# Create videos directory
os.makedirs("training_videos", exist_ok=True)

environments = ['highway-v0', 'intersection-v0', 'roundabout-v0', 'parking-v0']

for env_name in environments:
    print(f"Recording {env_name}...")
    
    # Create environment with video recording
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env, 
        video_folder=f"training_videos/{env_name}", 
        episode_trigger=lambda e: True
    )
    
    # Record 3 episodes with random policy
    for episode in range(3):
        obs, info = env.reset()
        done = truncated = False
        
        while not (done or truncated):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
    
    env.close()
    print(f"✅ {env_name} videos saved")

print("All videos recorded in training_videos/ directory")
EOF

# Run video recording
python3 record_videos.py
```

## 🔍 **Phase 9: Results and Analysis**

### Step 16: Check Results
```bash
# List generated files
echo "=== Generated Models ==="
find . -name "*.pth" -o -name "*.pkl" -o -name "*.zip" | head -10

echo "=== Generated Videos ==="
find . -name "*.mp4" | head -10

echo "=== Training Logs ==="
find . -name "*log*" -o -name "*tensorboard*" | head -10

# Check results directory
ls -la individual_test_results/ 2>/dev/null || echo "No test results yet"
ls -la training_videos/ 2>/dev/null || echo "No videos yet"
```

## 🚨 **Phase 10: Troubleshooting Commands**

### Step 17: Debug Issues
```bash
# Check Python environment
python3 --version
pip list | grep -E "(highway|gym|torch|stable)"

# Test individual components
python3 -c "import highway_env; print('✅ highway_env works')"
python3 -c "import gymnasium; print('✅ gymnasium works')"
python3 -c "import torch; print('✅ torch works')"
python3 -c "from stable_baselines3 import DQN; print('✅ stable-baselines3 works')"

# Check GPU availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test basic environment creation
python3 -c "
import gymnasium as gym
import highway_env
for env_name in ['highway-v0', 'intersection-v0', 'parking-v0']:
    try:
        env = gym.make(env_name)
        print(f'✅ {env_name} works')
        env.close()
    except Exception as e:
        print(f'❌ {env_name} failed: {e}')
"
```

## 📋 **Quick Command Reference**

### Essential Commands
```bash
# 1. Install everything
pip install -e . && pip install torch stable-baselines3[extra] opencv-python

# 2. Quick test
python3 run_tests.py --basic

# 3. Demo all environments
python3 run_simulation_demo.py

# 4. Train and test highway
python3 scripts/sb3_highway_dqn.py

# 5. Complete IL workflow
python3 example_usage.py

# 6. Test specific environment
python3 test_individual_environments.py --env highway-v0 --quick
```

### Environment-Specific Quick Tests
```bash
# Highway
python3 run_simulation_demo.py --env highway-v0

# Intersection  
python3 run_simulation_demo.py --env intersection-v0

# Roundabout
python3 run_simulation_demo.py --env roundabout-v0

# Parking
python3 run_simulation_demo.py --env parking-v0

# All environments
python3 test_individual_environments.py --all --quick
```

## 🎯 **Expected Outputs**

After running these commands, you should see:

1. **✅ Installation Success Messages**
2. **🎬 Visual Simulation Windows** showing cars driving
3. **📊 Training Progress** with loss/reward graphs
4. **📹 Video Files** in various directories
5. **📈 Performance Metrics** in terminal output
6. **🤖 Trained Models** saved as .pth/.zip files

## 💡 **Pro Tips**

```bash
# Run in background with logging
nohup python3 scripts/sb3_highway_dqn.py > highway_training.log 2>&1 &

# Monitor training progress
tail -f highway_training.log

# Quick visual test of all environments
for env in highway-v0 intersection-v0 roundabout-v0 parking-v0; do
    python3 run_simulation_demo.py --env $env --episodes 1
done

# Check system resources
htop  # or top
nvidia-smi  # if using GPU
```

Copy and paste these commands in order to get the complete highway-env experience! 🚗🎯