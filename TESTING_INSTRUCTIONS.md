# Testing Instructions for Highway-Env Imitation Learning Framework

This document provides comprehensive testing instructions for the imitation learning framework integrated with highway-env.

## 🚀 Quick Start Testing

### 1. Basic Functionality Test
```bash
# Test basic framework functionality
python3 run_tests.py --basic
```

### 2. Example Demonstration
```bash
# Run a complete example workflow
python3 run_tests.py --example
```

### 3. Individual Environment Testing
```bash
# Test specific environment
python3 test_individual_environments.py --env highway-v0 --quick

# Test all environments individually
python3 test_individual_environments.py --all --quick
```

### 4. Comprehensive Test Suite
```bash
# Run all tests (basic + environments + comprehensive)
python3 run_tests.py --all
```

## 🧪 Test Categories

### A. Basic Functionality Tests (`--basic`)
- **Highway-env basic functionality**: Environment creation and basic operations
- **Imitation learning models**: Policy creation and forward pass
- **Data collection functionality**: Expert data collection pipeline

### B. Individual Environment Tests (`--env`)
Tests each environment through the complete pipeline:
- Environment setup and validation
- Expert data collection
- Model training
- Model evaluation
- RL integration

**Supported Environments:**
- `highway-v0`: Multi-lane highway with kinematics
- `highway-fast-v0`: Highway with CNN observations
- `intersection-v0`: Traffic intersection navigation
- `roundabout-v0`: Roundabout navigation
- `parking-v0`: Parking maneuver (continuous control)

### C. Comprehensive Test Suite (`--comprehensive`)
- Full framework integration test
- Multi-environment testing
- End-to-end pipeline validation

### D. Example Demonstration (`--example`)
- Complete workflow demonstration
- Minimal working example
- Integration verification

## 🎯 Environment-Specific Testing

### Highway Environment (Kinematics)
```bash
python3 test_individual_environments.py --env highway-v0
```
**Tests:**
- Expert data collection with IDM policy
- MLP-based imitation learning
- DQN integration with warm start

### Highway Environment (CNN)
```bash
python3 test_individual_environments.py --env highway-fast-v0
```
**Tests:**
- CNN-based observations
- Visual feature extraction
- Image-based policy training

### Intersection Environment
```bash
python3 test_individual_environments.py --env intersection-v0
```
**Tests:**
- Goal-directed navigation
- Traffic light interaction
- Planned expert behavior

### Roundabout Environment
```bash
python3 test_individual_environments.py --env roundabout-v0
```
**Tests:**
- Yield behavior
- Circular navigation
- Conservative driving policy

### Parking Environment
```bash
python3 test_individual_environments.py --env parking-v0
```
**Tests:**
- Continuous action space
- Precision control
- Goal-conditioned tasks

## 🔧 Test Configuration Options

### Quick Tests
```bash
# Reduced episodes and epochs for faster testing
python3 test_individual_environments.py --all --quick
python3 run_tests.py --all
```

### Rendering Tests
```bash
# Enable visualization during testing
python3 test_individual_environments.py --env highway-v0 --render
```

### Output Directory
```bash
# Specify custom output directory
python3 test_individual_environments.py --all --output-dir ./my_test_results
```

## 📊 Test Results and Interpretation

### Success Criteria
- **Environment Creation**: ✅ Environment can be created and configured
- **Data Collection**: ✅ Expert trajectories collected successfully
- **Model Training**: ✅ Training converges without errors
- **Model Evaluation**: ✅ Model performs better than random baseline
- **RL Integration**: ✅ Weights successfully loaded into RL framework

### Expected Performance Metrics
| Environment | Mean Reward | Success Rate | Collision Rate |
|-------------|-------------|--------------|----------------|
| Highway-v0 | > 15.0 | > 0.7 | < 0.3 |
| Highway-fast-v0 | > 12.0 | > 0.6 | < 0.4 |
| Intersection-v0 | > 0.5 | > 0.8 | < 0.2 |
| Roundabout-v0 | > 0.3 | > 0.7 | < 0.3 |
| Parking-v0 | > 0.2 | > 0.6 | < 0.4 |

### Test Output Files
```
test_results/
├── comprehensive_test_results.json     # Complete test results
├── individual_test_results/            # Per-environment results
│   ├── highway-v0/
│   │   ├── test_results.json
│   │   ├── expert_data/
│   │   ├── trained_models/
│   │   └── highway-v0_training_plots.png
│   └── ...
└── all_environments_results.json      # Summary of all environments
```

## 🐛 Troubleshooting Common Issues

### 1. Import Errors
```bash
# Ensure highway-env is installed
pip install -e .

# Install additional dependencies
pip install torch torchvision stable-baselines3[extra]
```

### 2. CUDA/Memory Issues
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""

# Or edit test files to use smaller batch sizes
```

### 3. Environment Creation Failures
- Check highway-env version compatibility
- Verify gymnasium version (should be >= 1.0.0)
- Ensure all dependencies are installed

### 4. Data Collection Issues
- Lower quality thresholds for testing
- Reduce episode requirements
- Check expert policy implementations

### 5. Training Failures
- Reduce batch size and learning rate
- Use shorter training epochs for testing
- Check observation/action space compatibility

## 📝 Custom Testing

### Creating Custom Tests
```python
# Example custom test
from imitation_learning import ExpertDataCollector, create_policy_for_env
import gymnasium as gym
import highway_env

def test_custom_scenario():
    # Your custom test configuration
    env_config = {
        'observation': {'type': 'Kinematics'},
        'duration': 20,
        'vehicles_count': 15,
    }
    
    # Run your test
    collector = ExpertDataCollector(env_config=env_config)
    trajectories = collector.collect_rule_based_expert_data(
        env_name='highway-v0',
        expert_type='idm',
        render=False,
    )
    
    assert len(trajectories) > 0, "No trajectories collected"
    print(f"✅ Custom test passed - {len(trajectories)} trajectories")

if __name__ == "__main__":
    test_custom_scenario()
```

### Adding New Environments
1. Add environment configuration to test files
2. Implement expert policy for the environment
3. Add test cases to the test suite
4. Update expected performance metrics

## 🎯 Continuous Integration

### Automated Testing
```bash
#!/bin/bash
# CI test script
set -e

echo "Running Highway-Env Imitation Learning Tests"

# Basic functionality
python3 run_tests.py --basic

# Core environments
python3 test_individual_environments.py --env highway-v0 --quick
python3 test_individual_environments.py --env intersection-v0 --quick

# Example demonstration
python3 run_tests.py --example

echo "All CI tests passed!"
```

### Performance Monitoring
- Track training convergence speed
- Monitor evaluation metrics over time
- Compare with baseline performance

## 📞 Support and Debugging

### Debug Mode
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed output
python3 test_individual_environments.py --env highway-v0 --debug
```

### Common Solutions
1. **Test Failures**: Check test logs for specific error messages
2. **Performance Issues**: Use quick mode for faster iteration
3. **Memory Issues**: Reduce batch sizes and use CPU
4. **Version Conflicts**: Use compatible package versions

### Getting Help
1. Check the troubleshooting section in README
2. Review test output logs
3. Verify environment configurations
4. Test individual components in isolation

## ✅ Verification Checklist

Before deployment, ensure all tests pass:

- [ ] Basic functionality tests
- [ ] Individual environment tests
- [ ] Example demonstration
- [ ] Comprehensive test suite
- [ ] Performance metrics meet thresholds
- [ ] No critical errors in logs
- [ ] All environments properly configured
- [ ] RL integration working correctly

---

**Note**: All tests are designed to be non-interactive and can be run in automated environments. For visual verification, use the `--render` flag on individual environment tests.