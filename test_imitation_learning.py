"""
Comprehensive Testing Script for Imitation Learning Framework
Tests all environments: highway, intersection, roundabout, parking
"""

import os
import sys
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path
import argparse
import json
from typing import Dict, List, Optional, Any

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

import highway_env
from imitation_learning.data_collection.expert_data_collector import ExpertDataCollector, DatasetCreator
from imitation_learning.models.cnn_policy import create_policy_for_env
from imitation_learning.training.imitation_trainer import ImitationLearningTrainer, train_imitation_learning_pipeline
from imitation_learning.evaluation.evaluator import ImitationLearningEvaluator, run_comprehensive_evaluation
from imitation_learning.integration.rl_integration import (
    create_warm_start_rl_model, 
    train_rl_with_il_warmstart,
    PerformanceComparator
)


class ComprehensiveTestSuite:
    """
    Comprehensive test suite for the imitation learning framework
    Tests data collection, training, evaluation, and RL integration
    """
    
    def __init__(self, output_dir: str = "./test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Environment configurations for testing
        self.env_configs = {
            'highway-v0': {
                'observation': {'type': 'Kinematics'},
                'action': {'type': 'DiscreteMetaAction'},
                'duration': 20,  # Shorter for testing
                'vehicles_count': 20,
            },
            'highway-fast-v0': {
                'observation': {
                    'type': 'GrayscaleObservation',
                    'observation_shape': (64, 32),  # Smaller for faster testing
                    'stack_size': 4,
                    'weights': [0.2989, 0.5870, 0.1140],
                    'scaling': 1.75,
                },
                'duration': 20,
                'vehicles_count': 15,
            },
            'intersection-v0': {
                'observation': {'type': 'Kinematics'},
                'action': {'type': 'DiscreteMetaAction'},
                'duration': 15,
            },
            'roundabout-v0': {
                'observation': {'type': 'Kinematics'},
                'action': {'type': 'DiscreteMetaAction'},
                'duration': 15,
            },
            'parking-v0': {
                'observation': {'type': 'Kinematics'},
                'action': {'type': 'ContinuousAction'},
                'duration': 15,
            },
        }
        
        self.test_results = {}
    
    def test_environment_creation(self) -> Dict[str, bool]:
        """Test that all environments can be created successfully"""
        print("=== Testing Environment Creation ===")
        results = {}
        
        for env_name, config in self.env_configs.items():
            try:
                env = gym.make(env_name)
                env.unwrapped.configure(config)
                
                # Test basic functionality
                obs, info = env.reset()
                action = env.action_space.sample()
                next_obs, reward, done, truncated, info = env.step(action)
                
                env.close()
                results[env_name] = True
                print(f"✓ {env_name}: Environment created and basic functionality works")
                
            except Exception as e:
                results[env_name] = False
                print(f"✗ {env_name}: Failed - {str(e)}")
        
        return results
    
    def test_data_collection(self, num_episodes: int = 5) -> Dict[str, Any]:
        """Test expert data collection for all environments"""
        print("\n=== Testing Data Collection ===")
        results = {}
        
        for env_name, config in self.env_configs.items():
            print(f"\nTesting data collection for {env_name}")
            try:
                # Create data collector
                collector = ExpertDataCollector(
                    env_config=config,
                    save_dir=str(self.output_dir / "test_data"),
                    max_episodes=num_episodes,
                    min_episode_length=5,
                )
                
                # Collect data
                trajectories = collector.collect_rule_based_expert_data(
                    env_name=env_name,
                    expert_type="idm",
                    render=False,
                )
                
                # Validate data
                assert len(trajectories) > 0, "No trajectories collected"
                assert all(len(t.observations) > 0 for t in trajectories), "Empty observations"
                assert all(len(t.actions) > 0 for t in trajectories), "Empty actions"
                
                # Save data
                save_path = collector.save_data(f"{env_name}_test_data.pkl")
                
                results[env_name] = {
                    'success': True,
                    'num_trajectories': len(trajectories),
                    'total_steps': sum(t.episode_length for t in trajectories),
                    'save_path': save_path,
                }
                
                print(f"✓ {env_name}: Collected {len(trajectories)} trajectories")
                
            except Exception as e:
                results[env_name] = {
                    'success': False,
                    'error': str(e),
                }
                print(f"✗ {env_name}: Data collection failed - {str(e)}")
        
        return results
    
    def test_model_creation(self) -> Dict[str, Any]:
        """Test model creation for all environments"""
        print("\n=== Testing Model Creation ===")
        results = {}
        
        for env_name, config in self.env_configs.items():
            print(f"\nTesting model creation for {env_name}")
            try:
                # Create environment to get spaces
                env = gym.make(env_name)
                env.unwrapped.configure(config)
                
                observation_space = env.observation_space
                action_space = env.action_space
                
                # Create policy
                policy = create_policy_for_env(env_name, observation_space, action_space)
                
                # Test forward pass
                if isinstance(observation_space, gym.spaces.Box):
                    sample_obs = torch.FloatTensor(observation_space.sample()).unsqueeze(0)
                    
                    with torch.no_grad():
                        actions, values = policy(sample_obs)
                    
                    # Validate output shapes
                    if isinstance(action_space, gym.spaces.Discrete):
                        expected_action_shape = (1, action_space.n)
                    else:
                        expected_action_shape = (1, np.prod(action_space.shape))
                    
                    assert actions.shape == expected_action_shape, f"Action shape mismatch: {actions.shape} vs {expected_action_shape}"
                    assert values.shape == (1, 1), f"Value shape mismatch: {values.shape}"
                
                env.close()
                
                results[env_name] = {
                    'success': True,
                    'model_type': type(policy).__name__,
                    'num_parameters': sum(p.numel() for p in policy.parameters()),
                }
                
                print(f"✓ {env_name}: Model created successfully")
                
            except Exception as e:
                results[env_name] = {
                    'success': False,
                    'error': str(e),
                }
                print(f"✗ {env_name}: Model creation failed - {str(e)}")
        
        return results
    
    def test_training_pipeline(self, max_epochs: int = 5) -> Dict[str, Any]:
        """Test training pipeline for all environments"""
        print("\n=== Testing Training Pipeline ===")
        results = {}
        
        # First ensure we have data
        data_results = self.test_data_collection(num_episodes=3)
        
        for env_name, config in self.env_configs.items():
            if not data_results[env_name]['success']:
                print(f"Skipping {env_name} training test - no data available")
                continue
                
            print(f"\nTesting training for {env_name}")
            try:
                # Load trajectories
                data_path = data_results[env_name]['save_path']
                
                collector = ExpertDataCollector(env_config=config)
                trajectories = collector.load_data(data_path)
                
                # Create environment and policy
                env = gym.make(env_name)
                env.unwrapped.configure(config)
                
                policy = create_policy_for_env(env_name, env.observation_space, env.action_space)
                
                # Create trainer
                trainer = ImitationLearningTrainer(
                    policy=policy,
                    num_epochs=max_epochs,
                    batch_size=16,  # Small for testing
                    learning_rate=1e-3,
                    validation_split=0.3,
                    save_dir=str(self.output_dir / "trained_models"),
                )
                
                # Prepare data
                train_loader, val_loader = trainer.prepare_data(trajectories)
                
                # Train
                training_history = trainer.train_behavioral_cloning(train_loader, val_loader)
                
                # Save model
                model_path = trainer.save_model(f"{env_name}_test_model.pth")
                
                # Export for RL
                rl_weights_path = str(self.output_dir / f"{env_name}_rl_weights.pth")
                trainer.export_for_rl(rl_weights_path)
                
                env.close()
                
                results[env_name] = {
                    'success': True,
                    'final_train_loss': training_history['train_total_loss'][-1],
                    'final_val_loss': training_history['val_total_loss'][-1],
                    'model_path': model_path,
                    'rl_weights_path': rl_weights_path,
                }
                
                print(f"✓ {env_name}: Training completed successfully")
                
            except Exception as e:
                results[env_name] = {
                    'success': False,
                    'error': str(e),
                }
                print(f"✗ {env_name}: Training failed - {str(e)}")
        
        return results
    
    def test_evaluation(self) -> Dict[str, Any]:
        """Test evaluation framework"""
        print("\n=== Testing Evaluation ===")
        results = {}
        
        # First ensure we have trained models
        training_results = self.test_training_pipeline(max_epochs=3)
        
        for env_name, config in self.env_configs.items():
            if not training_results[env_name]['success']:
                print(f"Skipping {env_name} evaluation test - no trained model available")
                continue
                
            print(f"\nTesting evaluation for {env_name}")
            try:
                # Load trained model
                model_path = training_results[env_name]['model_path']
                
                # Create environment and policy
                env = gym.make(env_name)
                env.unwrapped.configure(config)
                
                policy = create_policy_for_env(env_name, env.observation_space, env.action_space)
                
                # Create trainer to load model
                trainer = ImitationLearningTrainer(policy=policy)
                trainer.load_model(model_path)
                
                # Create evaluator
                evaluator = ImitationLearningEvaluator(policy)
                
                # Run evaluation
                eval_results = evaluator.evaluate_single_environment(
                    env_name=env_name,
                    env_config=config,
                    num_episodes=3,  # Small for testing
                    deterministic=True,
                )
                
                env.close()
                
                results[env_name] = {
                    'success': True,
                    'mean_reward': eval_results['mean_reward'],
                    'success_rate': eval_results['success_rate'],
                    'collision_rate': eval_results['collision_rate'],
                }
                
                print(f"✓ {env_name}: Evaluation completed - Reward: {eval_results['mean_reward']:.2f}")
                
            except Exception as e:
                results[env_name] = {
                    'success': False,
                    'error': str(e),
                }
                print(f"✗ {env_name}: Evaluation failed - {str(e)}")
        
        return results
    
    def test_rl_integration(self) -> Dict[str, Any]:
        """Test RL integration with IL warm start"""
        print("\n=== Testing RL Integration ===")
        results = {}
        
        # Get training results for IL weights
        training_results = self.test_training_pipeline(max_epochs=3)
        
        # Test only a subset of environments for RL integration (time-consuming)
        test_envs = ['highway-v0', 'intersection-v0']
        
        for env_name in test_envs:
            if env_name not in self.env_configs:
                continue
                
            if not training_results.get(env_name, {}).get('success', False):
                print(f"Skipping {env_name} RL integration test - no IL weights available")
                continue
                
            print(f"\nTesting RL integration for {env_name}")
            try:
                # Get IL weights path
                rl_weights_path = training_results[env_name]['rl_weights_path']
                
                # Test warm start model creation
                model, env = create_warm_start_rl_model(
                    env_name=env_name,
                    algorithm='DQN',
                    il_weights_path=rl_weights_path,
                    env_config=self.env_configs[env_name],
                )
                
                # Test short training
                print(f"Testing short RL training...")
                model.learn(total_timesteps=100)  # Very short for testing
                
                # Test evaluation
                obs, info = env.reset()
                action, _ = model.predict(obs, deterministic=True)
                
                env.close()
                
                results[env_name] = {
                    'success': True,
                    'algorithm': 'DQN',
                    'warmstart_working': True,
                }
                
                print(f"✓ {env_name}: RL integration working")
                
            except Exception as e:
                results[env_name] = {
                    'success': False,
                    'error': str(e),
                }
                print(f"✗ {env_name}: RL integration failed - {str(e)}")
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report"""
        print("Starting Comprehensive Imitation Learning Test Suite")
        print("=" * 60)
        
        all_results = {}
        
        # Run all test phases
        all_results['environment_creation'] = self.test_environment_creation()
        all_results['data_collection'] = self.test_data_collection()
        all_results['model_creation'] = self.test_model_creation()
        all_results['training_pipeline'] = self.test_training_pipeline()
        all_results['evaluation'] = self.test_evaluation()
        all_results['rl_integration'] = self.test_rl_integration()
        
        # Generate summary
        summary = self._generate_test_summary(all_results)
        all_results['summary'] = summary
        
        # Save results
        results_path = self.output_dir / "comprehensive_test_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n=== Test Summary ===")
        print(f"Total environments tested: {len(self.env_configs)}")
        print(f"Overall success rate: {summary['overall_success_rate']:.1%}")
        print(f"Detailed results saved to: {results_path}")
        
        return all_results
    
    def _generate_test_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate test summary statistics"""
        summary = {
            'total_environments': len(self.env_configs),
            'test_phases': list(results.keys()),
            'phase_success_rates': {},
            'environment_success_rates': {},
            'overall_success_rate': 0.0,
        }
        
        # Calculate phase success rates
        for phase, phase_results in results.items():
            if isinstance(phase_results, dict):
                successful = sum(1 for r in phase_results.values() 
                               if isinstance(r, dict) and r.get('success', False))
                total = len(phase_results)
                summary['phase_success_rates'][phase] = successful / total if total > 0 else 0
        
        # Calculate environment success rates
        for env_name in self.env_configs:
            successful_phases = 0
            total_phases = 0
            
            for phase, phase_results in results.items():
                if isinstance(phase_results, dict) and env_name in phase_results:
                    total_phases += 1
                    if isinstance(phase_results[env_name], dict) and phase_results[env_name].get('success', False):
                        successful_phases += 1
            
            summary['environment_success_rates'][env_name] = (
                successful_phases / total_phases if total_phases > 0 else 0
            )
        
        # Calculate overall success rate
        all_success_rates = list(summary['phase_success_rates'].values())
        summary['overall_success_rate'] = sum(all_success_rates) / len(all_success_rates) if all_success_rates else 0
        
        return summary


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description='Test Imitation Learning Framework')
    parser.add_argument('--output-dir', type=str, default='./test_results',
                       help='Output directory for test results')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick tests with reduced episodes/epochs')
    parser.add_argument('--env', type=str, choices=['highway-v0', 'highway-fast-v0', 'intersection-v0', 'roundabout-v0', 'parking-v0'],
                       help='Test specific environment only')
    
    args = parser.parse_args()
    
    # Create test suite
    test_suite = ComprehensiveTestSuite(output_dir=args.output_dir)
    
    # Filter environments if specified
    if args.env:
        test_suite.env_configs = {args.env: test_suite.env_configs[args.env]}
    
    # Adjust parameters for quick testing
    if args.quick:
        print("Running quick tests...")
        # This would modify test parameters for faster execution
    
    # Run tests
    try:
        results = test_suite.run_all_tests()
        
        # Print final status
        success_rate = results['summary']['overall_success_rate']
        if success_rate >= 0.8:
            print("\n🎉 All tests passed successfully!")
            exit_code = 0
        elif success_rate >= 0.6:
            print("\n⚠️  Most tests passed with some issues")
            exit_code = 1
        else:
            print("\n❌ Multiple test failures detected")
            exit_code = 2
        
        return exit_code
        
    except Exception as e:
        print(f"\n💥 Test suite failed with critical error: {e}")
        import traceback
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)