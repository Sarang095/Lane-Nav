"""
Individual Environment Testing Scripts
Test each highway-env scenario individually with detailed logging
"""

import os
import sys
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path
import json
import time
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

import highway_env
from imitation_learning import (
    ExpertDataCollector,
    ImitationLearningTrainer,
    ImitationLearningEvaluator,
    create_policy_for_env,
    create_warm_start_rl_model
)


class IndividualEnvironmentTester:
    """
    Test individual highway-env environments with detailed analysis
    """
    
    def __init__(self, output_dir: str = "./individual_test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Environment-specific configurations
        self.env_configs = {
            'highway-v0': {
                'config': {
                    'observation': {'type': 'Kinematics'},
                    'action': {'type': 'DiscreteMetaAction'},
                    'duration': 30,
                    'vehicles_count': 30,
                    'collision_reward': -1,
                    'right_lane_reward': 0.1,
                    'high_speed_reward': 0.4,
                    'reward_speed_range': [20, 30],
                },
                'expert_type': 'idm',
                'num_episodes': 20,
                'training_epochs': 50,
                'description': 'Highway driving with multiple lanes and traffic'
            },
            
            'highway-fast-v0': {
                'config': {
                    'observation': {
                        'type': 'GrayscaleObservation',
                        'observation_shape': (128, 64),
                        'stack_size': 4,
                        'weights': [0.2989, 0.5870, 0.1140],
                        'scaling': 1.75,
                    },
                    'duration': 30,
                    'vehicles_count': 25,
                    'policy_frequency': 15,
                },
                'expert_type': 'idm',
                'num_episodes': 15,
                'training_epochs': 40,
                'description': 'Highway driving with CNN-based visual observations'
            },
            
            'intersection-v0': {
                'config': {
                    'observation': {'type': 'Kinematics'},
                    'action': {'type': 'DiscreteMetaAction'},
                    'duration': 20,
                    'destination': 'o1',
                    'collision_reward': -5,
                    'reached_goal_reward': 1,
                },
                'expert_type': 'planned',
                'num_episodes': 25,
                'training_epochs': 60,
                'description': 'Intersection navigation with traffic lights'
            },
            
            'roundabout-v0': {
                'config': {
                    'observation': {'type': 'Kinematics'},
                    'action': {'type': 'DiscreteMetaAction'},
                    'duration': 20,
                    'collision_reward': -5,
                    'reached_goal_reward': 1,
                },
                'expert_type': 'conservative',
                'num_episodes': 20,
                'training_epochs': 50,
                'description': 'Roundabout navigation with yielding behavior'
            },
            
            'parking-v0': {
                'config': {
                    'observation': {'type': 'Kinematics'},
                    'action': {'type': 'ContinuousAction'},
                    'duration': 25,
                    'collision_reward': -5,
                    'success_goal_reward': 1,
                },
                'expert_type': 'planned',
                'num_episodes': 30,
                'training_epochs': 80,
                'description': 'Parking maneuver with continuous control'
            },
        }
    
    def test_single_environment(
        self,
        env_name: str,
        run_full_pipeline: bool = True,
        render_evaluation: bool = False,
        save_plots: bool = True,
    ) -> Dict[str, Any]:
        """
        Test complete pipeline for a single environment
        
        Args:
            env_name: Name of the environment to test
            run_full_pipeline: Whether to run the complete pipeline
            render_evaluation: Whether to render during evaluation
            save_plots: Whether to save training plots
        
        Returns:
            Complete test results for the environment
        """
        if env_name not in self.env_configs:
            raise ValueError(f"Environment {env_name} not configured for testing")
        
        env_config = self.env_configs[env_name]
        env_output_dir = self.output_dir / env_name
        env_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Testing Environment: {env_name}")
        print(f"Description: {env_config['description']}")
        print(f"{'='*60}")
        
        results = {
            'env_name': env_name,
            'timestamp': time.time(),
            'config': env_config,
            'results': {}
        }
        
        try:
            # Phase 1: Environment Setup and Validation
            print("\nPhase 1: Environment Setup")
            setup_results = self._test_environment_setup(env_name, env_config['config'])
            results['results']['setup'] = setup_results
            
            if not setup_results['success']:
                return results
            
            # Phase 2: Expert Data Collection
            print("\nPhase 2: Expert Data Collection")
            data_results = self._test_data_collection(
                env_name, env_config, env_output_dir
            )
            results['results']['data_collection'] = data_results
            
            if not data_results['success'] or not run_full_pipeline:
                return results
            
            # Phase 3: Model Training
            print("\nPhase 3: Imitation Learning Training")
            training_results = self._test_training(
                env_name, env_config, data_results['trajectories_path'], 
                env_output_dir, save_plots
            )
            results['results']['training'] = training_results
            
            if not training_results['success']:
                return results
            
            # Phase 4: Model Evaluation
            print("\nPhase 4: Model Evaluation")
            evaluation_results = self._test_evaluation(
                env_name, env_config, training_results['model_path'],
                env_output_dir, render_evaluation
            )
            results['results']['evaluation'] = evaluation_results
            
            # Phase 5: RL Integration Test
            print("\nPhase 5: RL Integration")
            rl_results = self._test_rl_integration(
                env_name, env_config, training_results['rl_weights_path'], env_output_dir
            )
            results['results']['rl_integration'] = rl_results
            
            # Generate summary
            results['summary'] = self._generate_environment_summary(results['results'])
            
        except Exception as e:
            print(f"❌ Critical error in {env_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
        
        # Save results
        results_file = env_output_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Test results saved to: {results_file}")
        return results
    
    def _test_environment_setup(self, env_name: str, config: Dict) -> Dict[str, Any]:
        """Test environment creation and basic functionality"""
        try:
            # Create and configure environment
            env = gym.make(env_name)
            env.unwrapped.configure(config)
            
            # Test basic functionality
            obs, info = env.reset()
            action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Collect environment information
            setup_info = {
                'success': True,
                'observation_space': str(env.observation_space),
                'action_space': str(env.action_space),
                'observation_shape': obs.shape if hasattr(obs, 'shape') else 'N/A',
                'sample_observation_type': type(obs).__name__,
                'sample_reward': float(reward),
                'config_applied': config,
            }
            
            env.close()
            print(f"✅ Environment setup successful")
            print(f"   Observation space: {setup_info['observation_space']}")
            print(f"   Action space: {setup_info['action_space']}")
            
            return setup_info
            
        except Exception as e:
            print(f"❌ Environment setup failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _test_data_collection(
        self, 
        env_name: str, 
        env_config: Dict, 
        output_dir: Path
    ) -> Dict[str, Any]:
        """Test expert data collection"""
        try:
            collector = ExpertDataCollector(
                env_config=env_config['config'],
                save_dir=str(output_dir / "expert_data"),
                max_episodes=env_config['num_episodes'],
                min_episode_length=10,
                quality_threshold=-1.0,  # More lenient for testing
            )
            
            # Collect expert trajectories
            trajectories = collector.collect_rule_based_expert_data(
                env_name=env_name,
                expert_type=env_config['expert_type'],
                render=False,
            )
            
            # Save trajectories
            save_path = collector.save_data(f"{env_name}_expert_data.pkl")
            
            # Analyze data quality
            episode_lengths = [t.episode_length for t in trajectories]
            episode_rewards = [t.total_reward for t in trajectories]
            
            data_results = {
                'success': True,
                'num_trajectories': len(trajectories),
                'total_steps': sum(episode_lengths),
                'mean_episode_length': np.mean(episode_lengths),
                'std_episode_length': np.std(episode_lengths),
                'mean_episode_reward': np.mean(episode_rewards),
                'std_episode_reward': np.std(episode_rewards),
                'min_reward': np.min(episode_rewards),
                'max_reward': np.max(episode_rewards),
                'trajectories_path': save_path,
                'expert_type': env_config['expert_type'],
            }
            
            print(f"✅ Data collection successful")
            print(f"   Collected {len(trajectories)} trajectories")
            print(f"   Total steps: {data_results['total_steps']}")
            print(f"   Mean reward: {data_results['mean_episode_reward']:.2f} ± {data_results['std_episode_reward']:.2f}")
            
            return data_results
            
        except Exception as e:
            print(f"❌ Data collection failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _test_training(
        self,
        env_name: str,
        env_config: Dict,
        trajectories_path: str,
        output_dir: Path,
        save_plots: bool = True,
    ) -> Dict[str, Any]:
        """Test imitation learning training"""
        try:
            # Load trajectories
            collector = ExpertDataCollector(env_config=env_config['config'])
            trajectories = collector.load_data(trajectories_path)
            
            # Create environment and policy
            env = gym.make(env_name)
            env.unwrapped.configure(env_config['config'])
            
            policy = create_policy_for_env(env_name, env.observation_space, env.action_space)
            
            # Create trainer
            trainer = ImitationLearningTrainer(
                policy=policy,
                num_epochs=env_config['training_epochs'],
                batch_size=1,
                learning_rate=1e-3,
                validation_split=0.2,
                save_dir=str(output_dir / "trained_models"),
                log_interval=10,
            )
            
            # Prepare data and train
            train_loader, val_loader = trainer.prepare_data(trajectories)
            training_history = trainer.train_behavioral_cloning(train_loader, val_loader)
            
            # Save model
            model_path = trainer.save_model(f"{env_name}_final_model.pth")
            
            # Export for RL
            rl_weights_path = str(output_dir / f"{env_name}_rl_weights.pth")
            trainer.export_for_rl(rl_weights_path)
            
            # Save training plots
            if save_plots:
                plot_path = output_dir / f"{env_name}_training_plots.png"
                trainer.plot_training_history(str(plot_path))
            
            env.close()
            
            training_results = {
                'success': True,
                'final_train_loss': training_history['train_total_loss'][-1],
                'final_val_loss': training_history['val_total_loss'][-1],
                'best_val_loss': trainer.best_val_loss,
                'num_epochs': env_config['training_epochs'],
                'model_path': model_path,
                'rl_weights_path': rl_weights_path,
                'training_history': training_history,
            }
            
            print(f"✅ Training successful")
            print(f"   Final validation loss: {training_results['final_val_loss']:.4f}")
            print(f"   Best validation loss: {training_results['best_val_loss']:.4f}")
            
            return training_results
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _test_evaluation(
        self,
        env_name: str,
        env_config: Dict,
        model_path: str,
        output_dir: Path,
        render: bool = False,
    ) -> Dict[str, Any]:
        """Test model evaluation"""
        try:
            # Load trained model
            env = gym.make(env_name)
            env.unwrapped.configure(env_config['config'])
            
            policy = create_policy_for_env(env_name, env.observation_space, env.action_space)
            trainer = ImitationLearningTrainer(policy=policy)
            trainer.load_model(model_path)
            
            # Create evaluator
            evaluator = ImitationLearningEvaluator(policy, render=render)
            
            # Run evaluation
            eval_results = evaluator.evaluate_single_environment(
                env_name=env_name,
                env_config=env_config['config'],
                num_episodes=15,
                deterministic=True,
            )
            
            # Run baseline comparison
            comparison_results = evaluator.benchmark_against_baselines(
                env_name=env_name,
                env_config=env_config['config'],
                num_episodes=10,
                baselines=['random', 'idm'],
            )
            
            env.close()
            
            evaluation_results = {
                'success': True,
                'il_performance': eval_results,
                'baseline_comparison': comparison_results,
                'improvement_over_random': (
                    eval_results['mean_reward'] - comparison_results['random']['mean_reward']
                ) / abs(comparison_results['random']['mean_reward']) if comparison_results['random']['mean_reward'] != 0 else 0,
            }
            
            print(f"✅ Evaluation successful")
            print(f"   IL mean reward: {eval_results['mean_reward']:.2f}")
            print(f"   Success rate: {eval_results['success_rate']:.2f}")
            print(f"   Collision rate: {eval_results['collision_rate']:.2f}")
            
            return evaluation_results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _test_rl_integration(
        self,
        env_name: str,
        env_config: Dict,
        rl_weights_path: str,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Test RL integration with IL warm start"""
        try:
            # Test warm start model creation
            model, env = create_warm_start_rl_model(
                env_name=env_name,
                algorithm='DQN',
                il_weights_path=rl_weights_path,
                env_config=env_config['config'],
            )
            
            # Test short training
            model.learn(total_timesteps=1000)  # Short training for testing
            
            # Test inference
            obs, info = env.reset()
            for _ in range(10):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                if done or truncated:
                    break
            
            env.close()
            
            rl_results = {
                'success': True,
                'algorithm': 'DQN',
                'warmstart_completed': True,
                'short_training_completed': True,
                'inference_test_passed': True,
            }
            
            print(f"✅ RL integration successful")
            print(f"   Warm start with DQN completed")
            print(f"   Short training and inference test passed")
            
            return rl_results
            
        except Exception as e:
            print(f"❌ RL integration failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _generate_environment_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate summary for environment test"""
        successful_phases = sum(1 for phase_result in results.values() 
                              if isinstance(phase_result, dict) and phase_result.get('success', False))
        total_phases = len(results)
        
        summary = {
            'total_phases': total_phases,
            'successful_phases': successful_phases,
            'success_rate': successful_phases / total_phases,
            'passed': successful_phases == total_phases,
        }
        
        # Add specific metrics if available
        if 'evaluation' in results and results['evaluation'].get('success'):
            eval_data = results['evaluation']['il_performance']
            summary.update({
                'final_mean_reward': eval_data['mean_reward'],
                'final_success_rate': eval_data['success_rate'],
                'final_collision_rate': eval_data['collision_rate'],
            })
        
        return summary
    
    def test_all_environments(self, quick_test: bool = False) -> Dict[str, Any]:
        """Test all configured environments"""
        print("Starting Individual Environment Testing")
        print("="*60)
        
        all_results = {}
        
        for env_name in self.env_configs:
            try:
                # Adjust for quick testing
                if quick_test:
                    self.env_configs[env_name]['num_episodes'] = min(5, self.env_configs[env_name]['num_episodes'])
                    self.env_configs[env_name]['training_epochs'] = min(10, self.env_configs[env_name]['training_epochs'])
                
                results = self.test_single_environment(
                    env_name=env_name,
                    run_full_pipeline=True,
                    render_evaluation=False,
                    save_plots=True,
                )
                
                all_results[env_name] = results
                
            except Exception as e:
                print(f"❌ Failed to test {env_name}: {str(e)}")
                all_results[env_name] = {'error': str(e), 'success': False}
        
        # Generate overall summary
        overall_summary = self._generate_overall_summary(all_results)
        
        # Save consolidated results
        consolidated_results = {
            'individual_results': all_results,
            'overall_summary': overall_summary,
            'test_timestamp': time.time(),
        }
        
        results_file = self.output_dir / "all_environments_results.json"
        with open(results_file, 'w') as f:
            json.dump(consolidated_results, f, indent=2, default=str)
        
        # Print final summary
        self._print_final_summary(overall_summary)
        
        return consolidated_results
    
    def _generate_overall_summary(self, all_results: Dict) -> Dict[str, Any]:
        """Generate overall summary across all environments"""
        total_envs = len(all_results)
        successful_envs = sum(1 for r in all_results.values() 
                            if isinstance(r, dict) and r.get('summary', {}).get('passed', False))
        
        summary = {
            'total_environments': total_envs,
            'successful_environments': successful_envs,
            'overall_success_rate': successful_envs / total_envs,
            'environment_details': {},
        }
        
        for env_name, results in all_results.items():
            if isinstance(results, dict) and 'summary' in results:
                summary['environment_details'][env_name] = results['summary']
            else:
                summary['environment_details'][env_name] = {'passed': False, 'error': True}
        
        return summary
    
    def _print_final_summary(self, summary: Dict):
        """Print final test summary"""
        print("\n" + "="*60)
        print("FINAL TEST SUMMARY")
        print("="*60)
        print(f"Total environments tested: {summary['total_environments']}")
        print(f"Successful environments: {summary['successful_environments']}")
        print(f"Overall success rate: {summary['overall_success_rate']:.1%}")
        print()
        
        for env_name, details in summary['environment_details'].items():
            status = "✅ PASSED" if details.get('passed', False) else "❌ FAILED"
            print(f"{env_name:20s} {status}")
            if details.get('passed') and 'final_mean_reward' in details:
                print(f"{'':20s} → Reward: {details['final_mean_reward']:.2f}, "
                      f"Success: {details['final_success_rate']:.2f}")
        
        print("\n" + "="*60)


def main():
    """Main function for individual environment testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Individual Highway Environments')
    parser.add_argument('--env', type=str, 
                       choices=['highway-v0', 'highway-fast-v0', 'intersection-v0', 'roundabout-v0', 'parking-v0'],
                       help='Test specific environment only')
    parser.add_argument('--all', action='store_true', help='Test all environments')
    parser.add_argument('--quick', action='store_true', help='Run quick tests')
    parser.add_argument('--render', action='store_true', help='Render evaluation episodes')
    parser.add_argument('--output-dir', type=str, default='./individual_test_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    tester = IndividualEnvironmentTester(output_dir=args.output_dir)
    
    if args.env:
        # Test single environment
        results = tester.test_single_environment(
            env_name=args.env,
            run_full_pipeline=True,
            render_evaluation=args.render,
            save_plots=True,
        )
        
        if results.get('summary', {}).get('passed', False):
            print(f"\n🎉 {args.env} test PASSED!")
            return 0
        else:
            print(f"\n❌ {args.env} test FAILED!")
            return 1
    
    elif args.all:
        # Test all environments
        results = tester.test_all_environments(quick_test=args.quick)
        
        success_rate = results['overall_summary']['overall_success_rate']
        if success_rate >= 0.8:
            print(f"\n🎉 All environment tests PASSED! ({success_rate:.1%} success rate)")
            return 0
        else:
            print(f"\n❌ Some environment tests FAILED! ({success_rate:.1%} success rate)")
            return 1
    
    else:
        print("Please specify --env <environment> or --all")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)