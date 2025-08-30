# highway-env

[![build](https://github.com/eleurent/highway-env/workflows/build/badge.svg)](https://github.com/eleurent/highway-env/actions?query=workflow%3Abuild)
[![Documentation Status](https://github.com/Farama-Foundation/HighwayEnv/actions/workflows/build-docs-dev.yml/badge.svg)](https://farama-foundation.github.io/HighwayEnv/)
[![Downloads](https://img.shields.io/pypi/dm/highway-env)](https://pypi.org/project/highway-env/)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/63847d9328f64fce9c137b03fcafcc27)](https://app.codacy.com/manual/eleurent/highway-env?utm_source=github.com&utm_medium=referral&utm_content=eleurent/highway-env&utm_campaign=Badge_Grade_Dashboard)
[![GitHub contributors](https://img.shields.io/github/contributors/eleurent/highway-env)](https://github.com/eleurent/highway-env/graphs/contributors)


A collection of environments for *autonomous driving* and tactical decision-making tasks, developed and maintained by [Edouard Leurent](https://github.com/eleurent).

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/highway-env.gif?raw=true"><br/>
    <em>An episode of one of the environments available in highway-env.</em>
</p>

## The environments

### Highway

```python
env = gymnasium.make("highway-v0")
```

In this task, the ego-vehicle is driving on a multilane highway populated with other vehicles.
The agent's objective is to reach a high speed while avoiding collisions with neighbouring vehicles. Driving on the right side of the road is also rewarded.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/highway.gif?raw=true"><br/>
    <em>The highway-v0 environment.</em>
</p>

A faster variant, `highway-fast-v0` is also available, with a degraded simulation accuracy to improve speed for large-scale training.

### Merge

```python
env = gymnasium.make("merge-v0")
```

In this task, the ego-vehicle starts on a main highway but soon approaches a road junction with incoming vehicles on the access ramp. The agent's objective is now to maintain a high speed while making room for the vehicles so that they can safely merge in the traffic.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/merge-env.gif?raw=true"><br/>
    <em>The merge-v0 environment.</em>
</p>

### Roundabout

```python
env = gymnasium.make("roundabout-v0")
```

In this task, the ego-vehicle if approaching a roundabout with flowing traffic. It will follow its planned route automatically, but has to handle lane changes and longitudinal control to pass the roundabout as fast as possible while avoiding collisions.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/roundabout-env.gif?raw=true"><br/>
    <em>The roundabout-v0 environment.</em>
</p>

### Parking

```python
env = gymnasium.make("parking-v0")
```

A goal-conditioned continuous control task in which the ego-vehicle must park in a given space with the appropriate heading.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/parking-env.gif?raw=true"><br/>
    <em>The parking-v0 environment.</em>
</p>

### Intersection

```python
env = gymnasium.make("intersection-v0")
```

An intersection negotiation task with dense traffic.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/intersection-env.gif?raw=true"><br/>
    <em>The intersection-v0 environment.</em>
</p>

### Racetrack

```python
env = gymnasium.make("racetrack-v0")
```

A continuous control task involving lane-keeping and obstacle avoidance.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/racetrack-env.gif?raw=true"><br/>
    <em>The racetrack-v0 environment.</em>
</p>


## Examples of agents

Agents solving the `highway-env` environments are available in the [eleurent/rl-agents](https://github.com/eleurent/rl-agents) and [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) repositories.

See the [documentation](https://farama-foundation.github.io/HighwayEnv/quickstart/#training-an-agent) for some examples and notebooks.

### [Deep Q-Network](https://github.com/eleurent/rl-agents/tree/master/rl_agents/agents/deep_q_network)

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/dqn.gif?raw=true"><br/>
    <em>The DQN agent solving highway-v0.</em>
</p>

This model-free value-based reinforcement learning agent performs Q-learning with function approximation, using a neural network to represent the state-action value function Q.

### [Deep Deterministic Policy Gradient](https://github.com/openai/baselines/tree/master/baselines/her)

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/ddpg.gif?raw=true"><br/>
    <em>The DDPG agent solving parking-v0.</em>
</p>

This model-free policy-based reinforcement learning agent is optimized directly by gradient ascent. It uses Hindsight Experience Replay to efficiently learn how to solve a goal-conditioned task.

### [Value Iteration](https://github.com/eleurent/rl-agents/blob/master/rl_agents/agents/dynamic_programming/value_iteration.py)

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/ttcvi.gif?raw=true"><br/>
    <em>The Value Iteration agent solving highway-v0.</em>
</p>

The Value Iteration is only compatible with finite discrete MDPs, so the environment is first approximated by a [finite-mdp environment](https://github.com/eleurent/finite-mdp) using `env.to_finite_mdp()`. This simplified state representation describes the nearby traffic in terms of predicted Time-To-Collision (TTC) on each lane of the road. The transition model is simplistic and assumes that each vehicle will keep driving at a constant speed without changing lanes. This model bias can be a source of mistakes.

The agent then performs a Value Iteration to compute the corresponding optimal state-value function.

### [Monte-Carlo Tree Search](https://github.com/eleurent/rl-agents/blob/master/rl_agents/agents/tree_search/mcts.py)

This agent leverages a transition and reward models to perform a stochastic tree search [(Coulom, 2006)](https://hal.inria.fr/inria-00116992/document) of the optimal trajectory. No particular assumption is required on the state representation or transition model.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/mcts.gif?raw=true"><br/>
    <em>The MCTS agent solving highway-v0.</em>
</p>

## Installation

`pip install highway-env`

## Usage

```python
import gymnasium as gym

env = gym.make('highway-v0', render_mode='human')

obs, info = env.reset()
done = truncated = False
while not (done or truncated):
    action = ... # Your agent code here
    obs, reward, done, truncated, info = env.step(action)
```
