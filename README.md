# Augmented Random Search (ARS) with Multiple Optimizers and Deep RL Algorithms

本项目包含多种强化学习算法实现，主要包括:

1. **ARS (Augmented Random Search)** 及其多种优化器变种
2. **PPO (Proximal Policy Optimization)** - 基于Actor-Critic架构的on-policy算法
3. **DDPG (Deep Deterministic Policy Gradient)** - 基于Actor-Critic架构的off-policy算法

这些算法在连续动作空间控制任务（如MuJoCo环境）上进行了实现和测试。

## 介绍

### ARS (Augmented Random Search)

ARS是一种简单而有效的基于随机搜索的无梯度强化学习算法。该算法通过对策略参数的随机扰动来探索参数空间，并提高策略性能。其特点包括:

- 不需要计算梯度，仅用于线性策略训练
- 实现简单、计算高效，特别是在并行化后
- 在某些任务上表现与复杂算法相当但用时更少

本项目扩展了原始ARS算法，实现了多种优化器选项:

1. **SGD**：原始论文中的标准随机梯度下降
2. **Adam**：自适应矩估计优化器，具有动态学习率调整
3. **Zero-Order**：零阶优化方法，使用随机扰动估计梯度
4. **Multi-Scale Zero-Order**：多尺度零阶优化，使用多个扰动尺度

### PPO (Proximal Policy Optimization)

PPO是一种流行的基于策略梯度的深度强化学习算法，其特点包括:

- 使用裁剪目标函数限制策略更新幅度，提高训练稳定性
- 采用GAE (Generalized Advantage Estimation) 估计优势函数
- 多次利用采样数据，提高数据效率
- 可处理连续和离散动作空间
- 相对稳定的训练过程

### DDPG (Deep Deterministic Policy Gradient)

DDPG是一种深度确定性策略梯度算法，其特点包括:

- 结合DQN与策略梯度方法，面向连续动作空间
- 使用确定性策略减少方差
- 采用经验回放打破数据相关性
- 使用目标网络的软更新提高训练稳定性
- 添加噪声用于动作探索

## 安装

要运行此代码，您需要安装MuJoCo和OpenAI Gym，以及Ray用于并行处理。

```bash
pip install gym ray numpy torch matplotlib
```

确保根据您的环境正确设置MuJoCo。

## 使用方法

### 使用特定算法训练

#### 使用ARS

```bash
python code/ars.py --env_name HalfCheetah-v4 --n_iter 1000 --n_directions 8 --deltas_used 8 --step_size 0.02 --delta_std 0.03 --optimizer_type sgd
```

可用的优化器类型:
- `sgd` (默认)
- `adam`
- `zero_order`
- `multi_scale`

#### 使用PPO

```python
from models.ppo import PPO, make_env

env = make_env("HalfCheetah-v4")
ppo = PPO(env)
ppo.train(total_timesteps=1000000)
ppo.test(num_episodes=10, render=True)
```

#### 使用DDPG

```python
from models.ddpg import DDPG, make_env

env = make_env("HalfCheetah-v4")
ddpg = DDPG(env)
ddpg.train(total_timesteps=1000000)
ddpg.test(num_episodes=10, render=True)
```

### 比较不同算法

您可以使用`run_algorithms.py`脚本来比较不同算法的性能:

```bash
python code/run_algorithms.py --algorithms ppo --env_name Ant-v4 --timesteps 500000 --optimizer_type sgd
```

主要参数:
- `--algorithms`: 要比较的算法列表，可选 `ars`, `ppo`, `ddpg`
- `--env_name`: 环境名称
- `--timesteps`: 训练总步数
- `--optimizer_type`: ARS使用的优化器类型
- `--seed`: 随机种子，确保实验可重复性

更多参数可通过 `python code/run_algorithms.py --help` 查看。

## 项目结构

- `code/ars.py`: ARS算法实现
- `code/optimizers.py`: 不同优化器的实现
- `code/models/ppo.py`: PPO算法实现
- `code/models/ddpg.py`: DDPG算法实现 
- `code/run_optimizers.py`: 比较不同ARS优化器的脚本
- `code/run_algorithms.py`: 比较所有算法的脚本
- `code/utils.py`: 工具函数

## 算法比较

三种算法各有特点:

| 算法 | 类型 | 优点 | 缺点 |
|------|-----|------|------|
| ARS  | 无梯度、随机搜索 | 简单高效、易于并行、性能良好 | 仅适用于线性策略、难以处理高维空间 |
| PPO  | 基于梯度、On-policy | 稳定性好、数据效率高、适用范围广 | 计算复杂、超参数敏感 |
| DDPG | 基于梯度、Off-policy | 样本效率高、适合连续控制 | 训练不稳定、调参难度大 |

## 参考文献

- ARS论文: [Simple random search of static linear policies is competitive for reinforcement learning](https://arxiv.org/abs/1803.07055)
- PPO论文: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- DDPG论文: [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
- qtune-mysql: 用于MySQL数据库参数调优的优化方法

## 运行环境说明

请确保您的MuJoCo环境正确设置:

```bash
export MUJOCO_PY_MUJOCO_PATH="/home/liangchen/.mujoco/mujoco210/"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/liangchen/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

## Prerequisites for running ARS

Our ARS implementation relies on Python 3, OpenAI Gym version 0.9.3, mujoco-py 0.5.7, MuJoCo Pro version 1.31, and the Ray library for parallel computing.  

To install OpenAI Gym and MuJoCo dependencies follow the instructions here:
https://github.com/openai/gym

To install Ray execute:
``` 
pip install ray
```
For more information on Ray see http://ray.readthedocs.io/en/latest/. 

## Running ARS

First start Ray by executing a command of the following form:

```
ray start --head --redis-port=6379 --num-workers=18
```
This command starts multiple Python processes on one machine for parallel computations with Ray. 
Set "num_workers=X" for parallelizing ARS across X CPUs.
For parallelzing ARS on a cluster follow the instructions here: http://ray.readthedocs.io/en/latest/using-ray-on-a-large-cluster.html.

We recommend using single threaded linear algebra computations by setting: 
```
export MKL_NUM_THREADS=1
```

To train a policy for HalfCheetah-v1, execute the following command: 

```
python code/ars.py
```

All arguments passed into ARS are optional and can be modified to train other environments, use different hyperparameters, or use  different random seeds.
For example, to train a policy for Humanoid-v1, execute the following command:

```
python code/ars.py --env_name Humanoid-v1 --n_directions 230 --deltas_used 230 --step_size 0.02 --delta_std 0.0075 --n_workers 48 --shift 5
```

## Rendering Trained Policy

To render a trained policy, execute a command of the following form:

```
python code/run_policy.py trained_polices/env_name/policy_directory_path/policy_file_name.npz env_name --render
```

For example, to render Humanoid-v1 with a galloping gait execute:

```
python code/run_policy.py trained_policies/Humanoid-v1/policy_reward_11600/lin_policy_plus.npz Humanoid-v1 --render 
```


export MUJOCO_PY_MUJOCO_PATH="/home/liangchen/.mujoco/mujoco210/"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/liangchen/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia