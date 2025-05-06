# 深度强化学习算法实现

本目录包含了两种现代深度强化学习算法的实现：

1. **PPO (Proximal Policy Optimization)**
2. **DDPG (Deep Deterministic Policy Gradient)**

这些算法都是基于Actor-Critic架构，并面向连续动作空间的控制任务。

## 算法简介

### PPO (Proximal Policy Optimization)

PPO是一种on-policy的强化学习算法，特点是：

- 使用裁剪的目标函数，限制策略更新的幅度
- 采用GAE (Generalized Advantage Estimation) 估计优势函数
- 多次利用收集的轨迹进行多轮优化
- 熵正则化鼓励探索
- 相对稳定的训练过程，参数对超参数不太敏感

### DDPG (Deep Deterministic Policy Gradient)

DDPG是一种off-policy的强化学习算法，特点是：

- 结合了Deep Q-Network (DQN) 和策略梯度方法的优点
- 使用确定性策略，减少动作采样带来的方差
- 使用经验回放缓冲区打破数据相关性
- 采用软更新的目标网络提高稳定性
- 使用Ornstein-Uhlenbeck过程生成探索噪声

## 使用方法

### 单独使用

每个算法都可以作为独立模块导入并使用：

```python
# 使用PPO
from models.ppo import PPO, make_env

# 创建环境
env = make_env("HalfCheetah-v4")

# 创建PPO实例
ppo = PPO(env, 
          hidden_dim=256,
          actor_lr=3e-4,
          critic_lr=1e-3,
          gamma=0.99)

# 训练模型
ppo.train(total_timesteps=1000000)

# 测试模型
ppo.test(num_episodes=10, render=True)

# 保存与加载模型
ppo.save_model('models/ppo_model.pt')
ppo.load_model('models/ppo_model.pt')
```

```python
# 使用DDPG
from models.ddpg import DDPG, make_env

# 创建环境
env = make_env("HalfCheetah-v4")

# 创建DDPG实例
ddpg = DDPG(env, 
            hidden_dim=256,
            actor_lr=1e-4,
            critic_lr=1e-3)

# 训练模型
ddpg.train(total_timesteps=1000000)

# 测试模型
ddpg.test(num_episodes=10, render=True)

# 保存与加载模型
ddpg.save_model('models/ddpg_model.pt')
ddpg.load_model('models/ddpg_model.pt')
```

### 使用比较脚本

项目包含了一个比较不同算法的脚本：

```bash
python code/run_algorithms.py --algorithms ppo ddpg ars --env_name HalfCheetah-v4 --timesteps 500000 --optimizer_type sgd
```

主要参数：

- `--algorithms`: 要运行的算法列表，可选 `ppo`, `ddpg`, `ars`
- `--env_name`: 要运行的Gym环境名称
- `--timesteps`: 总训练步数
- `--optimizer_type`: ARS的优化器类型，可选 `sgd`, `adam`, `zero_order`, `multi_scale`

更多参数可通过 `python code/run_algorithms.py --help` 查看。

## 与ARS的比较

相比于ARS (Augmented Random Search)，PPO和DDPG是基于梯度的深度强化学习算法，它们：

1. 使用神经网络来表示策略和值函数，而不是线性策略
2. 可以直接处理高维状态空间和复杂任务
3. 通常在复杂任务上获得更高的回报，但训练时间更长
4. 对超参数的敏感性更高，调整难度更大
5. 计算资源需求更高，因为需要反向传播

## 主要文件

- `ppo.py`: PPO算法的完整实现
- `ddpg.py`: DDPG算法的完整实现
- `__init__.py`: 包导入定义

## 参考文献

- PPO: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- DDPG: [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
- ARS: [Simple random search provides a competitive approach to reinforcement learning](https://arxiv.org/abs/1803.07055) 