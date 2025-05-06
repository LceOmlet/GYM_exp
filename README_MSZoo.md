# 多尺度零阶优化 (MSZoo) 强化学习算法

本项目实现了多尺度零阶优化版本的PPO和DDPG算法，为强化学习提供更高效的策略优化方式。

## 特点

MSZoo优化器具有以下特点：

- **无梯度优化**：直接探索策略参数空间，不依赖于梯度下降
- **多尺度扰动**：同时使用多种扰动尺度，兼顾探索和精细调整
- **自适应权重调整**：根据训练阶段自动调整不同扰动尺度的权重
- **高效探索**：比传统零阶优化方法更高效地探索策略空间

## 实现的算法

- **MSZooPPO**：多尺度零阶优化版本的PPO算法
- **MSZooDDPG**：多尺度零阶优化版本的DDPG算法

## 使用方法

### 通过YAML配置文件运行

最简单的方式是使用YAML配置文件，它提供了更结构化、可重用的实验设置：

```bash
# 使用PPO配置文件运行MSZoo-PPO
python run_algorithms.py ppo --config configs/ppo_config.yaml

# 使用不同扰动半径的实验配置
python run_algorithms.py --config configs/exp1_large_rad.yaml
```

查看`configs/`目录下的README和示例配置文件，了解更多参数设置。

### 命令行运行

```bash
# 单独运行MSZooPPO
python run_algorithms.py ppo --env_name HalfCheetah-v4 --use_mszoo

# 并行版本
python run_algorithms.py ppo --env_name HalfCheetah-v4 --use_mszoo --parallel_collection --n_workers 4
```

#### 比较多个算法

```bash
# 比较MSZooPPO、MSZooDDPG与标准PPO、DDPG、ARS
python run_algorithms.py --env_name HalfCheetah-v4 --algorithms ms_ppo ms_ddpg ppo ddpg ars
```

### 代码中使用

```python
# MSZooPPO示例
from models.ms_ppo import MSZooPPO
import gym

# 创建环境
env = gym.make("HalfCheetah-v4")

# MSZoo配置
mszoo_config = {
    "perturbation_radii": [0.01, 0.02, 0.05],
    "population_size": 10, 
    "weight_update_interval": 20,
    "noise_std": 0.15,
    "noise_decay": 0.995
}

# 创建MSZooPPO实例
ppo = MSZooPPO(
    env=env,
    hidden_dim=1024,
    actor_lr=2e-4,
    critic_lr=1e-3,
    batch_size=512,
    mszoo_config=mszoo_config
)

# 训练
ppo.train(total_timesteps=1000000)
```

```python
# MSZooDDPG示例
from models.ms_ddpg import MSZooDDPG
import gym

# 创建环境
env = gym.make("HalfCheetah-v4")

# MSZoo配置
mszoo_config = {
    "perturbation_radii": [0.01, 0.02, 0.05],
    "population_size": 10, 
    "weight_update_interval": 20,
    "noise_std": 0.15,
    "noise_decay": 0.995
}

# 创建MSZooDDPG实例
ddpg = MSZooDDPG(
    env=env,
    hidden_dim=256,
    actor_lr=1e-4,
    critic_lr=1e-3,
    batch_size=64,
    mszoo_config=mszoo_config
)

# 训练
ddpg.train(total_timesteps=1000000)
```

## 参数说明

MSZoo配置参数说明：

| 参数名称 | 说明 | 默认值 |
|---------|------|-------|
| perturbation_radii | 扰动半径列表 | [0.01, 0.02, 0.05] |
| population_size | 每个扰动半径的群体大小 | 10 |
| weight_update_interval | 权重更新间隔 | 20 |
| noise_std | 初始噪声标准差 | 0.15 |
| noise_decay | 噪声衰减率 | 0.995 |

## 实验配置

我们提供了几个预配置的实验YAML文件，用于测试不同的MSZoo参数设置：

- `configs/exp1_large_rad.yaml`: 使用较大的扰动半径 [0.05, 0.1, 0.2]，适合需要广泛探索的问题
- `configs/exp2_small_rad.yaml`: 使用较小的扰动半径 [0.001, 0.005, 0.01]，适合需要精细优化的问题
- `configs/exp3_more_radii.yaml`: 使用更多的扰动半径 [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]，更全面地探索和优化

要创建自己的实验配置，可以复制并修改`configs/experiment_template.yaml`文件。

## 工作原理

MSZoo优化器基于以下原理工作：

1. **多尺度扰动**：为Actor网络参数生成多个扰动版本，使用不同大小的扰动半径
2. **策略评估**：使用Critic网络或其他评价指标评估每个扰动版本的性能
3. **自适应权重**：根据训练阶段和性能表现，动态调整不同扰动半径的权重
4. **权重更新**：根据评估结果和权重，更新Actor网络参数

MSZoo特别适合：
- 难以获得可靠梯度信息的环境
- 需要广泛探索的复杂策略空间
- 需要高精度调整的精细策略空间

## 与标准方法的比较

相比于标准的基于梯度的方法，MSZoo优化具有以下优势：

1. 不依赖梯度计算，避免了梯度消失/爆炸问题
2. 能够跳出局部最优，探索更广的参数空间
3. 同时兼顾粗粒度探索和细粒度优化
4. 自适应调整不同规模扰动的重要性

## 实现细节

MSZoo优化对于PPO和DDPG的实现都保留了原算法框架，只替换了Actor网络的更新方式：

- 在PPO中，MSZoo替代了传统的PPO裁剪目标梯度更新
- 在DDPG中，MSZoo替代了确定性策略梯度更新

两种实现都保留了原算法的关键特性：
- PPO保留了GAE优势估计和价值函数更新
- DDPG保留了经验回放和目标网络 