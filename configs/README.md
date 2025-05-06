# YAML 配置系统使用指南

本目录包含用于配置强化学习算法实验的YAML配置文件。这些配置文件可以替代命令行参数，提供更结构化、可重用的实验设置。

## 配置文件组织

- `ppo_config.yaml`: PPO算法配置
- `ddpg_config.yaml`: DDPG算法配置
- `ars_config.yaml`: ARS算法配置
- `comparison_config.yaml`: 多算法比较配置
- `experiment_template.yaml`: 实验模板，可复制修改创建新实验
- `exp1_large_rad.yaml`, `exp2_small_rad.yaml`, `exp3_more_radii.yaml`: 不同MSZoo参数的实验配置

## 使用方法

### 基本使用

使用配置文件运行实验非常简单，只需使用`--config`参数指向配置文件:

```bash
# 使用PPO配置文件运行
python run_algorithms.py ppo --config configs/ppo_config.yaml

# 使用DDPG配置文件运行
python run_algorithms.py --config configs/ddpg_config.yaml

# 比较多个算法
python run_algorithms.py --config configs/comparison_config.yaml
```

### 覆盖配置

您可以在命令行上覆盖配置文件中的参数:

```bash
# 覆盖环境名称
python run_algorithms.py --config configs/ppo_config.yaml --env_name Walker2d-v4

# 覆盖时间步数
python run_algorithms.py --config configs/comparison_config.yaml --timesteps 1000000
```

### 创建新的实验配置

1. 复制`experiment_template.yaml`:
   ```bash
   cp configs/experiment_template.yaml configs/my_experiment.yaml
   ```

2. 编辑新的配置文件，修改您想要的参数

3. 运行新的实验:
   ```bash
   python run_algorithms.py --config configs/my_experiment.yaml
   ```

## 配置参数说明

### 基本参数

| 参数名 | 说明 | 默认值 |
|-------|------|-------|
| exp_name | 实验名称 | "实验名称" |
| env_name | 环境名称 | HalfCheetah-v4 |
| seed | 随机种子 | 42 |
| timesteps | 训练总步数 | 500000 |
| algorithms | 要运行的算法列表 | [ms_ppo, ms_ddpg] |
| n_workers | 并行工作进程数量 | 8 |
| parallel_collection | 是否使用并行收集经验 | true |

### 算法特定参数

每个算法有特定的参数组，可查看模板文件了解详情。

### MSZoo特定参数

| 参数名 | 说明 | 默认值 |
|-------|------|-------|
| use_mszoo | 是否使用MSZoo优化器 | true |
| mszoo_config.perturbation_radii | 扰动半径列表 | [0.01, 0.02, 0.05] |
| mszoo_config.population_size | 每个扰动半径的群体大小 | 10 |
| mszoo_config.weight_update_interval | 权重更新间隔 | 20 |
| mszoo_config.noise_std | 初始噪声标准差 | 0.15 |
| mszoo_config.noise_decay | 噪声衰减率 | 0.995 |

## 示例说明

- `exp1_large_rad.yaml`: 使用较大的扰动半径，适合需要广泛探索的问题
- `exp2_small_rad.yaml`: 使用较小的扰动半径，适合需要精细优化的问题
- `exp3_more_radii.yaml`: 使用更多的扰动半径，既有大半径又有小半径，更全面地探索和优化 