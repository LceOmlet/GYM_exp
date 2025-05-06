#!/usr/bin/env python3
"""
多尺度零阶优化 (MSZoo) 强化学习算法运行脚本

这个脚本提供了一个简单的方式来运行MSZoo优化版本的PPO和DDPG算法。
它支持YAML配置文件和命令行参数，并可以在多种环境上进行训练和测试。

使用方法:
    # 运行MSZoo-PPO
    python run_mszoo.py ppo --env_name HalfCheetah-v4
    
    # 运行MSZoo-DDPG
    python run_mszoo.py ddpg --env_name Pendulum-v1
    
    # 使用YAML配置文件
    python run_mszoo.py ppo --config configs/exp1_large_rad.yaml
    
    # 并行训练
    python run_mszoo.py ppo --env_name HalfCheetah-v4 --parallel --n_workers 4
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import logging
import ray
from collections import deque
import datetime
import socket

# 尝试导入yaml, 如果未安装则给出安装提示
try:
    import yaml
except ImportError:
    print("错误: 未找到PyYAML库。请使用pip安装:")
    print("pip install pyyaml")
    exit(1)

# 导入MSZoo优化版本的算法
from src.models.ms_ppo import MSZooPPO
from src.models.ms_ddpg import MSZooDDPG
from src.models.ppo import make_env as make_env_ppo
from src.models.ddpg import make_env as make_env_ddpg

# 确保优化器被导入以注册到注册表
import src.optimizers

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MSZoo强化学习算法训练')
    
    # 基本参数
    parser.add_argument('algorithm', type=str, choices=['ppo', 'ddpg'],
                        help='要运行的算法: ppo 或 ddpg')
    
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4',
                        help='OpenAI Gym环境名称')
    
    parser.add_argument('--config', type=str, default=None,
                       help='YAML配置文件的路径')
    
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='训练的总时间步数')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    parser.add_argument('--parallel', action='store_true',
                        help='是否使用并行方式训练')
    
    parser.add_argument('--n_workers', type=int, default=4,
                        help='并行工作进程数量')
    
    parser.add_argument('--save_dir', type=str, default=None,
                        help='保存模型和结果的目录，默认使用时间戳创建')
    
    parser.add_argument('--save_interval', type=int, default=100,
                        help='模型保存间隔')
    
    parser.add_argument('--log_interval', type=int, default=10,
                        help='日志打印间隔')
    
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='测试时的评估回合数')
    
    parser.add_argument('--render', action='store_true', 
                        help='测试时是否渲染环境')
    
    parser.add_argument('--load_path', type=str, default=None,
                        help='加载模型的路径')
    
    parser.add_argument('--test_only', action='store_true',
                        help='只进行测试，不训练')
    
    # PPO相关参数
    parser.add_argument('--hidden_dim', type=int, default=1024,
                        help='隐藏层维度')
    
    parser.add_argument('--actor_lr', type=float, default=2e-4,
                        help='Actor学习率')
    
    parser.add_argument('--critic_lr', type=float, default=1e-3,
                        help='Critic学习率')
    
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小 (PPO默认512, DDPG默认64)')
    
    parser.add_argument('--ppo_epochs', type=int, default=5,
                        help='PPO每次更新的epoch数')
    
    parser.add_argument('--clip_ratio', type=float, default=0.2,
                        help='PPO裁剪参数')
    
    parser.add_argument('--target_kl', type=float, default=0.01,
                        help='PPO目标KL散度')
    
    parser.add_argument('--ent_coef', type=float, default=0.01,
                        help='PPO熵正则化系数')
    
    parser.add_argument('--policy_update_interval', type=int, default=2048,
                        help='PPO策略更新间隔的环境步数')
    
    parser.add_argument('--normalize_states', action='store_true', default=True,
                        help='是否对状态进行归一化')
    
    # DDPG相关参数
    parser.add_argument('--buffer_size', type=int, default=1000000,
                        help='DDPG的经验回放缓冲区大小')
    
    parser.add_argument('--tau', type=float, default=0.005,
                        help='DDPG软更新系数')
    
    parser.add_argument('--start_steps', type=int, default=10000,
                        help='DDPG训练前的随机步数')
    
    parser.add_argument('--noise_theta', type=float, default=0.15,
                        help='DDPG的OU噪声theta参数')
    
    parser.add_argument('--noise_sigma', type=float, default=0.2,
                        help='DDPG的OU噪声sigma参数')
    
    # MSZoo特定参数
    parser.add_argument('--perturbation_radii', type=str, default='0.01,0.02,0.05',
                       help='扰动半径列表，用逗号分隔')
    
    parser.add_argument('--population_size', type=int, default=10,
                       help='每个扰动半径的群体大小')
    
    parser.add_argument('--weight_update_interval', type=int, default=20,
                       help='多尺度零阶优化器权重更新间隔')
    
    parser.add_argument('--noise_std', type=float, default=0.15,
                       help='多尺度零阶优化器的噪声标准差')
    
    parser.add_argument('--noise_decay', type=float, default=0.995,
                       help='多尺度零阶优化器的噪声衰减率')
    
    args = parser.parse_args()
    
    # 设置算法特定的默认值
    if args.batch_size is None:
        if args.algorithm == 'ppo':
            args.batch_size = 512
        else:  # ddpg
            args.batch_size = 64
    
    return args

def setup_env_variables():
    """设置MuJoCo环境变量"""
    if "MUJOCO_PY_MUJOCO_PATH" not in os.environ:
        os.environ["MUJOCO_PY_MUJOCO_PATH"] = "/home/liangchen/.mujoco/mujoco210/"
    
    if "LD_LIBRARY_PATH" not in os.environ:
        os.environ["LD_LIBRARY_PATH"] = ""
    
    if "/home/liangchen/.mujoco/mujoco210/bin" not in os.environ["LD_LIBRARY_PATH"]:
        os.environ["LD_LIBRARY_PATH"] += ":/home/liangchen/.mujoco/mujoco210/bin"
    
    if "/usr/lib/nvidia" not in os.environ["LD_LIBRARY_PATH"]:
        os.environ["LD_LIBRARY_PATH"] += ":/usr/lib/nvidia"

def load_config(config_file):
    """从YAML文件加载配置参数"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"加载配置文件 {config_file} 失败: {e}")
        return {}

def update_args_with_config(args, config):
    """使用配置文件中的参数更新命令行参数"""
    # 只更新已存在于args中且在config中有定义的参数
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args

def create_mszoo_config(args):
    """根据参数创建MSZoo配置字典"""
    mszoo_config = {
        "perturbation_radii": [float(r) for r in args.perturbation_radii.split(',')],
        "population_size": args.population_size,
        "weight_update_interval": args.weight_update_interval,
        "noise_std": args.noise_std,
        "noise_decay": args.noise_decay
    }
    return mszoo_config

def train_mszoo_ppo(args, mszoo_config):
    """训练MSZoo优化的PPO算法"""
    logger.info(f"开始训练MSZoo-PPO，环境: {args.env_name}")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建环境
    env = make_env_ppo(args.env_name)
    
    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    models_dir = os.path.join(args.save_dir, 'models')
    plots_dir = os.path.join(args.save_dir, 'plots')
    logs_dir = os.path.join(args.save_dir, 'logs')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # 配置文件日志处理器
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'ms_ppo_training.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 记录配置参数
    logger.info("MSZoo-PPO配置:")
    logger.info(f"  环境: {args.env_name}")
    logger.info(f"  总时间步数: {args.timesteps}")
    logger.info(f"  隐藏层维度: {args.hidden_dim}")
    logger.info(f"  Actor学习率: {args.actor_lr}")
    logger.info(f"  Critic学习率: {args.critic_lr}")
    logger.info(f"  批次大小: {args.batch_size}")
    logger.info(f"  PPO更新epochs: {args.ppo_epochs}")
    logger.info(f"  策略更新间隔: {args.policy_update_interval}")
    logger.info(f"  裁剪比率: {args.clip_ratio}")
    logger.info(f"  目标KL散度: {args.target_kl}")
    logger.info(f"  熵系数: {args.ent_coef}")
    logger.info(f"  状态归一化: {args.normalize_states}")
    logger.info(f"  并行训练: {args.parallel}")
    logger.info(f"  工作进程数: {args.n_workers}")
    logger.info("MSZoo配置:")
    logger.info(f"  扰动半径: {mszoo_config['perturbation_radii']}")
    logger.info(f"  群体大小: {mszoo_config['population_size']}")
    logger.info(f"  权重更新间隔: {mszoo_config['weight_update_interval']}")
    logger.info(f"  噪声标准差: {mszoo_config['noise_std']}")
    logger.info(f"  噪声衰减率: {mszoo_config['noise_decay']}")
    
    # 创建MSZooPPO实例
    ppo = MSZooPPO(
        env=env,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=args.clip_ratio,
        target_kl=args.target_kl,
        ent_coef=args.ent_coef,
        batch_size=args.batch_size,
        update_epochs=args.ppo_epochs,
        policy_update_interval=args.policy_update_interval,
        num_workers=args.n_workers if args.parallel else 1,
        normalize_states=args.normalize_states,
        mszoo_config=mszoo_config
    )
    
    # 如果指定了加载路径，加载模型
    if args.load_path:
        ppo.load_model(args.load_path)
        logger.info(f"从 {args.load_path} 加载模型")
    
    # 如果只进行测试，不训练
    if args.test_only:
        logger.info("仅进行测试，跳过训练")
        rewards, mean_reward = ppo.test(num_episodes=args.eval_episodes, render=args.render)
        logger.info(f"测试结果 - 平均回报: {mean_reward:.2f}")
        for i, r in enumerate(rewards):
            logger.info(f"  Episode {i+1}: {r[0]:.2f}")
        return
    
    # 训练模型
    start_time = time.time()
    
    if args.parallel:
        # 初始化并行环境
        if not ray.is_initialized():
            ray.init()
        ppo.init_workers(args.env_name)
        
    # 训练
    episode_rewards, mean_rewards = ppo.train(
        total_timesteps=args.timesteps,
        env_id=args.env_name if args.parallel else None,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_path=os.path.join(models_dir, 'ms_ppo')
    )
    
    training_time = time.time() - start_time
    logger.info(f"训练完成! 耗时: {training_time:.2f}秒")
    
    # 保存训练数据和曲线
    ppo.save_training_data()
    
    # 绘制奖励曲线
    ppo.plot_rewards(os.path.join(plots_dir, 'ms_ppo_rewards.png'))
    
    # 测试模型
    logger.info("开始测试最终模型...")
    if args.parallel:
        rewards, mean_reward = ppo.test_parallel(args.env_name, num_episodes=args.eval_episodes, render=args.render)
    else:
        rewards, mean_reward = ppo.test(num_episodes=args.eval_episodes, render=args.render)
    
    logger.info(f"测试结果 - 平均回报: {mean_reward:.2f}")
    
    return episode_rewards, mean_rewards

def train_mszoo_ddpg(args, mszoo_config):
    """训练MSZoo优化的DDPG算法"""
    logger.info(f"开始训练MSZoo-DDPG，环境: {args.env_name}")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建环境
    env = make_env_ddpg(args.env_name)
    
    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    models_dir = os.path.join(args.save_dir, 'models')
    plots_dir = os.path.join(args.save_dir, 'plots')
    logs_dir = os.path.join(args.save_dir, 'logs')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # 配置文件日志处理器
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'ms_ddpg_training.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 记录配置参数
    logger.info("MSZoo-DDPG配置:")
    logger.info(f"  环境: {args.env_name}")
    logger.info(f"  总时间步数: {args.timesteps}")
    logger.info(f"  隐藏层维度: {args.hidden_dim}")
    logger.info(f"  Actor学习率: {args.actor_lr}")
    logger.info(f"  Critic学习率: {args.critic_lr}")
    logger.info(f"  批次大小: {args.batch_size}")
    logger.info(f"  缓冲区大小: {args.buffer_size}")
    logger.info(f"  tau: {args.tau}")
    logger.info(f"  随机步数: {args.start_steps}")
    logger.info(f"  噪声theta: {args.noise_theta}")
    logger.info(f"  噪声sigma: {args.noise_sigma}")
    logger.info("MSZoo配置:")
    logger.info(f"  扰动半径: {mszoo_config['perturbation_radii']}")
    logger.info(f"  群体大小: {mszoo_config['population_size']}")
    logger.info(f"  权重更新间隔: {mszoo_config['weight_update_interval']}")
    logger.info(f"  噪声标准差: {mszoo_config['noise_std']}")
    logger.info(f"  噪声衰减率: {mszoo_config['noise_decay']}")
    
    # 创建MSZooDDPG实例
    ddpg = MSZooDDPG(
        env=env,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=0.99,
        tau=args.tau,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        noise_theta=args.noise_theta,
        noise_sigma=args.noise_sigma,
        mszoo_config=mszoo_config
    )
    
    # 如果指定了加载路径，加载模型
    if args.load_path:
        ddpg.load_model(args.load_path)
        logger.info(f"从 {args.load_path} 加载模型")
    
    # 如果只进行测试，不训练
    if args.test_only:
        logger.info("仅进行测试，跳过训练")
        rewards, mean_reward = ddpg.test(num_episodes=args.eval_episodes, render=args.render)
        logger.info(f"测试结果 - 平均回报: {mean_reward:.2f}")
        for i, r in enumerate(rewards):
            logger.info(f"  Episode {i+1}: {r:.2f}")
        return
    
    # 训练模型
    start_time = time.time()
    
    # 训练
    episode_rewards, mean_rewards = ddpg.train(
        total_timesteps=args.timesteps,
        start_steps=args.start_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_path=os.path.join(models_dir, 'ms_ddpg')
    )
    
    training_time = time.time() - start_time
    logger.info(f"训练完成! 耗时: {training_time:.2f}秒")
    
    # 绘制奖励曲线
    ddpg.plot_rewards(os.path.join(plots_dir, 'ms_ddpg_rewards.png'))
    
    # 测试模型
    logger.info("开始测试最终模型...")
    rewards, mean_reward = ddpg.test(num_episodes=args.eval_episodes, render=args.render)
    
    logger.info(f"测试结果 - 平均回报: {mean_reward:.2f}")
    
    return episode_rewards, mean_rewards

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 如果提供了配置文件，从中加载参数
    if args.config:
        config = load_config(args.config)
        logger.info(f"从配置文件 {args.config} 加载参数")
        args = update_args_with_config(args, config)
    
    # 设置环境变量
    setup_env_variables()
    
    # 设置默认保存目录（如果未指定）
    if args.save_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"results/{args.algorithm}_{args.env_name}_{timestamp}"
    
    # 创建MSZoo配置
    mszoo_config = create_mszoo_config(args)
    
    # 根据算法类型运行相应的训练函数
    if args.algorithm == 'ppo':
        train_mszoo_ppo(args, mszoo_config)
    elif args.algorithm == 'ddpg':
        train_mszoo_ddpg(args, mszoo_config)
    
    logger.info(f"所有操作完成。结果已保存到: {args.save_dir}")

if __name__ == "__main__":
    main() 