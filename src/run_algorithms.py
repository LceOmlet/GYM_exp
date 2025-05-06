"""
运行和比较不同的强化学习算法 (ARS, PPO, DDPG) 在同一个环境上。

使用方法:
1. 运行单个算法:
   - 标准PPO: python run_algorithms.py ppo --env_name HalfCheetah-v4
   - MSZoo优化PPO: python run_algorithms.py ppo --env_name HalfCheetah-v4 --use_mszoo
   
2. 比较多个算法:
   python run_algorithms.py --env_name HalfCheetah-v4 --algorithms ppo ddpg ms_ppo ms_ddpg ars

MSZoo优化器为PPO和DDPG实现了多尺度零阶优化方法，无需梯度更新Actor网络。
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
from copy import deepcopy
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

# 导入不同的算法实现
from src.models.ppo import PPO, make_env as make_env_ppo
from src.models.ddpg import DDPG, make_env as make_env_ddpg
# 导入MSZoo优化版本的算法
from src.models.ms_ppo import MSZooPPO
from src.models.ms_ddpg import MSZooDDPG
# import optimizers
from src.ars import ARSLearner, run_ars

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@ray.remote
class DDPGWorker:
    """用于DDPG并行收集经验和测试的工作进程"""
    
    def __init__(self, env_id, seed=0, hidden_dim=1024):
        """
        初始化工作进程
        
        参数:
            env_id: 环境ID
            seed: 随机种子
            hidden_dim: 隐藏层维度
        """
        # 为MuJoCo设置环境变量
        self._setup_mujoco_env()
        
        # 导入必要的类和函数
        from src.models.ddpg import make_env, ActorNetwork, CriticNetwork
        
        # 创建环境
        self.env = make_env(env_id)
        
        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.action_space.seed(seed)
        self.env.reset(seed=seed)
        
        # 保存环境相关信息
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = [self.env.action_space.low, self.env.action_space.high]
        
        # 保存隐藏层维度
        self.hidden_dim = hidden_dim
        
        # 保存工作进程ID
        self.worker_id = seed
        
        print(f"DDPG工作进程 {self.worker_id} 初始化完成，hidden_dim={self.hidden_dim}")
        
    def _setup_mujoco_env(self):
        """设置MuJoCo环境变量"""
        if "MUJOCO_PY_MUJOCO_PATH" not in os.environ:
            os.environ["MUJOCO_PY_MUJOCO_PATH"] = "/home/liangchen/.mujoco/mujoco210/"
        
        if "LD_LIBRARY_PATH" not in os.environ:
            os.environ["LD_LIBRARY_PATH"] = ""
        
        if "/home/liangchen/.mujoco/mujoco210/bin" not in os.environ["LD_LIBRARY_PATH"]:
            os.environ["LD_LIBRARY_PATH"] += ":/home/liangchen/.mujoco/mujoco210/bin"
        
        if "/usr/lib/nvidia" not in os.environ["LD_LIBRARY_PATH"]:
            os.environ["LD_LIBRARY_PATH"] += ":/usr/lib/nvidia"
            
    def collect_experiences(self, actor_weights, critic_weights, noise_scale=0.1, 
                           max_steps=1000, start_steps=1000, current_steps=0):
        """
        收集经验
        
        参数:
            actor_weights: Actor网络权重
            critic_weights: Critic网络权重
            noise_scale: 探索噪声比例
            max_steps: 最大步数
            start_steps: 初始随机步数
            current_steps: 当前全局步数
            
        返回:
            experiences: 收集的经验字典
        """
        try:
            # 创建本地Actor和Critic网络
            from src.models.ddpg import ActorNetwork, CriticNetwork
            
            # 创建网络实例，使用worker的hidden_dim
            actor = ActorNetwork(self.state_dim, self.action_dim, self.action_bound, hidden_dim=self.hidden_dim).to('cpu')
            critic = CriticNetwork(self.state_dim, self.action_dim, hidden_dim=self.hidden_dim).to('cpu')
            
            # 加载权重
            actor.load_state_dict(actor_weights)
            critic.load_state_dict(critic_weights)
            
            # 收集经验的变量
            experiences = []
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            total_steps = 0
            
            # 记录完整回合的信息
            episodes_completed = 0
            episode_rewards = []
            
            # 将max_steps改为更大的值，以便更有可能完成整个回合
            # 这样可以使得工作进程更可能完成回合，减少"未完成回合"的提示
            target_steps = max(1000, max_steps)  # 至少1000步
            
            while total_steps < target_steps:
                episode_steps += 1
                total_steps += 1
                
                # 选择动作
                if current_steps + total_steps < start_steps:
                    # 随机动作
                    action = self.env.action_space.sample()
                else:
                    # 根据策略选择动作并添加噪声
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        action = actor(state_tensor).detach().numpy()[0]
                        action = action + np.random.normal(0, noise_scale, size=action.shape)
                        # 裁剪动作到动作空间范围
                        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # 存储经验 (s, a, r, s', done)
                experiences.append((state, action, reward, next_state, float(done)))
                
                # 更新状态和回报
                state = next_state
                episode_reward += reward
                
                # 如果回合结束，重置环境
                if done:
                    # 记录完成的回合信息
                    episode_rewards.append(float(episode_reward))
                    episodes_completed += 1
                    
                    # 打印回合信息
                    print(f"DDPG工作进程 {self.worker_id} 完成回合，奖励: {episode_reward:.2f}, 步数: {episode_steps}")
                    
                    # 重置环境
                    state, _ = self.env.reset()
                    episode_reward = 0
                    episode_steps = 0
                    
                    # 如果已经收集了足够多的步数，可以提前结束循环
                    if total_steps >= max_steps:
                        break
            
            # 返回收集的经验及统计信息
            return {
                'experiences': experiences,
                'total_steps': total_steps,
                'episodes_completed': episodes_completed,
                'episode_rewards': episode_rewards,
                'ongoing_episode_reward': float(episode_reward) if episode_steps > 0 else None
            }
            
        except Exception as e:
            import traceback
            print(f"DDPG工作进程 {self.worker_id} 收集经验时出错: {e}")
            print(traceback.format_exc())
            return {
                'experiences': [],
                'total_steps': 0,
                'episodes_completed': 0,
                'episode_rewards': [],
                'error': str(e)
            }
    
    def test_policy(self, actor_weights, num_episodes=1, render=False):
        """
        测试策略
        
        参数:
            actor_weights: Actor网络权重
            num_episodes: 测试回合数
            render: 是否渲染
            
        返回:
            rewards: 每个回合的奖励
        """
        try:
            # 创建本地Actor网络
            from src.models.ddpg import ActorNetwork
            
            # 创建网络实例，使用worker的hidden_dim
            actor = ActorNetwork(self.state_dim, self.action_dim, self.action_bound, hidden_dim=self.hidden_dim).to('cpu')
            
            # 加载权重
            actor.load_state_dict(actor_weights)
            
            # 设置为评估模式
            actor.eval()
            
            # 测试策略
            total_rewards = []
            total_steps = 0
            
            for episode in range(num_episodes):
                episode_reward = 0
                state, _ = self.env.reset()
                done = False
                step_count = 0
                
                while not done:
                    if render:
                        self.env.render()
                    
                    # 选择动作
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        action = actor(state_tensor).detach().numpy()[0]
                    
                    # 执行动作
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    
                    # 更新状态和奖励
                    state = next_state
                    episode_reward += reward
                    step_count += 1
                    total_steps += 1
                
                total_rewards.append(episode_reward)
                print(f"DDPG工作进程 {self.worker_id} 测试回合 {episode+1}/{num_episodes}, 奖励: {episode_reward:.2f}, 步数: {step_count}")
            
            return total_rewards
            
        except Exception as e:
            import traceback
            print(f"DDPG工作进程 {self.worker_id} 测试策略时出错: {e}")
            print(traceback.format_exc())
            return [-1000.0] * num_episodes  # 返回一个负值列表，表示测试失败

def parse_args():
    """解析命令行参数"""
    import sys
    

    
    # 创建统一的参数解析器
    parser = argparse.ArgumentParser(description='强化学习算法训练与比较')
    
    # 配置文件参数
    parser.add_argument('--config', type=str, default=None,
                       help='YAML配置文件的路径')
    
    parser.add_argument('--direct_algo', type=str, default=None,
                        help='直接运行的算法')
    
    # 基本参数
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4',
                        help='OpenAI Gym环境名称')
    
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='训练的总时间步数')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    parser.add_argument('--n_workers', type=int, default=4,
                        help='并行工作进程数量')
                        
    parser.add_argument('--parallel_collection', action='store_true',
                        help='是否使用并行收集经验')
    
    parser.add_argument('--save_dir', type=str, default='results',
                        help='保存模型和结果的目录')
    
    parser.add_argument('--save_path', type=str, default=None, 
                        help='模型保存路径，如果为None则使用默认路径')
    
    parser.add_argument('--load_path', type=str, default=None, 
                        help='模型加载路径')
    
    parser.add_argument('--render', action='store_true', 
                        help='是否渲染测试')
    
    parser.add_argument('--log_interval', type=int, default=1,
                        help='日志打印间隔')
    
    parser.add_argument('--save_interval', type=int, default=100,
                        help='模型保存间隔')
    
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='评估轮数')
    
    # PPO专用参数
    parser.add_argument('--hidden_dim', type=int, default=1024,
                        help='隐藏层维度')
    
    parser.add_argument('--actor_lr', type=float, default=2e-4,
                        help='Actor学习率')
    
    parser.add_argument('--critic_lr', type=float, default=1e-3,
                        help='Critic学习率')
    
    parser.add_argument('--batch_size', type=int, default=512,
                        help='批次大小 (PPO: 512, DDPG: 100)')
    
    parser.add_argument('--ppo_epochs', type=int, default=5,
                        help='PPO每次更新的epoch数')
    
    parser.add_argument('--target_kl', type=float, default=0.01,
                        help='PPO目标KL散度')
    
    parser.add_argument('--ent_coef', type=float, default=0.01,
                        help='PPO熵正则化系数')
    
    parser.add_argument('--policy_update_interval', type=int, default=2048,
                        help='PPO策略更新间隔的环境步数')
    
    parser.add_argument('--normalize_states', action='store_true', default=True,
                        help='是否对状态进行归一化')
    
    # MSZoo选项 (仅在PPO直接运行模式下使用)
    parser.add_argument('--use_mszoo', action='store_true',
                        help='是否使用多尺度零阶优化器')
    
    
    # ARS特定参数
    parser.add_argument('--n_directions', type=int, default=16,
                        help='ARS的扰动方向数量')
    
    parser.add_argument('--deltas_used', type=int, default=16,
                        help='ARS使用的最优方向数量')
    
    parser.add_argument('--step_size', type=float, default=0.02,
                        help='步长/学习率')
    
    parser.add_argument('--delta_std', type=float, default=0.03,
                        help='ARS扰动的标准差')
    
    parser.add_argument('--optimizer_type', type=str, default='sgd',
                        choices=['sgd', 'adam', 'zero_order', 'multi_scale'],
                        help='ARS使用的优化器类型')
    
    parser.add_argument('--rollout_length', type=int, default=1000,
                    help='ARS的每次策略评估的最大步数')
    
    # DDPG特定参数
    parser.add_argument('--buffer_size', type=int, default=1000000,
                        help='DDPG的经验回放缓冲区大小')
    
    parser.add_argument('--start_steps', type=int, default=10000,
                        help='DDPG开始训练前的随机动作步数')
    
    parser.add_argument('--noise_scale', type=float, default=0.1,
                        help='DDPG探索噪声比例')
    
    parser.add_argument('--collection_interval', type=int, default=1000,
                        help='DDPG并行收集经验的步数间隔')
    
    args = parser.parse_args()
    
    # 设置默认的保存路径（如果未指定）
    if args.save_path is None:

        args.save_path = f'models/{args.direct_algo}'
    
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

def create_dirs(base_dir):
    """创建结果目录"""
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)

def run_ars_custom(args):
    """运行ARS算法"""
    logger.info("开始训练ARS...")
    
    # 确保初始化Ray
    if not ray.is_initialized():
        ray.init()
    
    # 确定保存路径
    dir_path = args.ars_save_path if hasattr(args, 'ars_save_path') else os.path.join(args.save_dir, 'ars')
    
    # 创建参数字典
    ars_params = {
        'env_name': args.env_name,
        'n_iter': args.timesteps // 1000,  # ARS以迭代为单位，大致估计
        'n_directions': args.n_directions,
        'deltas_used': args.deltas_used,
        'step_size': args.step_size,
        'delta_std': args.delta_std,
        'n_workers': args.n_workers,
        'rollout_length': args.rollout_length,  # 使用命令行参数
        'shift': 0,  # 默认为HalfCheetah
        'seed': args.seed,
        'policy_type': 'linear',
        'dir_path': dir_path,
        'filter': 'MeanStdFilter',  # 默认值
        'optimizer_type': args.optimizer_type
    }
    
    # 运行ARS
    start_time = time.time()
    run_ars(ars_params)
    training_time = time.time() - start_time
    
    # 读取训练结果
    try:
        progress_file = os.path.join(ars_params['dir_path'], 'progress.txt')
        data = np.genfromtxt(progress_file, names=True, skip_header=0,
                            dtype=None, deletechars='', encoding='utf-8')
        
        # 提取迭代和平均奖励
        iterations = data['Iteration']
        rewards = data['AverageReward']
        
        # 绘制回报曲线
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, rewards, label='ARS')
        plt.xlabel('Iteration')
        plt.ylabel('Average Reward')
        plt.title(f'ARS Training Rewards on {args.env_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.save_dir, 'plots', 'ars_rewards.png'))
        plt.close()
        
        logger.info(f"ARS训练完成。训练时间: {training_time:.2f}秒, 最终奖励: {rewards[-1]:.2f}")
        
        return {
            'algorithm': 'ARS',
            'final_reward': rewards[-1],
            'training_time': training_time,
            'rewards': rewards,
            'iterations': iterations
        }
    except Exception as e:
        logger.error(f"读取ARS结果失败: {e}")
        return {
            'algorithm': 'ARS',
            'final_reward': 0,
            'training_time': training_time,
            'rewards': [],
            'iterations': []
        }

def run_ppo(env_name, timesteps, seed, save_path, **kwargs):
    """运行PPO算法"""
    start_time = time.time()
    
    # 获取命令行参数或使用默认值
    hidden_dim = kwargs.get('hidden_dim', 1024)
    actor_lr = kwargs.get('actor_lr', 2e-4)
    critic_lr = kwargs.get('critic_lr', 1e-3)
    policy_update_interval = kwargs.get('policy_update_interval', 2048)
    batch_size = kwargs.get('batch_size', 512)
    ppo_epochs = kwargs.get('ppo_epochs', 5)
    ent_coef = kwargs.get('ent_coef', 0.01)
    target_kl = kwargs.get('target_kl', 0.01)
    normalize_states = kwargs.get('normalize_states', True)
    
    # 是否使用MSZoo优化器
    use_mszoo = kwargs.get('use_mszoo', False)
    mszoo_config = kwargs.get('mszoo_config', None)
    
    # 创建环境
    env = make_env_ppo(env_name)
    
    # 创建PPO算法实例 - 根据是否使用MSZoo决定使用哪个类
    if use_mszoo:
        logger.info("使用多尺度零阶优化的PPO")
        ppo = MSZooPPO(
            env=env,
            hidden_dim=hidden_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            clip_ratio=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            target_kl=target_kl,
            ent_coef=ent_coef,
            batch_size=batch_size, 
            update_epochs=ppo_epochs,
            policy_update_interval=policy_update_interval,
            normalize_states=normalize_states,
            mszoo_config=mszoo_config
        )
    else:
        logger.info("使用标准PPO")
        ppo = PPO(
            env=env,
            hidden_dim=hidden_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            clip_ratio=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            target_kl=target_kl,
            ent_coef=ent_coef,
            batch_size=batch_size, 
            update_epochs=ppo_epochs,
            policy_update_interval=policy_update_interval,
            normalize_states=normalize_states
        )
    
    # 训练PPO
    rewards, mean_returns = ppo.train(timesteps)
    
    # 记录训练信息
    train_time = time.time() - start_time
    print(f"PPO训练完成！耗时: {train_time:.2f}秒")
    print(f"最终奖励: {(rewards[-1] if rewards else 0):.2f}")
    
    # 评估训练后的模型
    eval_rewards = []
    for i in range(10):  # 评估10个回合
        eval_reward = ppo.test(1)[0]
        eval_rewards.append(eval_reward)
        print(f"评估回合 {i+1}: 奖励 = {eval_reward[0]:.2f}")
    
    print(f"平均评估奖励: {np.mean(eval_rewards):.2f}")
    
    # 保存模型和奖励曲线
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ppo.save_model(save_path)
    
    # 绘制奖励曲线并保存
    plot_dir = os.path.join(os.path.dirname(save_path), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{'mszoo_' if use_mszoo else ''}ppo_rewards.png")
    ppo.plot_rewards(plot_path)
    
    return rewards, eval_rewards

def run_ppo_parallel(env_name, timesteps, seed, save_path, n_workers=4, **kwargs):
    """运行并行版本的PPO算法"""
    start_time = time.time()
    
    # 获取命令行参数或使用默认值
    hidden_dim = kwargs.get('hidden_dim', 1024)
    actor_lr = kwargs.get('actor_lr', 2e-4)
    critic_lr = kwargs.get('critic_lr', 1e-3)
    policy_update_interval = kwargs.get('policy_update_interval', 2048)
    batch_size = kwargs.get('batch_size', 512)
    ppo_epochs = kwargs.get('ppo_epochs', 5)
    ent_coef = kwargs.get('ent_coef', 0.01)
    target_kl = kwargs.get('target_kl', 0.01)
    normalize_states = kwargs.get('normalize_states', True)
    
    # 是否使用MSZoo优化器
    use_mszoo = kwargs.get('use_mszoo', False)
    mszoo_config = kwargs.get('mszoo_config', None)
    
    # 创建环境
    env = make_env_ppo(env_name)
    
    # 创建PPO算法实例 - 根据是否使用MSZoo决定使用哪个类
    if use_mszoo:
        logger.info("使用多尺度零阶优化的并行PPO")
        ppo = MSZooPPO(
            env=env,
            hidden_dim=hidden_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            clip_ratio=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            target_kl=target_kl,
            ent_coef=ent_coef,
            batch_size=batch_size, 
            update_epochs=ppo_epochs,
            policy_update_interval=policy_update_interval,
            normalize_states=normalize_states,
            num_workers=n_workers,
            mszoo_config=mszoo_config
        )
    else:
        logger.info("使用标准并行PPO")
        ppo = PPO(
            env=env,
            hidden_dim=hidden_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            clip_ratio=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            target_kl=target_kl,
            ent_coef=ent_coef,
            batch_size=batch_size, 
            update_epochs=ppo_epochs,
            policy_update_interval=policy_update_interval,
            normalize_states=normalize_states,
            num_workers=n_workers
        )
    
    # 初始化并行工作进程
    ppo.init_workers(env_name)
    
    # 训练PPO
    rewards, mean_returns = ppo.train(timesteps)
    
    # 记录训练信息
    train_time = time.time() - start_time
    print(f"并行PPO训练完成！耗时: {train_time:.2f}秒")
    print(f"最终奖励: {(rewards[-1] if rewards else 0):.2f}")
    
    # 评估训练后的模型
    eval_rewards = []
    for i in range(10):  # 评估10个回合
        eval_reward = ppo.test(1)[0]
        eval_rewards.append(eval_reward)
        print(f"评估回合 {i+1}: 奖励 = {eval_reward[0]:.2f}")
    
    print(f"平均评估奖励: {np.mean(eval_rewards):.2f}")
    
    # 保存模型和奖励曲线
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ppo.save_model(save_path)
    
    # 绘制奖励曲线并保存
    plot_dir = os.path.join(os.path.dirname(save_path), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{'mszoo_' if use_mszoo else ''}ppo_parallel_rewards.png")
    ppo.plot_rewards(plot_path)
    
    return rewards, eval_rewards

def run_ddpg(args):
    """运行DDPG算法"""
    logger.info("开始训练DDPG...")
    
    # 创建环境
    env = make_env_ddpg(args.env_name)
    
    # 创建DDPG实例
    ddpg = DDPG(env, 
                hidden_dim=256,
                actor_lr=1e-4,
                critic_lr=1e-3,
                gamma=0.99, 
                tau=0.005,
                batch_size=64,
                buffer_size=args.buffer_size)
    
    # 设置保存路径
    save_path = args.ddpg_save_path if hasattr(args, 'ddpg_save_path') else os.path.join(args.save_dir, 'models', 'ddpg')
    plot_path = os.path.join(os.path.dirname(save_path), 'plots', 'ddpg_rewards.png')
    
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    # 设置日志目录为args.save_dir，确保rewards.csv被记录到正确的位置
    ddpg.setup_logging(args.save_dir)
    
    # 检查CSV文件是否被创建
    rewards_csv_path = os.path.join(args.save_dir, 'data', 'rewards.csv')
    if os.path.exists(rewards_csv_path):
        logger.info(f"已创建rewards.csv文件: {rewards_csv_path}")
    else:
        logger.warning(f"rewards.csv文件未创建: {rewards_csv_path}")
    
    # 如果指定了加载路径，加载模型
    if args.load_path:
        ddpg.load_model(args.load_path)
        logger.info(f"从 {args.load_path} 加载模型")
    
    # 训练模型
    start_time = time.time()
    episode_rewards, mean_rewards = ddpg.train(
        total_timesteps=args.timesteps, 
        start_steps=args.start_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_path=save_path
    )
    training_time = time.time() - start_time
    
    # 绘制回报曲线
    ddpg.plot_rewards(save_path=plot_path)
    
    # 评估模型
    eval_rewards, mean_eval_reward = ddpg.test(num_episodes=args.eval_episodes, render=args.render)
    
    logger.info(f"DDPG训练完成。训练时间: {training_time:.2f}秒, 最终奖励: {(mean_rewards[-1] if mean_rewards else 0):.2f}, 评估奖励: {mean_eval_reward:.2f}")
    
    return {
        'algorithm': 'DDPG',
        'final_reward': mean_rewards[-1] if mean_rewards else 0,
        'eval_reward': mean_eval_reward,
        'training_time': training_time,
        'rewards': episode_rewards,
        'mean_rewards': mean_rewards
    }

def run_ddpg_parallel(args):
    """运行并行化的DDPG算法"""
    logger.info("开始并行训练DDPG...")
    
    # 设置日志级别为INFO，确保debug日志不会显示
    logger.setLevel(logging.INFO)
    
    # 确保Ray已初始化
    if not ray.is_initialized():
        ray.init()
    
    # 创建环境来获取环境信息（用于创建模型）
    env = make_env_ddpg(args.env_name)
    
    # 获取隐藏层维度 - 使用参数中的hidden_dim或默认值256
    hidden_dim = args.hidden_dim if hasattr(args, 'hidden_dim') else 256
    
    # 创建DDPG实例
    ddpg = DDPG(env, 
                hidden_dim=hidden_dim,
                actor_lr=1e-4,
                critic_lr=1e-3,
                gamma=0.99, 
                tau=0.005,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size)
    
    # 设置保存路径
    save_path = args.ddpg_save_path if hasattr(args, 'ddpg_save_path') else os.path.join(args.save_dir, 'models', 'ddpg_parallel')
    plot_path = os.path.join(os.path.dirname(save_path), 'plots', 'ddpg_parallel_rewards.png')
    
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    # 设置日志目录为args.save_dir，确保rewards.csv被记录到正确的位置
    ddpg.setup_logging(args.save_dir)
    
    # 检查CSV文件是否被创建
    rewards_csv_path = os.path.join(args.save_dir, 'data', 'rewards.csv')
    if os.path.exists(rewards_csv_path):
        logger.info(f"已创建rewards.csv文件: {rewards_csv_path}")
    else:
        logger.warning(f"rewards.csv文件未创建: {rewards_csv_path}")
    
    # 如果指定了加载路径，加载模型
    if args.load_path:
        ddpg.load_model(args.load_path)
        logger.info(f"从 {args.load_path} 加载模型")
    
    # 创建工作进程，传递hidden_dim参数
    workers = [DDPGWorker.remote(args.env_name, seed=i, hidden_dim=hidden_dim) for i in range(args.n_workers)]
    logger.info(f"已创建 {args.n_workers} 个DDPG工作进程，hidden_dim={hidden_dim}")
    
    # M参数
    total_steps = 0
    max_steps = args.timesteps
    collection_interval = args.collection_interval
    episode_rewards = []
    mean_rewards = []
    update_count = 0
    
    # 预先初始化损失变量为空列表，而不是deque
    policy_losses = []
    value_losses = []
    
    # 用于计算平均回报的队列
    reward_queue = deque(maxlen=100)
    
    # 记录训练开始时间
    start_time = time.time()
    
    # 主训练循环
    while total_steps < max_steps:
        # 获取当前权重
        actor_weights = {k: v.cpu() for k, v in ddpg.actor.state_dict().items()}
        critic_weights = {k: v.cpu() for k, v in ddpg.critic.state_dict().items()}
        
        # 并行收集经验
        collection_start = time.time()
        # 增加每个工作进程收集的步数，使工作进程可以完成整个回合
        steps_per_worker = max(collection_interval // args.n_workers, 1000)  # 至少1000步
        
        collection_ids = [worker.collect_experiences.remote(
            actor_weights, critic_weights,
            noise_scale=args.noise_scale,
            max_steps=steps_per_worker,
            start_steps=args.start_steps,
            current_steps=total_steps
        ) for worker in workers]
        
        # 等待所有worker完成
        results = ray.get(collection_ids)
        collection_time = time.time() - collection_start
        
        # 处理收集的经验
        collected_steps = 0
        completed_episodes = 0
        for result in results:
            # 添加经验到经验回放缓冲区
            for exp in result['experiences']:
                ddpg.replay_buffer.add(exp[0], exp[1], exp[2], exp[3], exp[4])
            
            # 更新步数统计
            collected_steps += result['total_steps']
            completed_episodes += result.get('episodes_completed', 0)
            
            # 记录回报
            if 'episode_rewards' in result and result['episode_rewards']:
                # 添加所有完成的回合奖励
                for episode_reward in result['episode_rewards']:
                    episode_rewards.append(episode_reward)
                    reward_queue.append(episode_reward)
                    
                    # 计算平均回报
                    if len(reward_queue) > 0:
                        mean_reward = np.mean(reward_queue)
                        mean_rewards.append(mean_reward)
                        
                        # 记录数据到CSV文件
                        actor_loss = np.mean(policy_losses) if policy_losses else 0
                        critic_loss = np.mean(value_losses) if value_losses else 0
                        ddpg.log_episode_data(len(episode_rewards), total_steps, episode_reward, mean_reward, actor_loss, critic_loss, 0, 0)
                        
                        # 每个log_interval打印一次信息
                        if len(episode_rewards) % args.log_interval == 0:
                            logger.info(f"Episode {len(episode_rewards)} | Steps: {total_steps} | " +
                                      f"Reward: {episode_reward:.2f} | Mean Reward: {mean_reward:.2f}")
            
            # 如果有未完成的回合，记录其信息但不计入统计
            # 只在debug级别打印未完成回合信息，以减少日志噪音
            if 'ongoing_episode_reward' in result and result['ongoing_episode_reward'] is not None:
                ongoing_reward = result['ongoing_episode_reward']
                logger.debug(f"工作进程有未完成回合，当前奖励: {ongoing_reward:.2f}")
                
            # 记录错误信息（如果有）
            if 'error' in result and result['error']:
                logger.warning(f"工作进程报告错误: {result['error']}")
        
        # 打印收集统计信息
        logger.info(f"完成{completed_episodes}个回合，收集{collected_steps}步数据")
        
        # 更新总步数
        total_steps += collected_steps
        
        # 更新策略（多次）
        update_start = time.time()
        
        # 增加更新次数，确保模型充分学习
        # 每收集32步更新一次，每次更新最多批次为1000
        update_count_per_collection = min(collected_steps, 2000)  # 增加更新次数上限
        update_interval = max(1, update_count_per_collection // 200)  # 减小更新间隔，每次收集增加更新频率
        
        # 清空本次收集的损失值记录
        policy_losses = []
        value_losses = []
        
        logger.info(f"执行{update_count_per_collection // update_interval}次策略更新")
        
        for i in range(0, update_count_per_collection, update_interval):
            # 确保缓冲区中有足够的样本
            if len(ddpg.replay_buffer) > args.batch_size:
                critic_loss, actor_loss = ddpg.update()  # 使用update而不是update_policy
                policy_losses.append(actor_loss)
                value_losses.append(critic_loss)
                update_count += 1
        
        update_time = time.time() - update_start
        
        # 计算平均损失
        avg_policy_loss = np.mean(policy_losses) if policy_losses else 0
        avg_value_loss = np.mean(value_losses) if value_losses else 0
        
        logger.info(f"收集步数: {collected_steps} | 缓冲区大小: {len(ddpg.replay_buffer)} | " +
                   f"平均策略损失: {avg_policy_loss:.6f} | 平均值函数损失: {avg_value_loss:.6f}")
        logger.info(f"收集时间: {collection_time:.2f}秒 | 更新时间: {update_time:.2f}秒")
        
        # 每隔一定步数评估一次策略并保存模型
        if update_count % args.save_interval == 0:
            # 并行评估策略
            eval_rewards = test_ddpg_parallel(ddpg, args.env_name, args.n_workers, args.eval_episodes)
            mean_eval_reward = np.mean(eval_rewards)
            logger.info(f"评估 | 平均回报: {mean_eval_reward:.2f}")
            
            # 保存模型
            if save_path:
                model_path = os.path.join(save_path, f"ddpg_model_{total_steps}.pt")
                # 确保模型保存目录存在
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                ddpg.save_model(model_path)
                logger.info(f"模型已保存到 {model_path}")
    
    # 训练结束，记录总时间
    training_time = time.time() - start_time
    logger.info(f"DDPG并行训练完成。训练时间: {training_time:.2f}秒, 总步数: {total_steps}")
    
    # 保存最终模型
    if save_path:
        final_model_path = os.path.join(save_path, "ddpg_model_final.pt")
        # 确保模型保存目录存在
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        ddpg.save_model(final_model_path)
    
    # 绘制回报曲线
    if plot_path and episode_rewards:
        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards, label='Episode Rewards', alpha=0.3)
        plt.plot(mean_rewards, label='Mean 100 Rewards', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('DDPG Training Rewards')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_path)
        plt.close()
    
    # 最终评估
    final_eval_rewards = test_ddpg_parallel(ddpg, args.env_name, args.n_workers, args.eval_episodes)
    final_mean_reward = np.mean(final_eval_rewards)
    
    return {
        'algorithm': 'DDPG (Parallel)',
        'final_reward': mean_rewards[-1] if mean_rewards else 0,
        'eval_reward': final_mean_reward,
        'training_time': training_time,
        'rewards': episode_rewards,
        'mean_rewards': mean_rewards
    }

def test_ddpg_parallel(ddpg, env_id, num_workers, num_episodes):
    """并行测试DDPG策略"""
    # 创建临时工作进程（如果需要）
    workers = [DDPGWorker.remote(env_id, seed=i) for i in range(num_workers)]
    
    # 获取Actor网络权重，确保在CPU上
    actor_weights = {k: v.cpu() for k, v in ddpg.actor.state_dict().items()}
    
    # 每个工作进程测试的回合数
    episodes_per_worker = max(1, num_episodes // num_workers)
    remaining_episodes = num_episodes % num_workers
    
    # 并行测试
    test_ids = []
    for i, worker in enumerate(workers):
        # 最后一个工作进程处理剩余的回合
        worker_episodes = episodes_per_worker + (remaining_episodes if i == len(workers) - 1 else 0)
        if worker_episodes > 0:
            test_ids.append(worker.test_policy.remote(actor_weights, num_episodes=worker_episodes))
    
    # 等待所有工作进程完成
    results = ray.get(test_ids)
    
    # 合并结果
    all_rewards = [reward for worker_rewards in results for reward in worker_rewards]
    
    return all_rewards

def plot_comparison(results, args):
    """绘制不同算法的比较图"""
    plt.figure(figsize=(15, 10))
    
    # 绘制训练回报曲线
    plt.subplot(2, 1, 1)
    for result in results:
        if result['algorithm'] == 'ARS' and 'iterations' in result and len(result['iterations']) > 0:
            plt.plot(result['iterations'], result['rewards'], label=f"ARS ({args.optimizer_type})")
        elif 'mean_rewards' in result and len(result['mean_rewards']) > 0:
            x = np.arange(len(result['mean_rewards']))
            plt.plot(x, result['mean_rewards'], label=result['algorithm'])
    
    plt.xlabel('Iterations/Episodes')
    plt.ylabel('Average Reward')
    plt.title(f'Comparison of RL Algorithms on {args.env_name}')
    plt.legend()
    plt.grid(True)
    
    # 绘制最终性能和训练时间的柱状图
    plt.subplot(2, 1, 2)
    algorithms = [result['algorithm'] for result in results]
    final_rewards = [result.get('final_reward', 0) for result in results]
    training_times = [result.get('training_time', 0) / 60.0 for result in results]  # 转换为分钟
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    ax1 = plt.gca()
    ax1.bar(x - width/2, final_rewards, width, label='Final Reward')
    
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, training_times, width, label='Training Time (min)', color='orange')
    
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Final Average Reward')
    ax2.set_ylabel('Training Time (minutes)')
    
    plt.title('Final Performance and Training Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms)
    
    # 合并两个轴的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'plots', 'algorithm_comparison.png'))
    plt.close()

def configure_logging(args):
    """配置日志"""
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建文件处理器
    log_dir = os.path.join(args.save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"日志配置完成，将写入到 {log_file}")

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

def run_ms_ddpg(args):
    """运行多尺度零阶优化版本的DDPG算法"""
    logger.info("开始训练多尺度零阶优化DDPG...")
    
    # 创建环境
    env = make_env_ddpg(args.env_name)
    
    # MSZoo配置
    mszoo_config = getattr(args, 'mszoo_config', {})
    
    # 创建MSZooDDPG实例
    ddpg = MSZooDDPG(env, 
                hidden_dim=1024,  # 使用与PPO相同的hidden_dim值
                actor_lr=args.actor_lr if hasattr(args, 'actor_lr') else 1e-4,
                critic_lr=args.critic_lr if hasattr(args, 'critic_lr') else 1e-3,
                gamma=0.99, 
                tau=0.005,
                batch_size=args.batch_size if hasattr(args, 'batch_size') else 64,
                buffer_size=args.buffer_size if hasattr(args, 'buffer_size') else 1000000,
                mszoo_config=mszoo_config)
    
    # 设置保存路径
    save_path = args.ms_ddpg_save_path if hasattr(args, 'ms_ddpg_save_path') else os.path.join(args.save_dir, 'models', 'ms_ddpg')
    plot_path = os.path.join(os.path.dirname(save_path), 'plots', 'ms_ddpg_rewards.png')
    
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    # 设置日志目录为args.save_dir，确保rewards.csv被记录到正确的位置
    ddpg.setup_logging(args.save_dir)
    
    # 检查CSV文件是否被创建
    rewards_csv_path = os.path.join(args.save_dir, 'data', 'rewards.csv')
    if os.path.exists(rewards_csv_path):
        logger.info(f"已创建rewards.csv文件: {rewards_csv_path}")
    else:
        logger.warning(f"rewards.csv文件未创建: {rewards_csv_path}")
    
    # 如果指定了加载路径，加载模型
    if hasattr(args, 'load_path') and args.load_path:
        ddpg.load_model(args.load_path)
        logger.info(f"从 {args.load_path} 加载模型")
    
    # 训练模型
    start_time = time.time()
    episode_rewards, mean_rewards = ddpg.train(
        total_timesteps=args.timesteps, 
        start_steps=args.start_steps if hasattr(args, 'start_steps') else 10000,
        log_interval=args.log_interval if hasattr(args, 'log_interval') else 10,
        save_interval=args.save_interval if hasattr(args, 'save_interval') else 100,
        save_path=save_path
    )
    training_time = time.time() - start_time
    
    # 绘制回报曲线
    ddpg.plot_rewards(save_path=plot_path)
    
    # 评估模型
    eval_episodes = args.eval_episodes if hasattr(args, 'eval_episodes') else 10
    render = args.render if hasattr(args, 'render') else False
    eval_rewards, mean_eval_reward = ddpg.test(num_episodes=eval_episodes, render=render)
    
    logger.info(f"MSZooDDPG训练完成。训练时间: {training_time:.2f}秒, 最终奖励: {(mean_rewards[-1] if mean_rewards else 0):.2f}, 评估奖励: {mean_eval_reward:.2f}")
    
    return {
        'algorithm': 'MSZoo-DDPG',
        'final_reward': mean_rewards[-1] if mean_rewards else 0,
        'eval_reward': mean_eval_reward,
        'training_time': training_time,
        'rewards': episode_rewards,
        'mean_rewards': mean_rewards
    }

def run_ms_ddpg_parallel(args):
    """运行并行化的多尺度零阶优化DDPG算法"""
    logger.info("开始并行训练MSZooDDPG...")
    
    # 设置日志级别为INFO，确保debug日志不会显示
    logger.setLevel(logging.INFO)
    
    # 确保Ray已初始化
    if not ray.is_initialized():
        ray.init()
    
    # 创建环境来获取环境信息（用于创建模型）
    env = make_env_ddpg(args.env_name)
    
    # MSZoo配置
    mszoo_config = getattr(args, 'mszoo_config', {})
    
    # 获取隐藏层维度 - 使用参数中的hidden_dim或默认值1024
    hidden_dim = args.hidden_dim if hasattr(args, 'hidden_dim') else 1024
    
    # 创建MSZooDDPG实例
    ddpg = MSZooDDPG(env, 
                hidden_dim=hidden_dim,  # 使用从参数获取的hidden_dim
                actor_lr=args.actor_lr if hasattr(args, 'actor_lr') else 1e-4,
                critic_lr=args.critic_lr if hasattr(args, 'critic_lr') else 1e-3,
                gamma=0.99, 
                tau=0.005,
                batch_size=args.batch_size if hasattr(args, 'batch_size') else 64,
                buffer_size=args.buffer_size if hasattr(args, 'buffer_size') else 1000000,
                mszoo_config=mszoo_config)
    
    # 设置保存路径
    save_path = args.ms_ddpg_save_path if hasattr(args, 'ms_ddpg_save_path') else os.path.join(args.save_dir, 'models', 'ms_ddpg_parallel')
    plot_path = os.path.join(os.path.dirname(save_path), 'plots', 'ms_ddpg_parallel_rewards.png')
    
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    # 设置日志目录为args.save_dir，确保rewards.csv被记录到正确的位置
    ddpg.setup_logging(args.save_dir)
    
    # 检查CSV文件是否被创建
    rewards_csv_path = os.path.join(args.save_dir, 'data', 'rewards.csv')
    if os.path.exists(rewards_csv_path):
        logger.info(f"已创建rewards.csv文件: {rewards_csv_path}")
    else:
        logger.warning(f"rewards.csv文件未创建: {rewards_csv_path}")
    
    # 如果指定了加载路径，加载模型
    if hasattr(args, 'load_path') and args.load_path:
        ddpg.load_model(args.load_path)
        logger.info(f"从 {args.load_path} 加载模型")
    
    # 创建工作进程 - 重用DDPGWorker，传递hidden_dim参数
    workers = [DDPGWorker.remote(args.env_name, seed=i, hidden_dim=hidden_dim) for i in range(args.n_workers)]
    logger.info(f"已创建 {args.n_workers} 个DDPG工作进程，hidden_dim={hidden_dim}")
    
    # 训练参数
    total_steps = 0
    max_steps = args.timesteps
    collection_interval = args.collection_interval
    episode_rewards = []
    mean_rewards = []
    update_count = 0
    
    # 预先初始化损失变量为空列表
    policy_losses = []
    value_losses = []
    entropy_values = []
    kl_values = []
    
    # 用于计算平均回报的队列
    reward_queue = deque(maxlen=100)
    
    # 记录训练开始时间
    start_time = time.time()
    
    # 主训练循环
    while total_steps < max_steps:
        # 获取当前权重
        actor_weights = {k: v.cpu() for k, v in ddpg.actor.state_dict().items()}
        critic_weights = {k: v.cpu() for k, v in ddpg.critic.state_dict().items()}
        
        # 并行收集经验
        collection_start = time.time()
        # 增加每个工作进程收集的步数，使工作进程可以完成整个回合
        steps_per_worker = max(collection_interval // args.n_workers, 1000)  # 至少1000步
        
        collection_ids = [worker.collect_experiences.remote(
            actor_weights, critic_weights,
            noise_scale=args.noise_scale,
            max_steps=steps_per_worker,
            start_steps=args.start_steps,
            current_steps=total_steps
        ) for worker in workers]
        
        # 等待所有worker完成
        results = ray.get(collection_ids)
        collection_time = time.time() - collection_start
        
        # 处理收集的经验
        collected_steps = 0
        completed_episodes = 0
        for result in results:
            # 添加经验到经验回放缓冲区
            for exp in result['experiences']:
                ddpg.replay_buffer.add(exp[0], exp[1], exp[2], exp[3], exp[4])
            
            # 更新步数统计
            collected_steps += result['total_steps']
            completed_episodes += result.get('episodes_completed', 0)
            
            # 记录回报
            if 'episode_rewards' in result and result['episode_rewards']:
                # 添加所有完成的回合奖励
                for episode_reward in result['episode_rewards']:
                    episode_rewards.append(episode_reward)
                    reward_queue.append(episode_reward)
                    
                    # 计算平均回报
                    if len(reward_queue) > 0:
                        mean_reward = np.mean(reward_queue)
                        mean_rewards.append(mean_reward)
                        
                        # 记录数据到CSV文件
                        policy_loss = np.mean(policy_losses) if policy_losses else 0
                        value_loss = np.mean(value_losses) if value_losses else 0
                        entropy = np.mean(entropy_values) if entropy_values else 0
                        kl = np.mean(kl_values) if kl_values else 0
                        
                        ddpg.log_episode_data(len(episode_rewards), total_steps, episode_reward, mean_reward, policy_loss, value_loss, kl, entropy)
                        
                        # 每个log_interval打印一次信息
                        if len(episode_rewards) % args.log_interval == 0:
                            logger.info(f"Episode {len(episode_rewards)} | Steps: {total_steps} | " +
                                      f"Reward: {episode_reward:.2f} | Mean Reward: {mean_reward:.2f}")
                            
                            if hasattr(ddpg.ms_optimizer, 'radius_weights'):
                                logger.info(f"MSZoo权重: {ddpg.ms_optimizer.radius_weights}, 噪声标准差: {ddpg.ms_optimizer.noise_std}")
                                logger.info(f"策略损失: {policy_loss:.6f} | 值函数损失: {value_loss:.6f} | KL: {kl:.6f} | 熵: {entropy:.6f}")
            
            # 如果有未完成的回合，记录其信息但不计入统计
            # 只在debug级别打印未完成回合信息，以减少日志噪音
            if 'ongoing_episode_reward' in result and result['ongoing_episode_reward'] is not None:
                ongoing_reward = result['ongoing_episode_reward']
                logger.debug(f"工作进程有未完成回合，当前奖励: {ongoing_reward:.2f}")
                
            # 记录错误信息（如果有）
            if 'error' in result and result['error']:
                logger.warning(f"工作进程报告错误: {result['error']}")
        
        # 打印收集统计信息
        logger.info(f"完成{completed_episodes}个回合，收集{collected_steps}步数据")
        
        # 更新总步数
        total_steps += collected_steps
        
        # 更新策略（多次）
        update_start = time.time()
        
        # 增加更新次数，确保模型充分学习
        update_count_per_collection = min(collected_steps, 2000)  # 增加更新次数上限
        update_interval = max(1, update_count_per_collection // 200)  # 减小更新间隔，每次收集增加更新频率
        
        # 清空本次收集的损失值记录
        policy_losses = []
        value_losses = []
        entropy_values = []
        kl_values = []
        
        logger.info(f"执行{update_count_per_collection // update_interval}次策略更新")
        
        for i in range(0, update_count_per_collection, update_interval):
            # 确保缓冲区中有足够的样本
            if len(ddpg.replay_buffer) > args.batch_size:
                critic_loss, policy_loss, entropy, kl = ddpg.update()
                policy_losses.append(policy_loss)
                value_losses.append(critic_loss)
                entropy_values.append(entropy)
                kl_values.append(kl)
                update_count += 1
        
        update_time = time.time() - update_start
        
        # 计算平均损失
        avg_policy_loss = np.mean(policy_losses) if policy_losses else 0
        avg_value_loss = np.mean(value_losses) if value_losses else 0
        avg_entropy = np.mean(entropy_values) if entropy_values else 0
        avg_kl = np.mean(kl_values) if kl_values else 0
        
        logger.info(f"收集步数: {collected_steps} | 缓冲区大小: {len(ddpg.replay_buffer)} | " +
                   f"平均策略损失: {avg_policy_loss:.6f} | 平均值函数损失: {avg_value_loss:.6f} | " +
                   f"平均熵: {avg_entropy:.6f} | 平均KL: {avg_kl:.6f}")
        logger.info(f"收集时间: {collection_time:.2f}秒 | 更新时间: {update_time:.2f}秒")
        
        # 每隔一定步数评估一次策略并保存模型
        if update_count % args.save_interval == 0:
            # 并行评估策略
            eval_rewards = test_ddpg_parallel(ddpg, args.env_name, args.n_workers, args.eval_episodes)
            mean_eval_reward = np.mean(eval_rewards)
            logger.info(f"评估 | 平均回报: {mean_eval_reward:.2f}")
            
            # 保存模型
            if save_path:
                model_path = os.path.join(save_path, f"ms_ddpg_model_{total_steps}.pt")
                # 确保模型保存目录存在
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                ddpg.save_model(model_path)
                logger.info(f"模型已保存到 {model_path}")
    
    # 训练结束，记录总时间
    training_time = time.time() - start_time
    logger.info(f"MSZooDDPG并行训练完成。训练时间: {training_time:.2f}秒, 总步数: {total_steps}")
    
    # 保存最终模型
    if save_path:
        final_model_path = os.path.join(save_path, "ms_ddpg_model_final.pt")
        # 确保模型保存目录存在
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        ddpg.save_model(final_model_path)
        
        # 保存MSZoo配置
        with open(os.path.join(save_path, "ms_ddpg_config.json"), 'w') as f:
            import json
            json.dump({
                'mszoo_config': mszoo_config,
                'final_weights': ddpg.ms_optimizer.radius_weights if hasattr(ddpg.ms_optimizer, 'radius_weights') else None,
                'noise_std': ddpg.ms_optimizer.noise_std if hasattr(ddpg.ms_optimizer, 'noise_std') else None,
                'optimizer_count': ddpg.ms_optimizer._count if hasattr(ddpg.ms_optimizer, '_count') else 0
            }, f, indent=4)
    
    # 绘制回报曲线
    if plot_path and episode_rewards:
        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards, label='Episode Rewards', alpha=0.3)
        plt.plot(mean_rewards, label='Mean 100 Rewards', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('MSZooDDPG Training Rewards')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_path)
        plt.close()
    
    # 最终评估
    final_eval_rewards = test_ddpg_parallel(ddpg, args.env_name, args.n_workers, args.eval_episodes)
    final_mean_reward = np.mean(final_eval_rewards)
    
    return {
        'algorithm': 'MSZoo-DDPG (Parallel)',
        'final_reward': mean_rewards[-1] if mean_rewards else 0,
        'eval_reward': final_mean_reward,
        'training_time': training_time,
        'rewards': episode_rewards,
        'mean_rewards': mean_rewards
    }

def main():
    """主函数，根据命令行参数运行指定的算法"""
    # 解析命令行参数
    args = parse_args()
    
    # 如果提供了配置文件，从中加载参数
    if args.config:
        config = load_config(args.config)
        logger.info(f"从配置文件 {args.config} 加载参数")
        args = update_args_with_config(args, config)
    
    # 设置环境变量
    setup_env_variables()
    
    # 设置完整的保存路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_dir = f"logs/{args.env_name}_{timestamp}"
    
    # 创建必要的目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'data'), exist_ok=True)
    
    # 配置日志
    configure_logging(args)
    
    # ----- 优化Ray初始化，类似ARS中的方式 -----
    local_ip = socket.gethostbyname(socket.gethostname())
    # 获取当前工作目录，用于解决import路径问题
    working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 设置环境变量并创建runtime_env配置以传递给所有worker
    runtime_env = {
        "env_vars": {
            "MUJOCO_PY_MUJOCO_PATH": "/home/liangchen/.mujoco/mujoco210/",
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "") + ":/home/liangchen/.mujoco/mujoco210/bin:/usr/lib/nvidia",
            "PYTHONPATH": working_dir  # 确保工作进程的Python路径包含项目根目录
        },
        "working_dir": working_dir  # 添加工作目录配置
    }
    
    # 尝试连接到现有Ray集群或创建新集群
    try:
        if not ray.is_initialized():
            ray.init(address=f"{local_ip}:6379", runtime_env=runtime_env, ignore_reinit_error=True)
            logger.info(f"已连接到现有Ray集群: {local_ip}:6379")
    except Exception as e:
        logger.warning(f"连接到现有Ray集群失败: {e}, 创建新集群")
        if not ray.is_initialized():
            ray.init(runtime_env=runtime_env)
            logger.info("已创建新的Ray集群")
    # ----- 优化结束 -----

    if args.direct_algo:
        # 单算法直接运行模式
        logger.info(f"直接运行单算法模式: {args.direct_algo}")
        
        # 设置随机种子
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        # 根据算法类型运行对应的算法
        if args.direct_algo == 'ppo':
            # 创建环境
            env = make_env_ppo(args.env_name)
            
            # 创建PPO实例 - 根据是否使用MSZoo选择不同的类
            if args.use_mszoo:
                logger.info("使用多尺度零阶优化的PPO")
                ppo = MSZooPPO(
                    env=env,
                    hidden_dim=args.hidden_dim,
                    actor_lr=args.actor_lr,
                    critic_lr=args.critic_lr,
                    clip_ratio=0.2,
                    gamma=0.99,
                    gae_lambda=0.95,
                    target_kl=args.target_kl,
                    ent_coef=args.ent_coef,
                    batch_size=args.batch_size, 
                    update_epochs=args.ppo_epochs,
                    policy_update_interval=args.policy_update_interval,
                    normalize_states=args.normalize_states,
                    num_workers=args.n_workers,
                    mszoo_config=getattr(args, 'mszoo_config', {})
                )
            else:
                logger.info("使用标准PPO")
                ppo = PPO(
                    env=env,
                    hidden_dim=args.hidden_dim,
                    actor_lr=args.actor_lr,
                    critic_lr=args.critic_lr,
                    clip_ratio=0.2,
                    gamma=0.99,
                    gae_lambda=0.95,
                    target_kl=args.target_kl,
                    ent_coef=args.ent_coef,
                    batch_size=args.batch_size, 
                    update_epochs=args.ppo_epochs,
                    policy_update_interval=args.policy_update_interval,
                    normalize_states=args.normalize_states,
                    num_workers=args.n_workers
                )
            
            # 如果指定了加载路径，加载模型
            if args.load_path:
                ppo.load_model(args.load_path)
                logger.info(f"从 {args.load_path} 加载模型")
            
            # 确保保存目录存在
            save_dir = os.path.dirname(args.save_path)
            if save_dir:  # 如果save_path包含路径部分
                os.makedirs(save_dir, exist_ok=True)
            
            # 训练模型
            logger.info(f"开始训练PPO，环境：{args.env_name}，总步数：{args.timesteps}，并行：{args.parallel_collection}")
            start_time = time.time()
            
            if args.parallel_collection:
                # 并行训练
                rewards, eval_rewards = run_ppo_parallel(
                    args.env_name, args.timesteps, args.seed, args.save_path,
                    n_workers=args.n_workers,
                    hidden_dim=args.hidden_dim,
                    actor_lr=args.actor_lr,
                    critic_lr=args.critic_lr,
                    policy_update_interval=args.policy_update_interval,
                    batch_size=args.batch_size,
                    ppo_epochs=args.ppo_epochs,
                    ent_coef=args.ent_coef,
                    target_kl=args.target_kl,
                    normalize_states=args.normalize_states,
                    use_mszoo=args.use_mszoo
                )
            else:
                # 串行训练
                rewards, eval_rewards = run_ppo(
                    args.env_name, args.timesteps, args.seed, args.save_path,
                    hidden_dim=args.hidden_dim,
                    actor_lr=args.actor_lr,
                    critic_lr=args.critic_lr,
                    policy_update_interval=args.policy_update_interval,
                    batch_size=args.batch_size,
                    ppo_epochs=args.ppo_epochs,
                    ent_coef=args.ent_coef,
                    target_kl=args.target_kl,
                    normalize_states=args.normalize_states,
                    use_mszoo=args.use_mszoo
                )
            
            training_time = time.time() - start_time
            
            # 打印训练结果
            logger.info("\n===== PPO训练结果 =====")
            logger.info(f"环境: {args.env_name}")
            logger.info(f"训练步数: {args.timesteps}")
            logger.info(f"训练时间: {training_time/60:.2f} 分钟")
            logger.info(f"最终奖励: {(rewards[-1] if rewards else 0):.2f}")
            logger.info(f"评估奖励: {(np.mean(eval_rewards) if eval_rewards else 0):.2f}")
            logger.info(f"结果已保存到: {args.save_path}")
            
        elif args.direct_algo == 'ddpg':
            # 设置DDPG保存路径
            if not args.save_path:
                args.save_path = os.path.join(args.save_dir, 'models', 'ddpg')
            
            if args.parallel_collection:
                logger.info("使用并行方式运行DDPG")
                result = run_ddpg_parallel(args)
            else:
                logger.info("使用串行方式运行DDPG")
                result = run_ddpg(args)
                
            # 打印训练结果
            logger.info("\n===== DDPG训练结果 =====")
            logger.info(f"环境: {args.env_name}")
            logger.info(f"训练步数: {args.timesteps}")
            logger.info(f"训练时间: {result.get('training_time', 0)/60:.2f} 分钟")
            logger.info(f"最终奖励: {result.get('final_reward', 0):.2f}")
            logger.info(f"评估奖励: {result.get('eval_reward', 0):.2f}")
            logger.info(f"结果已保存到: {args.save_path}")
            
        elif args.direct_algo == 'ms_ddpg':
            logger.info("运行多尺度零阶优化的DDPG算法...")
            # 设置MSZoo DDPG保存路径
            if not args.save_path:
                args.save_path = os.path.join(args.save_dir, 'models', 'ms_ddpg')
            
            # 根据parallel_collection标志决定使用并行还是串行模式
            if args.parallel_collection:
                logger.info("使用并行方式运行MSZooDDPG")
                result = run_ms_ddpg_parallel(args)
            else:
                logger.info("使用串行方式运行MSZooDDPG")
                result = run_ms_ddpg(args)
            
            # 打印训练结果
            logger.info("\n===== MSZoo-DDPG训练结果 =====")
            logger.info(f"环境: {args.env_name}")
            logger.info(f"训练步数: {args.timesteps}")
            logger.info(f"训练时间: {result.get('training_time', 0)/60:.2f} 分钟")
            logger.info(f"最终奖励: {result.get('final_reward', 0):.2f}")
            logger.info(f"评估奖励: {result.get('eval_reward', 0):.2f}")
            logger.info(f"结果已保存到: {args.save_path}")
            
        elif args.direct_algo == 'ms_ppo':
            # 创建环境
            env = make_env_ppo(args.env_name)
            
            # 多尺度零阶优化器配置
            mszoo_config = getattr(args, 'mszoo_config', {})
            
            # 创建MSZooPPO实例
            logger.info("使用多尺度零阶优化的PPO")
            ppo = MSZooPPO(
                env=env,
                hidden_dim=args.hidden_dim,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                clip_ratio=0.2,
                gamma=0.99,
                gae_lambda=0.95,
                target_kl=args.target_kl,
                ent_coef=args.ent_coef,
                batch_size=args.batch_size, 
                update_epochs=args.ppo_epochs,
                policy_update_interval=args.policy_update_interval,
                normalize_states=args.normalize_states,
                num_workers=args.n_workers,
                mszoo_config=mszoo_config
            )
            
            # 如果指定了加载路径，加载模型
            if args.load_path:
                ppo.load_model(args.load_path)
                logger.info(f"从 {args.load_path} 加载模型")
            
            # 确保保存目录存在
            save_dir = os.path.dirname(args.save_path)
            if save_dir:  # 如果save_path包含路径部分
                os.makedirs(save_dir, exist_ok=True)
            
            # 训练模型
            logger.info(f"开始训练MSZoo-PPO，环境：{args.env_name}，总步数：{args.timesteps}，并行：{args.parallel_collection}")
            start_time = time.time()
            
            if args.parallel_collection:
                # 并行训练
                rewards, eval_rewards = run_ppo_parallel(
                    args.env_name, args.timesteps, args.seed, args.save_path,
                    n_workers=args.n_workers,
                    hidden_dim=args.hidden_dim,
                    actor_lr=args.actor_lr,
                    critic_lr=args.critic_lr,
                    policy_update_interval=args.policy_update_interval,
                    batch_size=args.batch_size,
                    ppo_epochs=args.ppo_epochs,
                    ent_coef=args.ent_coef,
                    target_kl=args.target_kl,
                    normalize_states=args.normalize_states,
                    use_mszoo=True,
                    mszoo_config=mszoo_config
                )
            else:
                # 串行训练
                rewards, eval_rewards = run_ppo(
                    args.env_name, args.timesteps, args.seed, args.save_path,
                    hidden_dim=args.hidden_dim,
                    actor_lr=args.actor_lr,
                    critic_lr=args.critic_lr,
                    policy_update_interval=args.policy_update_interval,
                    batch_size=args.batch_size,
                    ppo_epochs=args.ppo_epochs,
                    ent_coef=args.ent_coef,
                    target_kl=args.target_kl,
                    normalize_states=args.normalize_states,
                    use_mszoo=True,
                    mszoo_config=mszoo_config
                )
            
            training_time = time.time() - start_time
            
            # 打印训练结果
            logger.info("\n===== MSZoo-PPO训练结果 =====")
            logger.info(f"环境: {args.env_name}")
            logger.info(f"训练步数: {args.timesteps}")
            logger.info(f"训练时间: {training_time/60:.2f} 分钟")
            logger.info(f"最终奖励: {(rewards[-1] if rewards else 0):.2f}")
            logger.info(f"评估奖励: {(np.mean(eval_rewards) if eval_rewards else 0):.2f}")
            logger.info(f"结果已保存到: {args.save_path}")
            
        elif args.direct_algo == 'ars':
            logger.info("运行ARS算法...")
            # 确定保存路径
            if not args.save_path:
                args.save_path = os.path.join(args.save_dir, 'ars')
            args.ars_save_path = args.save_path
            
            # 运行ARS
            result = run_ars_custom(args)
            
            # 打印训练结果
            logger.info("\n===== ARS训练结果 =====")
            logger.info(f"环境: {args.env_name}")
            logger.info(f"训练步数: {args.timesteps}")
            logger.info(f"训练时间: {result.get('training_time', 0)/60:.2f} 分钟")
            logger.info(f"最终奖励: {result.get('final_reward', 0):.2f}")
            logger.info(f"结果已保存到: {args.save_path}")
    
    else:
        # 算法比较模式
        logger.info(f"开始比较强化学习算法在环境 {args.env_name} 上的表现")
        
        # 运行选择的算法
        results = []
        
        if hasattr(args, 'algorithms'):
            algorithms = args.algorithms
        else:
            # 默认运行所有算法
            algorithms = ['ars', 'ppo', 'ddpg']
            
        for algo in algorithms:
            # 如果用户指定了保存路径，则为每个算法创建子目录
            algo_save_path = None
            if args.save_path:
                algo_save_path = os.path.join(args.save_path, algo.lower())
                os.makedirs(os.path.dirname(algo_save_path), exist_ok=True)
            
            if algo.lower() == 'ars':
                logger.info("运行ARS算法...")
                # 更新ARS参数的保存路径
                if algo_save_path:
                    args.ars_save_path = algo_save_path
                ars_result = run_ars_custom(args)
                results.append(ars_result)
            
            elif algo.lower() == 'ppo':
                logger.info("运行PPO算法...")
                # 设置PPO保存路径
                ppo_save_path = algo_save_path if algo_save_path else os.path.join(args.save_dir, 'models', 'ppo')
                
                if args.parallel_collection:
                    logger.info("使用并行方式运行PPO")
                    ppo_parallel_path = f"{ppo_save_path}_parallel"
                    rewards, eval_rewards = run_ppo_parallel(
                        args.env_name, args.timesteps, args.seed,
                        ppo_parallel_path,
                        n_workers=args.n_workers,
                        hidden_dim=args.hidden_dim,
                        actor_lr=args.actor_lr,
                        critic_lr=args.critic_lr,
                        policy_update_interval=args.policy_update_interval,
                        batch_size=args.batch_size,
                        ppo_epochs=args.ppo_epochs,
                        ent_coef=args.ent_coef,
                        target_kl=args.target_kl,
                        normalize_states=args.normalize_states,
                        use_mszoo=False
                    )
                else:
                    logger.info("使用串行方式运行PPO")
                    rewards, eval_rewards = run_ppo(
                        args.env_name, args.timesteps, args.seed,
                        ppo_save_path,
                        hidden_dim=args.hidden_dim,
                        actor_lr=args.actor_lr,
                        critic_lr=args.critic_lr,
                        policy_update_interval=args.policy_update_interval,
                        batch_size=args.batch_size,
                        ppo_epochs=args.ppo_epochs,
                        ent_coef=args.ent_coef,
                        target_kl=args.target_kl,
                        normalize_states=args.normalize_states,
                        use_mszoo=False
                    )
                
                results.append({
                    'algorithm': 'PPO' + (' Parallel' if args.parallel_collection else ''),
                    'final_reward': rewards[-1] if rewards else 0,
                    'eval_reward': np.mean(eval_rewards) if eval_rewards else 0,
                    'training_time': 0,  # 需要在run_ppo中添加计时和返回
                    'rewards': rewards,
                    'eval_rewards': eval_rewards
                })
            
            elif algo.lower() == 'ms_ppo':
                logger.info("运行多尺度零阶优化的PPO算法...")
                # 设置MSZoo PPO保存路径
                ms_ppo_save_path = algo_save_path if algo_save_path else os.path.join(args.save_dir, 'models', 'ms_ppo')
                
                # MSZoo配置
                mszoo_config = getattr(args, 'mszoo_config', {})
                
                if args.parallel_collection:
                    logger.info("使用并行方式运行MSZoo PPO")
                    ms_ppo_parallel_path = f"{ms_ppo_save_path}_parallel"
                    rewards, eval_rewards = run_ppo_parallel(
                        args.env_name, args.timesteps, args.seed,
                        ms_ppo_parallel_path,
                        n_workers=args.n_workers,
                        hidden_dim=args.hidden_dim,
                        actor_lr=args.actor_lr,
                        critic_lr=args.critic_lr,
                        policy_update_interval=args.policy_update_interval,
                        batch_size=args.batch_size,
                        ppo_epochs=args.ppo_epochs,
                        ent_coef=args.ent_coef,
                        target_kl=args.target_kl,
                        normalize_states=args.normalize_states,
                        use_mszoo=True,
                        mszoo_config=mszoo_config
                    )
                else:
                    logger.info("使用串行方式运行MSZoo PPO")
                    rewards, eval_rewards = run_ppo(
                        args.env_name, args.timesteps, args.seed,
                        ms_ppo_save_path,
                        hidden_dim=args.hidden_dim,
                        actor_lr=args.actor_lr,
                        critic_lr=args.critic_lr,
                        policy_update_interval=args.policy_update_interval,
                        batch_size=args.batch_size,
                        ppo_epochs=args.ppo_epochs,
                        ent_coef=args.ent_coef,
                        target_kl=args.target_kl,
                        normalize_states=args.normalize_states,
                        use_mszoo=True,
                        mszoo_config=mszoo_config
                    )
                
                results.append({
                    'algorithm': 'MSZoo-PPO' + (' Parallel' if args.parallel_collection else ''),
                    'final_reward': rewards[-1] if rewards else 0,
                    'eval_reward': np.mean(eval_rewards) if eval_rewards else 0,
                    'training_time': 0,
                    'rewards': rewards,
                    'eval_rewards': eval_rewards
                })
            
            elif algo.lower() == 'ddpg':
                logger.info("运行DDPG算法...")
                # 设置DDPG保存路径
                if algo_save_path:
                    args.ddpg_save_path = algo_save_path
                
                if args.parallel_collection:
                    logger.info("使用并行方式运行DDPG")
                    ddpg_result = run_ddpg_parallel(args)
                else:
                    logger.info("使用串行方式运行DDPG")
                    ddpg_result = run_ddpg(args)
                results.append(ddpg_result)
            
            elif algo.lower() == 'ms_ddpg':
                logger.info("运行多尺度零阶优化的DDPG算法...")
                # 设置MSZoo DDPG保存路径
                if algo_save_path:
                    args.ms_ddpg_save_path = algo_save_path
                
                # 运行MSZoo DDPG
                ms_ddpg_result = run_ms_ddpg(args)
                results.append(ms_ddpg_result)
        
        # 打印算法性能摘要
        logger.info("\n===== 算法性能摘要 =====")
        for result in results:
            logger.info(f"{result['algorithm']}:")
            logger.info(f"  - 最终奖励: {result.get('final_reward', 0):.2f}")
            if 'eval_reward' in result:
                logger.info(f"  - 评估奖励: {result.get('eval_reward', 0):.2f}")
            logger.info(f"  - 训练时间: {result.get('training_time', 0)/60:.2f} 分钟")
        
        # 如果有多个算法结果，绘制比较图
        if len(results) > 1:
            plot_comparison(results, args)
            
        logger.info(f"\n结果已保存到: {args.save_dir}")

if __name__ == "__main__":
    main()