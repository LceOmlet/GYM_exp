import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import LambdaLR
import gym
import time
import ray
import os
import logging
import matplotlib.pyplot as plt
import argparse
import datetime
import json
import csv
from copy import deepcopy
from src.utils import RewardProcessor, RewardWrapperBase

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorNetwork(nn.Module):
    """PPO的Actor网络，输出连续动作空间的动作分布"""
    
    def __init__(self, state_dim, action_dim, action_bound, hidden_dim=256):
        """
        初始化Actor网络
        
        参数:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            action_bound: 动作边界 [low, high]
            hidden_dim: 隐藏层维度
        """
        super(ActorNetwork, self).__init__()
        
        self.action_bound = action_bound
        # 确保action_scale和action_bias是torch.Tensor
        # 计算标量值然后创建tensor，避免警告
        scale_value = (action_bound[1] - action_bound[0]) / 2.0
        bias_value = (action_bound[1] + action_bound[0]) / 2.0
        self.action_scale = torch.tensor(scale_value, dtype=torch.float32)
        self.action_bias = torch.tensor(bias_value, dtype=torch.float32)
        
        # 网络层定义 - 增加一个隐藏层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # 策略均值头
        self.mean_head = nn.Linear(hidden_dim // 2, action_dim)
        
        # 策略标准差头 (学习状态无关的log_std)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
    def forward(self, state):
        """前向传播计算动作分布"""
        # 确保action_scale和action_bias在与state相同的设备上
        device = state.device
        action_scale = self.action_scale.to(device) if isinstance(self.action_scale, torch.Tensor) else torch.tensor(self.action_scale, device=device)
        action_bias = self.action_bias.to(device) if isinstance(self.action_bias, torch.Tensor) else torch.tensor(self.action_bias, device=device)
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # 计算均值，使用tanh激活函数将输出限制在[-1, 1]
        mean = torch.tanh(self.mean_head(x))
        
        # 缩放到动作空间
        mean = mean * action_scale + action_bias
        
        # 获取标准差并确保其为正值
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample_action(self, state):
        """
        根据当前状态采样动作
        
        参数:
            state: 状态输入
            
        返回:
            action: 采样的动作
            log_prob: 动作的对数概率
            mean: 动作分布的均值
            std: 动作分布的标准差
        """
        # 确保action_scale和action_bias在与state相同的设备上
        device = state.device
        action_scale = self.action_scale.to(device) if isinstance(self.action_scale, torch.Tensor) else torch.tensor(self.action_scale, device=device)
        action_bias = self.action_bias.to(device) if isinstance(self.action_bias, torch.Tensor) else torch.tensor(self.action_bias, device=device)
        
        # 提取动作分布参数
        mean, std = self.forward(state)
        
        # 创建正态分布
        normal = torch.distributions.Normal(mean, std)
        
        # 对分布进行重参数化采样
        x_t = normal.rsample()  # 重参数化技巧
        
        # 通过tanh对采样动作进行压缩，将值域限制在[-1, 1]
        y_t = torch.tanh(x_t)
        
        # 将裁剪后的动作映射到环境的动作范围
        action = y_t * action_scale + action_bias
        
        # 计算log概率，考虑tanh变换的影响
        log_prob = normal.log_prob(x_t)
        
        # 修正log_prob，考虑tanh变换导致的分布变化
        # log prob = log prob - log(1 - tanh(x)^2)
        log_prob -= torch.log(action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # 确保所有返回值在转换为numpy前都已移动到CPU
        # 将所有张量移动到CPU
        action_cpu = action.cpu()
        log_prob_cpu = log_prob.cpu()
        mean_cpu = mean.cpu()
        std_cpu = std.cpu()
        
        # 转换为numpy数组
        action_np = action_cpu.numpy()
        log_prob_np = log_prob_cpu.numpy()
        mean_np = mean_cpu.numpy()
        std_np = std_cpu.numpy()
        
        return action_np, log_prob_np, mean_np, std_np
    
    def evaluate_action(self, state, action):
        """评估动作的log概率和熵"""
        try:
            # 确保输入有效
            if state.shape[0] == 0 or action.shape[0] == 0:
                logger.warning(f"evaluate_action收到空输入: state.shape={state.shape}, action.shape={action.shape}")
                return torch.zeros_like(action[:, 0:1]), torch.zeros_like(action[:, 0:1])
                
            # 确保action_scale和action_bias在与state相同的设备上
            device = state.device
            action_scale = self.action_scale.to(device) if isinstance(self.action_scale, torch.Tensor) else torch.tensor(self.action_scale, device=device)
            action_bias = self.action_bias.to(device) if isinstance(self.action_bias, torch.Tensor) else torch.tensor(self.action_bias, device=device)
            
            # 动作从实际范围转回tanh之前的范围
            action_normalized = (action - action_bias) / action_scale
            action_normalized = torch.clamp(action_normalized, -0.999, 0.999)
            
            # 获取分布参数
            mean, std = self.forward(state)
            
            # 创建正态分布
            normal = Normal(mean, std)
            
            # tanh逆变换
            x_t = torch.atanh(action_normalized)
            
            # 计算log概率
            log_prob = normal.log_prob(x_t)
            
            # 由于tanh变换，需要调整log_prob
            log_prob -= torch.log(action_scale * (1 - action_normalized.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            
            # 计算熵
            entropy = normal.entropy().sum(1, keepdim=True)
            
            return log_prob, entropy
            
        except Exception as e:
            # 记录错误信息，返回零值
            logger.error(f"evaluate_action错误: {str(e)}")
            return torch.zeros_like(action[:, 0:1]), torch.zeros_like(action[:, 0:1])

class CriticNetwork(nn.Module):
    """PPO的Critic网络，评估状态值函数"""
    
    def __init__(self, state_dim, hidden_dim=256):
        """
        初始化Critic网络
        
        参数:
            state_dim: 状态空间维度
            hidden_dim: 隐藏层维度
        """
        super(CriticNetwork, self).__init__()
        
        # 网络层定义 - 增加一个隐藏层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, state):
        """前向传播计算状态值"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.fc4(x)
        
        return value

class PPOMemory:
    """经验回放缓冲区"""
    
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = []
        self.returns = []
        self.batch_size = batch_size
        
    def store(self, state, action, reward, done, log_prob, value):
        """存储一条经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
    def store_batch(self, states, actions, rewards, dones, log_probs, values):
        """批量存储经验数据
        
        参数:
            states: 状态列表或数组
            actions: 动作列表或数组
            rewards: 奖励列表或数组
            dones: 完成标志列表或数组
            log_probs: 动作对数概率列表或数组
            values: 状态值列表或数组
        """
        self.states.extend(states)
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.dones.extend(dones)
        self.log_probs.extend(log_probs)
        self.values.extend(values)
        
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = []
        self.returns = []
    
    def get_all(self):
        """返回所有存储的经验数据"""
        return self.states, self.actions, self.log_probs, self.rewards, self.dones, self.values
        
    def compute_advantages_and_returns(self, last_value, gamma, gae_lambda):
        """计算GAE优势函数和回报"""
        # 检查是否有足够的数据
        if len(self.rewards) == 0:
            self.advantages = []
            self.returns = []
            return
            
        values = self.values + [last_value]
        advantages = []
        returns = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            
        self.advantages = advantages
        self.returns = returns
        
    def get_batches(self):
        """生成批次数据"""
        # 检查是否有足够的数据
        n_states = len(self.states)
        if n_states == 0:
            return (torch.FloatTensor([]).to(device), torch.FloatTensor([]).to(device), 
                    torch.FloatTensor([]).to(device), torch.FloatTensor([]).to(device), 
                    torch.FloatTensor([]).to(device), [])
        
        # 创建批次索引
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        
        # 转换为PyTorch张量
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.FloatTensor(np.array(self.actions)).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        advantages = torch.FloatTensor(np.array(self.advantages)).to(device)
        returns = torch.FloatTensor(np.array(self.returns)).to(device)
        
        # 标准化优势
        if len(advantages) > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return states, actions, old_log_probs, advantages, returns, batches

class RunningMeanStd:
    """
    计算输入数据的运行均值和标准差，用于状态归一化
    """
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

class StateNormalizer:
    """
    状态归一化器，使用运行均值和标准差来归一化状态
    """
    def __init__(self, shape=()):
        self.rms = RunningMeanStd(shape=shape)
        self.cliprange = 5.0
    
    def normalize(self, x):
        """归一化状态"""
        self.rms.update(x)
        normalized_x = np.clip(
            (x - self.rms.mean) / np.sqrt(self.rms.var + 1e-8),
            -self.cliprange, self.cliprange
        )
        return normalized_x
    
    def denormalize(self, x):
        """反归一化状态"""
        denormalized_x = x * np.sqrt(self.rms.var + 1e-8) + self.rms.mean
        return denormalized_x

class PPO:
    """PPO算法实现"""
    
    def __init__(self, env, hidden_dim=1024, actor_lr=2e-4, critic_lr=1e-3,
                 gamma=0.99, gae_lambda=0.97, clip_ratio=0.2, target_kl=0.01,
                 ent_coef=0.01, batch_size=512, update_epochs=5, 
                 max_grad_norm=0.5, policy_update_interval=2048,
                 num_workers=8, log_dir=None, normalize_states=True):
        """
        初始化PPO算法
        
        参数:
            env: 环境
            hidden_dim: 隐藏层维度
            actor_lr: Actor网络学习率
            critic_lr: Critic网络学习率
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            clip_ratio: PPO裁剪参数
            target_kl: 目标KL散度
            ent_coef: 熵系数
            batch_size: 批次大小 (增大为512以处理更多样本)
            update_epochs: 每次策略更新的epoch数
            max_grad_norm: 梯度裁剪的最大范数
            policy_update_interval: 策略更新间隔的环境步数 (增大为2048以收集足够多的完整回合)
            num_workers: 并行工作进程数量
            log_dir: 日志目录，如果为None则创建基于时间戳的目录
            normalize_states: 是否对状态进行归一化
        """
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self.action_bound = [self.action_low, self.action_high]
        
        # 设置设备
        self.device = device
        
        # 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.ent_coef = ent_coef
        self.update_epochs = update_epochs
        self.max_grad_norm = max_grad_norm
        self.policy_update_interval = policy_update_interval
        self.num_workers = num_workers
        self.normalize_states = normalize_states
        
        # 状态归一化器
        self.state_normalizer = StateNormalizer(shape=(self.state_dim,)) if normalize_states else None
        
        # 初始化网络
        self.actor = ActorNetwork(self.state_dim, self.action_dim, 
                                 self.action_bound, hidden_dim).to(self.device)
        self.critic = CriticNetwork(self.state_dim, hidden_dim).to(self.device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 学习率调度器
        self.actor_scheduler = None
        self.critic_scheduler = None
        self.total_timesteps = None
        
        # 经验回放缓冲区
        self.memory = PPOMemory(batch_size)
        
        # 训练统计
        self.episode_rewards = []
        self.mean_rewards = []
        self.kl_divs = []
        self.value_losses = []
        self.policy_losses = []
        self.entropies = []
        
        # 工作进程
        self.workers = None
        
        # 设置日志目录
        self.setup_logging(log_dir)
        
    def setup_logging(self, log_dir=None):
        """设置日志目录和文件"""
        # 如果没有指定日志目录，创建基于时间戳的目录
        if log_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            env_name = self.env.unwrapped.spec.id.replace('/', '_')
            log_dir = os.path.join('logs', f"{env_name}_{timestamp}")
        
        # 创建日志目录结构
        self.log_dir = log_dir
        self.models_dir = os.path.join(log_dir, 'models')
        self.plots_dir = os.path.join(log_dir, 'plots')
        self.data_dir = os.path.join(log_dir, 'data')
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 设置文件日志处理器
        self.rewards_file = os.path.join(self.data_dir, 'rewards.csv')
        self.log_file = os.path.join(self.log_dir, 'training.log')
        self.config_file = os.path.join(self.log_dir, 'config.json')
        
        # 配置文件日志
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # 保存配置
        self.save_config()
        
        # 初始化CSV文件
        with open(self.rewards_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Timestep', 'Reward', 'MeanReward', 'PolicyLoss', 'ValueLoss', 'KL', 'Entropy'])
        
        logger.info(f"日志将保存到: {self.log_dir}")
    
    def save_config(self):
        """保存训练配置到JSON文件"""
        config = {
            'env_id': self.env.unwrapped.spec.id,
            'hidden_dim': self.actor.fc1.out_features,  # 从网络获取隐藏层维度
            'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
            'critic_lr': self.critic_optimizer.param_groups[0]['lr'],
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_ratio': self.clip_ratio,
            'target_kl': self.target_kl,
            'ent_coef': self.ent_coef,
            'batch_size': self.memory.batch_size,
            'update_epochs': self.update_epochs,
            'max_grad_norm': self.max_grad_norm,
            'policy_update_interval': self.policy_update_interval,
            'num_workers': self.num_workers,
            'device': str(self.device),
            'obs_dim': self.state_dim,
            'action_dim': self.action_dim,
            'action_low': self.action_low.tolist() if hasattr(self.action_low, 'tolist') else self.action_low,
            'action_high': self.action_high.tolist() if hasattr(self.action_high, 'tolist') else self.action_high,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)
            
    def log_episode_data(self, episode, timestep, reward, mean_reward, policy_loss=None, value_loss=None, kl=None, entropy=None):
        """记录回合数据到CSV文件"""
        with open(self.rewards_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, timestep, reward, mean_reward, 
                             policy_loss if policy_loss is not None else '',
                             value_loss if value_loss is not None else '',
                             kl if kl is not None else '',
                             entropy if entropy is not None else ''])

    def init_workers(self, env_id):
        """
        初始化工作进程
        
        参数:
            env_id: 环境ID
        """
        # 初始化Ray (如果尚未初始化)
        if not ray.is_initialized():
            try:
                ray.init(ignore_reinit_error=True)
                logger.info("Ray初始化成功")
            except Exception as e:
                logger.error(f"Ray初始化失败: {e}")
                raise RuntimeError("无法初始化Ray，并行训练将无法进行")
            
        # 创建工作进程 - 简化初始化流程
        try:
            logger.info(f"正在创建 {self.num_workers} 个工作进程...")
            self.workers = [PPOWorker.remote(env_id, seed=i) for i in range(self.num_workers)]
            logger.info(f"已初始化 {len(self.workers)} 个工作进程")
            
            # 移除连接测试，避免额外的RPC调用
        except Exception as e:
            logger.error(f"创建工作进程失败: {e}")
            raise RuntimeError("无法创建工作进程，并行训练将无法进行")

    def _get_cpu_weights(self):
        """获取模型在CPU上的权重副本，用于传递给工作进程"""
        # 将Actor和Critic模型移至CPU并提取参数
        actor_weights = {k: v.cpu() for k, v in self.actor.state_dict().items()}
        critic_weights = {k: v.cpu() for k, v in self.critic.state_dict().items()}
        return actor_weights, critic_weights

    def _merge_experiences(self, all_experiences):
        """合并从多个工作进程收集的经验数据"""
        if not all_experiences:
            logger.warning("没有经验数据可合并")
            return
        
        # 初始化计数器
        total_steps = 0
        total_episodes_completed = 0
        all_episode_rewards = []
        
        # 预分配内存容量
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        # 合并所有采集的经验
        for exp in all_experiences:
            if not isinstance(exp, dict) or 'states' not in exp:
                logger.warning(f"收到无效的经验数据格式: {type(exp)}")
                continue
                
            # 更新计数器
            current_steps = len(exp['states'])
            total_steps += current_steps
            
            # 记录统计信息
            if 'episodes_completed' in exp:
                total_episodes_completed += exp['episodes_completed']
            
            if 'episode_rewards' in exp and exp['episode_rewards']:
                all_episode_rewards.extend(exp['episode_rewards'])
            
            # 高效地批量追加数据，而不是一个一个追加
            states.extend(exp['states'])
            actions.extend(exp['actions'])
            rewards.extend(exp['rewards'])
            values.extend(exp['values'])
            log_probs.extend(exp['log_probs'])
            dones.extend(exp['dones'])
        
        # 记录合并后的经验数量
        logger.info(f"合并了 {total_steps} 步经验数据，完成了 {total_episodes_completed} 个episode")
        
        # 记录奖励统计信息
        if all_episode_rewards:
            mean_reward = np.mean(all_episode_rewards)
            min_reward = np.min(all_episode_rewards)
            max_reward = np.max(all_episode_rewards)
            logger.info(f"已完成episode的奖励统计: 均值={mean_reward:.2f}, 最小值={min_reward:.2f}, 最大值={max_reward:.2f}, 数量={len(all_episode_rewards)}")
        
        # 一次性添加到内存，提高效率
        if states:
            self.memory.store_batch(
                states=states,
                actions=actions, 
                rewards=rewards,
                values=values,
                log_probs=log_probs,
                dones=dones
            )
        else:
            logger.warning("无法合并经验数据：数据为空")

    def setup_lr_schedulers(self, total_timesteps):
        """
        设置学习率调度器，使学习率随着训练进度逐渐降低
        
        参数:
            total_timesteps: 总训练时间步数
        """
        self.total_timesteps = total_timesteps
        
        # 定义学习率衰减函数，从1衰减到0.3，衰减更平缓
        def lr_lambda(step):
            return 1.0 - 0.7 * (step / total_timesteps)
        
        # 创建学习率调度器
        self.actor_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=lr_lambda)
        self.critic_scheduler = LambdaLR(self.critic_optimizer, lr_lambda=lr_lambda)
        
        logger.info("已设置学习率调度器，学习率将随训练进度平缓线性衰减")
    
    def train(self, total_timesteps, env_id=None, log_interval=10, save_interval=None, save_path=None):
        """
        训练PPO算法
        
        参数:
            total_timesteps: 总时间步数
            env_id: 环境ID (用于并行训练)
            log_interval: 日志记录间隔
            save_interval: 模型保存间隔
            save_path: 模型保存路径
        
        返回:
            episode_rewards: 每个回合的奖励列表
            mean_rewards: 平滑后的平均奖励
        """
        # 记录训练开始时间
        training_start_time = time.time()
        
        # 设置学习率调度器
        self.setup_lr_schedulers(total_timesteps)
        env_id = self.env.unwrapped.spec.id
        
        # 使用日志目录中的模型目录作为默认保存路径
        if save_path is None:
            save_path = self.models_dir
        
        # 是否使用并行训练
        use_parallel = self.num_workers > 0
        
        # 如果使用并行训练，初始化工作进程
        if use_parallel and self.workers is None:
            self.init_workers(env_id)
            logger.info(f"使用 {self.num_workers} 个工作进程进行并行训练")
        
        # 训练状态初始化
        if not use_parallel:
            state, _ = self.env.reset()
            
        current_ep_reward = 0
        episode_rewards = []
        mean_rewards = []
        timestep = 0
        episode = 0
        update_count = 0
        last_policy_loss = 0
        last_value_loss = 0
        last_kl = 0
        last_entropy = 0
        
        # 创建保存目录
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
        
        logger.info(f"开始训练PPO，总时间步数: {total_timesteps}")
        
        # 设置更频繁的收集和更新
        # 批量处理设置 - 确保处理足够多的样本
        update_interval = min(self.policy_update_interval, 16000)  # 每次更新收集足够多的样本，与ARS类似
        logger.info(f"使用更新间隔: {update_interval} 步")
        
        # 主训练循环
        while timestep < total_timesteps:
            print("use parallel: ", use_parallel)
            if use_parallel:
                # 并行收集经验
                # 显著增加每个工作进程收集的步数，确保能完成足够的回合
                timesteps_per_worker = max(1000, update_interval // self.num_workers)  # 最少1000步而不是几十步
                start_time = time.time()
                
                # 获取当前网络参数（确保在CPU上）
                actor_weights, critic_weights = self._get_cpu_weights()
                
                # 添加调试日志
                logger.info(f"开始并行收集经验，每个工作进程收集 {timesteps_per_worker} 步, 共 {self.num_workers} 个工作进程")
                logger.info(f"预计总共收集 {timesteps_per_worker * self.num_workers} 步经验数据")
                
                # 并行收集经验
                experience_ids = []
                for worker_idx, worker in enumerate(self.workers):
                    experience_ids.append(worker.collect_experiences.remote(
                        actor_weights, 
                        critic_weights, 
                        max_steps=timesteps_per_worker,
                        gamma=self.gamma, 
                        gae_lambda=self.gae_lambda
                    ))
                
                # 等待所有工作进程完成
                all_experiences = ray.get(experience_ids)
                collection_time = time.time() - start_time
                
                # 调试日志
                for worker_idx, exp in enumerate(all_experiences):
                    if 'steps' in exp:
                        logger.info(f"工作进程 {worker_idx} 收集了 {exp['steps']} 步")
                    
                    # 详细记录完整回合奖励
                    if 'episode_rewards' in exp and exp['episode_rewards']:
                        rewards_str = ', '.join([f"{r:.1f}" for r in exp['episode_rewards']])
                        logger.info(f"工作进程 {worker_idx} 完成了 {len(exp['episode_rewards'])} 个回合，" +
                                   f"奖励: [{rewards_str}]")
                    elif 'episode_rewards' in exp:
                        logger.info(f"工作进程 {worker_idx} 没有完成任何回合")
                    
                    # 记录不完整回合的奖励
                    if 'ongoing_episode_reward' in exp and exp['ongoing_episode_reward'] is not None:
                        logger.info(f"工作进程 {worker_idx} 有未完成回合，当前奖励: {exp['ongoing_episode_reward']:.1f}")
                
                # 计算总步数
                collected_timesteps = sum(exp.get('steps', 0) for exp in all_experiences)
                
                # 收集所有回合奖励
                all_episode_rewards = []
                for exp in all_experiences:
                    if 'episode_rewards' in exp and len(exp['episode_rewards']) > 0:
                        all_episode_rewards.extend(exp['episode_rewards'])
                        # 额外打印每个worker完成的回合奖励，帮助调试
                        rewards_stats = {
                            'avg': sum(exp['episode_rewards']) / len(exp['episode_rewards']),
                            'min': min(exp['episode_rewards']),
                            'max': max(exp['episode_rewards'])
                        }
                        logger.info(f"工作进程奖励统计: avg={rewards_stats['avg']:.2f}, min={rewards_stats['min']:.2f}, max={rewards_stats['max']:.2f}")
                    
                    # 记录不完整回合的奖励
                    if 'ongoing_episode_reward' in exp and exp['ongoing_episode_reward'] is not None:
                        all_episode_rewards.append(exp['ongoing_episode_reward'])
                        logger.info(f"记录不完整回合奖励: {exp['ongoing_episode_reward']:.2f}")
                
                # 更新状态
                timestep += collected_timesteps
                
                # 计算奖励总结统计
                if all_episode_rewards:
                    reward_stats = {
                        'avg': sum(all_episode_rewards) / len(all_episode_rewards),
                        'min': min(all_episode_rewards),
                        'max': max(all_episode_rewards),
                        'std': np.std(all_episode_rewards),
                        'count': len(all_episode_rewards)
                    }
                    logger.info(f"===== 回合奖励总结: avg={reward_stats['avg']:.2f}, min={reward_stats['min']:.2f}, " +
                               f"max={reward_stats['max']:.2f}, std={reward_stats['std']:.2f}, count={reward_stats['count']} =====")
                
                # 记录奖励
                for reward in all_episode_rewards:
                    episode_rewards.append(reward)
                    mean_reward = np.mean(episode_rewards[-100:])
                    mean_rewards.append(mean_reward)
                    
                    # 记录到CSV文件 - 使用最新的指标
                    policy_loss = self.policy_losses[-1] if self.policy_losses else 0
                    value_loss = self.value_losses[-1] if self.value_losses else 0
                    kl = self.kl_divs[-1] if self.kl_divs else 0
                    entropy = self.entropies[-1] if self.entropies else 0
                    
                    self.log_episode_data(episode, timestep, reward, mean_reward, 
                                         policy_loss, value_loss, kl, entropy)
                    
                    logger.info(f"Episode: {episode} | Timestep: {timestep} | Reward: {reward:.2f} | Mean Reward: {mean_reward:.2f}")
                    episode += 1
                
                # 合并经验
                try:
                    # 确保memory是PPOMemory对象
                    if not isinstance(self.memory, PPOMemory):
                        # 如果memory不是PPOMemory对象，重新创建一个
                        logger.warning("Memory不是PPOMemory对象，重新创建")
                        self.memory = PPOMemory(batch_size=128)  # 使用默认batch_size
                    
                    # 合并经验
                    self._merge_experiences(all_experiences)
                    logger.info(f"经验合并完成，缓冲区大小: {len(self.memory.states)}")
                except Exception as e:
                    logger.error(f"合并经验时出错: {e}")
                    # 重新创建内存缓冲区
                    self.memory = PPOMemory(batch_size=128)  # 使用默认batch_size
                    
                    # 简单合并方法
                    for exp in all_experiences:
                        if all(k in exp for k in ['states', 'actions', 'rewards', 'dones', 'log_probs', 'values']):
                            for i in range(len(exp['states'])):
                                self.memory.store(
                                    exp['states'][i],
                                    exp['actions'][i],
                                    exp['rewards'][i],
                                    exp['dones'][i],
                                    exp['log_probs'][i],
                                    exp['values'][i]
                                )
                
                logger.info(f"并行收集经验完成。用时: {collection_time:.2f}秒, 收集步数: {collected_timesteps}, 回合数: {len(all_episode_rewards)}")
                
            else:
                # 串行收集经验
                collected_timesteps = 0
                
                # 重置内存缓冲区
                self.memory.clear()
                
                # 使用更小的更新间隔
                while collected_timesteps < update_interval:
                    # 转换状态为张量
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    
                    # 选择动作
                    with torch.no_grad():
                        action, log_prob, _, _ = self.actor.sample_action(state_tensor)
                    
                    # action和log_prob现在已经是numpy数组，由sample_action方法确保
                    # 执行动作（注意action是一个batch，我们需要取第一个元素）
                    next_state, reward, terminated, truncated, _ = self.env.step(action[0])
                    done = terminated or truncated
                    
                    # 获取值函数估计
                    value = self.critic(state_tensor).detach().cpu().numpy()
                    
                    # 存储转换（同样注意取log_prob的第一个元素）
                    self.memory.store(state, action[0], reward, done, log_prob[0], value[0])
                    
                    # 更新当前回合奖励
                    current_ep_reward += reward
                    collected_timesteps += 1
                    timestep += 1
                    
                    # 更新状态
                    state = next_state
                    
                    # 如果回合结束，重置环境
                    if done:
                        episode_rewards.append(current_ep_reward)
                        
                        # 计算平滑平均奖励（最近100个回合）
                        mean_reward = np.mean(episode_rewards[-100:])
                        mean_rewards.append(mean_reward)
                        
                        # 记录到CSV文件 - 使用最新的指标
                        policy_loss = self.policy_losses[-1] if self.policy_losses else 0
                        value_loss = self.value_losses[-1] if self.value_losses else 0
                        kl = self.kl_divs[-1] if self.kl_divs else 0
                        entropy = self.entropies[-1] if self.entropies else 0
                        
                        self.log_episode_data(episode, timestep, current_ep_reward, mean_reward, 
                                             policy_loss, value_loss, kl, entropy)
                        
                        logger.info(f"Episode: {episode} | Timestep: {timestep} | Reward: {current_ep_reward:.2f} | Mean Reward: {mean_reward:.2f}")
                        
                        # 重置状态和回合奖励
                        state, _ = self.env.reset()
                        current_ep_reward = 0
                        episode += 1
                        
                        # 保存模型
                        if save_interval and save_path and episode % save_interval == 0:
                            model_path = os.path.join(save_path, f"ppo_model_{timestep}.pt")
                            self.save_model(model_path)
                
                # 计算最后状态的值函数估计，用于计算GAE
                if not done:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        last_value = self.critic(state_tensor).detach().cpu().numpy()[0]
                else:
                    last_value = 0.0
                
                # 计算优势估计和回报
                self.memory.compute_advantages_and_returns(last_value, self.gamma, self.gae_lambda)
            
            # 更新策略
            start_time = time.time()
            last_policy_loss, last_value_loss, last_entropy, last_kl = self.update_policy()
            update_time = time.time() - start_time
            
            # 记录更新统计信息
            self.log_update_data(update_count, timestep, last_policy_loss, last_value_loss, 
                                last_entropy, last_kl, update_time)
            
            # 记录到训练统计中
            self.policy_losses.append(last_policy_loss)
            self.value_losses.append(last_value_loss)
            self.entropies.append(last_entropy)
            self.kl_divs.append(last_kl)
            
            # 更新学习率调度器
            if self.actor_scheduler is not None:
                self.actor_scheduler.step()
                self.critic_scheduler.step()
                
                # 每10次更新记录一次当前学习率
                if update_count % 10 == 0:
                    actor_lr = self.actor_optimizer.param_groups[0]['lr']
                    critic_lr = self.critic_optimizer.param_groups[0]['lr']
                    logger.info(f"当前学习率: actor_lr={actor_lr:.6f}, critic_lr={critic_lr:.6f}")
            
            update_count += 1
            
            # 每隔一段时间测试策略
            if save_interval and update_count % save_interval == 0:
                # 测试模型性能，使用ARS类似的方式记录完整回合奖励
                logger.info(f"============ 在时间步 {timestep} 测试模型性能 ============")
                
                if use_parallel:
                    test_rewards, test_mean = self.test_parallel(env_id, num_episodes=20)  # 增加测试回合数以获得更可靠的评估
                else:
                    test_rewards, test_mean = self.test(num_episodes=20)  # 增加测试回合数以获得更可靠的评估
                
                # 使用ARS类似的格式记录测试结果
                with open(os.path.join(self.data_dir, 'test_results.csv'), 'a', newline='') as f:
                    writer = csv.writer(f)
                    if update_count == save_interval:  # 第一次写入标题行
                        writer.writerow(['Time', 'Iteration', 'AverageReward', 'StdRewards', 'MaxRewardRollout', 'MinRewardRollout', 'timesteps'])
                    
                    # 计算时间
                    current_time = time.time() - training_start_time
                    
                    writer.writerow([
                        current_time, 
                        update_count // save_interval,  # 迭代次数
                        test_mean, 
                        np.std(test_rewards), 
                        np.max(test_rewards), 
                        np.min(test_rewards), 
                        timestep
                    ])
                
                # 记录ARS格式的日志
                logger.info(f"Time\tIteration\tAverageReward\tStdRewards\tMaxRewardRollout\tMinRewardRollout\ttimesteps")
                logger.info(f"{current_time:.2f}\t{update_count // save_interval}\t{test_mean:.2f}\t{np.std(test_rewards):.2f}\t{np.max(test_rewards):.2f}\t{np.min(test_rewards):.2f}\t{timestep}")
                
                # 保存模型
                if save_path:
                    model_path = os.path.join(save_path, f"ppo_model_{timestep}.pt")
                    self.save_model(model_path)
                    
                    # 如果表现最好，额外保存一个best模型
                    if len(mean_rewards) > 0 and test_mean > max(mean_rewards):
                        best_model_path = os.path.join(save_path, "ppo_model_best.pt")
                        self.save_model(best_model_path)
                        logger.info(f"保存最佳模型，奖励: {test_mean:.2f}")
        
        # 保存最终模型和训练曲线
        if save_path:
            final_model_path = os.path.join(save_path, "ppo_model_final.pt")
            self.save_model(final_model_path)
            
            # 保存训练曲线
            rewards_plot_path = os.path.join(self.plots_dir, "training_rewards.png")
            self.plot_rewards(episode_rewards=episode_rewards, mean_rewards=mean_rewards, save_path=rewards_plot_path)
        
        return episode_rewards, mean_rewards
    
    def test_parallel(self, env_id, num_episodes=10, render=False):
        """
        并行测试训练好的模型
        
        参数:
            env_id: 环境ID
            num_episodes: 测试回合数
            render: 是否渲染
            
        返回:
            total_rewards: 每个回合的奖励
            mean_reward: 平均奖励
        """
        # 确保工作进程已初始化
        if self.workers is None:
            self.init_workers(env_id)
        
        # 获取当前网络参数（确保在CPU上）
        actor_weights, _ = self._get_cpu_weights()
        
        # 根据工作进程数量分配测试回合
        episodes_per_worker = max(1, num_episodes // self.num_workers)
        remaining_episodes = num_episodes % self.num_workers
        
        # 并行测试
        test_ids = []
        for i, worker in enumerate(self.workers):
            worker_episodes = episodes_per_worker
            if i == len(self.workers) - 1:
                worker_episodes += remaining_episodes
                
            test_ids.append(worker.test_policy.remote(actor_weights, num_episodes=worker_episodes, render=render))
        
        # 等待所有工作进程完成
        all_rewards = ray.get(test_ids)
        
        # 合并结果
        total_rewards = []
        for rewards in all_rewards:
            total_rewards.extend(rewards)
        
        # 计算详细统计
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        min_reward = np.min(total_rewards)
        max_reward = np.max(total_rewards)
        
        logger.info(f"===== 并行测试统计 =====")
        logger.info(f"回合数: {len(total_rewards)}")
        logger.info(f"平均奖励: {mean_reward:.2f}, 标准差: {std_reward:.2f}")
        logger.info(f"最小奖励: {min_reward:.2f}, 最大奖励: {max_reward:.2f}")
        logger.info(f"=========================")
        
        return total_rewards, mean_reward
    
    def update_policy(self):
        """更新策略和价值网络"""
        # 从记忆缓冲区中获取经验
        states, actions, old_log_probs, rewards, dones, values = self.memory.get_all()
        
        # 将NumPy数组转换为PyTorch张量并移动到设备上
        # 先转换为NumPy数组再转换为张量，避免性能警告
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        
        # 计算优势函数和回报
        advantages, returns = self._compute_gae(rewards, values, dones)
        advantages = torch.FloatTensor(np.array(advantages)).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).to(self.device)
        
        # 归一化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 准备小批量
        batch_size = min(self.memory.batch_size, states.size(0))
        indices = np.arange(states.size(0))
        
        # 初始化统计信息
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        
        # 是否需要初始预训练
        if not hasattr(self, 'initial_update_done') or not self.initial_update_done:
            # 随机选择一些批次进行预训练
            np.random.shuffle(indices)
            batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
            
            for batch_idx in batches[:2]:  # 只使用前2个批次进行预训练，减少预训练时间
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # 计算价值损失
                value_pred = self.critic(batch_states)
                value_loss = F.mse_loss(value_pred, batch_returns)
                
                # 评估当前动作的log概率和熵
                new_log_probs, entropy = self.actor.evaluate_action(batch_states, batch_actions)
                
                # 使用简单的策略梯度损失，不使用PPO裁剪
                policy_loss = -(new_log_probs * batch_advantages).mean()
                
                # 更新Critic网络
                self.critic_optimizer.zero_grad()
                value_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # 更新Actor网络
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                logger.info(f"预训练: policy_loss={policy_loss.item():.6f}, value_loss={value_loss.item():.6f}")
            
            self.initial_update_done = True
        
        # 减少更新epoch数量，加快训练速度
        update_epochs = min(self.update_epochs, 5)  # 最多5个epoch
        
        # 多个epoch更新
        for epoch in range(update_epochs):
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_entropy = 0
            epoch_kl = 0
            num_batch_updates = 0
            
            # 在每个epoch开始时打乱数据
            np.random.shuffle(indices)
            batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
            
            for batch_idx in batches:
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # 评估当前动作的log概率和熵
                new_log_probs, entropy = self.actor.evaluate_action(batch_states, batch_actions)
                
                # 计算比率和裁剪的目标函数
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                
                # 策略损失
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 加入熵正则项
                policy_loss = policy_loss - self.ent_coef * entropy.mean()
                
                # 计算价值损失 - 加入值函数裁剪以提高稳定性
                value_pred = self.critic(batch_states)
                value_pred_clipped = batch_returns + torch.clamp(
                    value_pred - batch_returns,
                    -self.clip_ratio, 
                    self.clip_ratio
                )
                value_loss1 = F.mse_loss(value_pred, batch_returns)
                value_loss2 = F.mse_loss(value_pred_clipped, batch_returns)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2)  # 使用值函数裁剪
                
                # 计算KL散度
                kl = (batch_old_log_probs - new_log_probs).mean().item()
                
                
                # 使用复合损失函数 - 策略损失和值函数损失的加权和
                # 调低值函数的权重，因为从日志观察到值函数损失比策略损失大很多
                loss = policy_loss + 0.5 * value_loss
                
                # 更新网络
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # 记录损失
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy.mean().item()
                epoch_kl += kl
                num_batch_updates += 1
            
            # 记录每个epoch的平均损失
            if num_batch_updates > 0:
                total_policy_loss += epoch_policy_loss / num_batch_updates
                total_value_loss += epoch_value_loss / num_batch_updates
                total_entropy += epoch_entropy / num_batch_updates
                total_kl += epoch_kl / num_batch_updates
        
        # 清空缓冲区
        self.memory.clear()
        
        # 返回平均损失
        actual_epochs = min(update_epochs, epoch + 1)  # 考虑到可能提前停止
        avg_policy_loss = total_policy_loss / actual_epochs
        avg_value_loss = total_value_loss / actual_epochs
        avg_entropy = total_entropy / actual_epochs
        avg_kl = total_kl / actual_epochs
        
        return avg_policy_loss, avg_value_loss, avg_entropy, avg_kl
    
    def save_model(self, path):
        """保存模型"""
        # 确保模型参数在CPU上，这样可以在不同设备上加载
        actor_state = {k: v.cpu() for k, v in self.actor.state_dict().items()}
        critic_state = {k: v.cpu() for k, v in self.critic.state_dict().items()}
        
        torch.save({
            'actor_state_dict': actor_state,
            'critic_state_dict': critic_state,
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'hyperparams': {
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_ratio': self.clip_ratio,
                'target_kl': self.target_kl,
                'ent_coef': self.ent_coef
            }
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """加载模型"""
        # 添加map_location=torch.device('cpu')确保在CPU上加载
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        # 加载状态字典
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # 然后将模型移到适当的设备上
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)
        
        # 加载优化器状态
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # 加载超参数
        hyperparams = checkpoint['hyperparams']
        self.gamma = hyperparams['gamma']
        self.gae_lambda = hyperparams['gae_lambda']
        self.clip_ratio = hyperparams['clip_ratio']
        self.target_kl = hyperparams['target_kl']
        self.ent_coef = hyperparams['ent_coef']
        
        logger.info(f"Model loaded from {path}")
    
    def test(self, num_episodes=10, render=False):
        """测试训练好的模型"""
        total_rewards = []
        total_steps = 0
        
        for i in range(num_episodes):
            episode_reward = 0
            state, _ = self.env.reset()
            done = False
            step_count = 0
            
            while not done:
                if render:
                    self.env.render()
                
                # 选择动作
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, _, _, _ = self.actor.sample_action(state_tensor)
                
                # action已经是numpy数组，我们需要取第一个元素执行
                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action[0])
                done = terminated or truncated
                
                # 更新状态和回报
                state = next_state
                episode_reward += reward
                step_count += 1
                total_steps += 1
            
            total_rewards.append(episode_reward)
            logger.info(f"Test Episode: {i+1} | Reward: {episode_reward:.2f} | Steps: {step_count}")
        
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        min_reward = np.min(total_rewards)
        max_reward = np.max(total_rewards)
        avg_steps = total_steps / num_episodes
        
        logger.info(f"===== 测试统计 =====")
        logger.info(f"平均奖励: {mean_reward:.2f}, 标准差: {std_reward:.2f}")
        logger.info(f"最小奖励: {min_reward:.2f}, 最大奖励: {max_reward:.2f}")
        logger.info(f"平均步数: {avg_steps:.1f}, 总步数: {total_steps}")
        logger.info(f"====================")
        
        return total_rewards, mean_reward
    
    def plot_rewards(self, episode_rewards=None, mean_rewards=None, save_path=None):
        """
        绘制奖励曲线
        
        参数:
            episode_rewards: 每个回合的奖励列表
            mean_rewards: 平滑后的平均奖励
            save_path: 保存图片的路径
        """
        plt.figure(figsize=(10, 6))
        
        if episode_rewards is None:
            episode_rewards = self.episode_rewards
        if mean_rewards is None:
            mean_rewards = self.mean_rewards
        
        plt.plot(episode_rewards, label='Episode Rewards', alpha=0.3)
        plt.plot(mean_rewards, label='Mean 100 Rewards', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('PPO Training Rewards')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
            
    def save_training_data(self):
        """保存所有训练数据到文件"""
        # 保存奖励历史
        rewards_path = os.path.join(self.data_dir, 'episode_rewards.npy')
        mean_rewards_path = os.path.join(self.data_dir, 'mean_rewards.npy')
        
        np.save(rewards_path, np.array(self.episode_rewards))
        np.save(mean_rewards_path, np.array(self.mean_rewards))
        
        # 保存训练信息到JSON
        training_info = {
            'episodes': len(self.episode_rewards),
            'final_mean_reward': float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0,
            'max_reward': float(np.max(self.episode_rewards)) if self.episode_rewards else 0,
            'min_reward': float(np.min(self.episode_rewards)) if self.episode_rewards else 0,
            'std_reward': float(np.std(self.episode_rewards)) if self.episode_rewards else 0,
        }
        
        with open(os.path.join(self.log_dir, 'training_summary.json'), 'w') as f:
            json.dump(training_info, f, indent=4)
            
        logger.info(f"训练数据已保存到 {self.data_dir}")

    def log_update_data(self, update_count, timestep, policy_loss, value_loss, entropy, kl, update_time):
        """
        记录每次策略更新的数据
        
        参数:
            update_count: 更新计数
            timestep: 当前时间步
            policy_loss: 策略损失
            value_loss: 值函数损失
            entropy: 熵
            kl: KL散度
            update_time: 更新时间
        """
        # 保存每次更新的结果
        with open(os.path.join(self.data_dir, 'updates.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            if update_count == 1:  # 写入标题行
                writer.writerow(['Update', 'Timestep', 'PolicyLoss', 'ValueLoss', 'Entropy', 'KL', 'UpdateTime', 'EntCoef'])
            writer.writerow([update_count, timestep, policy_loss, value_loss, entropy, kl, update_time, self.ent_coef])
        
        logger.info(f"Update #{update_count} | Actor Loss: {policy_loss:.4f} | Critic Loss: {value_loss:.4f} | Entropy: {entropy:.4f} | Approx KL: {kl:.4f} | Time: {update_time:.2f}s | Ent Coef: {self.ent_coef:.6f}")

    def _compute_gae(self, rewards, values, dones):
        """
        计算广义优势估计(GAE)和回报
        
        参数:
            rewards: 奖励列表
            values: 价值估计列表
            dones: 终止标志列表
            
        返回:
            advantages: 优势函数值
            returns: 回报
        """
        # 处理空列表情况
        if len(rewards) == 0:
            return [], []
            
        # 计算最后一个状态的值函数估计
        if not dones[-1]:
            state = self.memory.states[-1]
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                last_value = self.critic(state_tensor).cpu().numpy()[0]
        else:
            last_value = 0.0
            
        # 计算优势函数和回报
        values = np.array(values + [last_value])
        advantages = []
        returns = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns

def make_env(env_id):
    """创建环境的包装函数"""
    # ARS实现一般使用较长的episode长度，例如1000步
    try:
        # 对于MuJoCo环境，默认的max_episode_steps通常是1000
        # 但我们显式设置它以确保一致性
        env = gym.make(env_id, max_episode_steps=1000, terminate_when_unhealthy=True)
        logger.info(f"创建环境: {env_id}, 最大步数: 1000, 不健康时终止: True")
    except Exception as e:
        try:
            # 如果第一次尝试失败，尝试不带terminate_when_unhealthy参数
            env = gym.make(env_id, max_episode_steps=1000)
            logger.info(f"创建环境: {env_id}, 最大步数: 1000")
        except Exception as e2:
            logger.warning(f"创建环境时无法指定max_episode_steps: {e2}")
            env = gym.make(env_id)
            logger.info(f"使用默认参数创建环境: {env_id}")
    
    # 创建一个奖励转换的包装器，使PPO使用与ARS相同的reward处理方式
    env = RewardScaleWrapper(env)
        
    # 打印环境信息以便调试
    logger.info(f"环境信息: {env_id}")
    logger.info(f"观测空间: {env.observation_space}")
    logger.info(f"动作空间: {env.action_space}")
    
    try:
        logger.info(f"最大episode步数: {env.spec.max_episode_steps}")
        
        # 检查环境设置是否正确
        if env.spec.max_episode_steps < 1000:
            logger.warning(f"环境的最大步数可能过小: {env.spec.max_episode_steps}, 这可能导致episodes过早结束")
    except:
        logger.info("无法获取最大episode步数")
    
    return env

# 添加一个奖励缩放包装器
class RewardScaleWrapper(RewardWrapperBase):
    """
    包装环境以对奖励进行处理，保持环境原始奖励定义
    
    该类继承自utils.RewardWrapperBase，确保正确处理奖励
    """
    def __init__(self, env, shift=None):
        super(RewardScaleWrapper, self).__init__(env, algorithm_type="PPO", shift=shift)
        # PPO算法不需要对奖励进行shift处理，保持原始奖励
        logger.info(f"PPO使用环境原始奖励, 环境:{env.spec.id}")
    
    def step(self, action):
        # 使用原始步骤实现，确保不改变奖励
        return self.env.step(action)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

@ray.remote
class PPOWorker:
    """用于并行收集经验和测试的工作进程"""
    
    def __init__(self, env_id, seed=0):
        """初始化工作进程"""
        # 为MuJoCo设置环境变量
        self._setup_mujoco_env()
        
        # 创建环境
        self.env = make_env(env_id)
        
        # 设置随机种子
        np.random.seed(seed)
        self.env.action_space.seed(seed)
        self.env.reset(seed=seed)
        
        # 初始化Actor和Critic网络
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.action_bound = [self.action_low, self.action_high]
        
        # 设置设备
        self.device = 'cpu'  # 工作进程始终使用CPU
        
        # 状态归一化
        self.normalize_states = False  # 默认不使用状态归一化
        self.state_normalizer = None
        
        # 内存缓冲区
        self.memory = None
        
        # 工作进程ID
        self.worker_id = seed
        
        print(f"工作进程 {self.worker_id} 初始化完成")
    
    def _setup_mujoco_env(self):
        """设置MuJoCo环境变量 - 优化版"""
        # 环境变量已通过runtime_env传递，不需要重复设置
        # 这能显著加快Worker初始化速度
        pass
    
    def collect_experiences(self, actor_weights, critic_weights, max_steps=1000, gamma=0.99, gae_lambda=0.95):
        """
        收集经验
        
        参数:
            actor_weights: Actor网络权重
            critic_weights: Critic网络权重
            max_steps: 要收集的步数
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            
        返回:
            experiences: 收集的经验字典
        """
        try:
            # 创建本地Actor和Critic网络，使用与主模型相同的hidden_dim=1024
            actor = ActorNetwork(self.state_dim, self.action_dim, self.action_bound, hidden_dim=1024).to(self.device)
            critic = CriticNetwork(self.state_dim, hidden_dim=1024).to(self.device)
            
            # 加载权重
            actor.load_state_dict(actor_weights)
            critic.load_state_dict(critic_weights)
            
            # 设置为评估模式
            actor.eval()
            critic.eval()
            
            # 初始化缓冲区
            states = []
            actions = []
            rewards = []
            dones = []
            log_probs = []
            values = []
            
            # 收集经验
            state, _ = self.env.reset()
            done = False
            
            steps = 0
            episode_rewards = []  # 完整回合的奖励
            current_episode_reward = 0
            episodes_completed = 0
            episode_steps = 0  # 当前episode的步数
            
            # 检查环境是否使用了奖励转换
            is_reward_wrapped = isinstance(self.env, RewardScaleWrapper)
            if is_reward_wrapped:
                print(f"工作进程 {self.worker_id} 使用了奖励转换包装器，shift={self.env.shift}")
            
            # 设置收集条件：至少完成3个回合或收集到足够的步数
            min_episodes = 3
            
            # 确保收集足够的经验 - 至少完成min_episodes个回合并且步数不少于max_steps
            while steps < max_steps or episodes_completed < min_episodes:
                # 如果步数已经大大超过了max_steps且已完成至少一个回合，则停止收集
                if steps > max_steps * 2 and episodes_completed > 0:
                    print(f"工作进程 {self.worker_id} 步数已大大超过限制，停止收集: {steps} > {max_steps * 2}")
                    break
                
                # 转换状态为张量
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # 采样动作
                with torch.no_grad():
                    action, log_prob, _, _ = actor.sample_action(state_tensor)
                    value = critic(state_tensor).cpu().numpy()[0]
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action[0])
                done = terminated or truncated
                
                # 保存转换
                states.append(state)
                actions.append(action[0])
                rewards.append(reward)
                dones.append(done)
                log_probs.append(log_prob[0])
                values.append(value)
                
                # 更新状态和回合奖励
                state = next_state
                current_episode_reward += reward
                steps += 1
                episode_steps += 1
                
                # 如果回合结束
                if done:
                    # 记录回合奖励
                    episode_rewards.append(float(current_episode_reward))
                    print(f"工作进程 {self.worker_id} 完成回合，奖励: {current_episode_reward:.2f}, 步数: {episode_steps}")
                    
                    # 重置环境
                    state, _ = self.env.reset()
                    episodes_completed += 1
                    current_episode_reward = 0
                    episode_steps = 0
                    
                    # 如果已经完成足够的回合并且收集了足够的步数，则停止
                    if episodes_completed >= min_episodes and steps >= max_steps:
                        break
            
            # 如果最后一个回合尚未结束但已经收集了足够的步数，也记录当前的不完整回合奖励
            ongoing_episode_reward = None
            if current_episode_reward > 0:
                ongoing_episode_reward = float(current_episode_reward)
                print(f"工作进程 {self.worker_id} 不完整回合，当前奖励: {current_episode_reward:.2f}, 当前步数: {episode_steps}")
            
            # 计算优势函数和回报
            advantages, returns = self._compute_advantages_and_returns(rewards, values, dones, gamma, gae_lambda)
            
            # 检查reward量级
            if episode_rewards:
                avg_reward = sum(episode_rewards) / len(episode_rewards)
                print(f"工作进程 {self.worker_id} 平均回合奖励: {avg_reward:.2f}, 最小: {min(episode_rewards):.2f}, 最大: {max(episode_rewards):.2f}")
            
            # 返回经验
            result = {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'dones': dones,
                'log_probs': log_probs,
                'values': values,
                'advantages': advantages,
                'returns': returns,
                'steps': steps,
                'episode_rewards': episode_rewards,
                'episodes_completed': episodes_completed,
                'ongoing_episode_reward': ongoing_episode_reward  # 添加不完整回合的奖励
            }
            
            print(f"工作进程 {self.worker_id} 收集完成: {steps} 步, {episodes_completed} 个回合")
            return result
            
        except Exception as e:
            import traceback
            print(f"工作进程 {self.worker_id} 收集经验时出错: {e}")
            print(traceback.format_exc())
            # 返回空经验
            return {
                'states': [],
                'actions': [],
                'rewards': [],
                'dones': [],
                'log_probs': [],
                'values': [],
                'advantages': [],
                'returns': [],
                'steps': 0,
                'episode_rewards': [],
                'episodes_completed': 0,
                'ongoing_episode_reward': None,
                'error': str(e)
            }
    
    def _compute_advantages_and_returns(self, rewards, values, dones, gamma, gae_lambda):
        """
        计算广义优势估计(GAE)和回报
        
        参数:
            rewards: 奖励列表
            values: 价值估计列表
            dones: 终止标志列表
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            
        返回:
            advantages: 优势函数值
            returns: 回报
        """
        # 检查是否有足够的数据
        if len(rewards) == 0:
            return [], []
            
        # 计算GAE
        advantages = []
        returns = []
        gae = 0
        
        # 如果最后一个状态未终止，获取它的价值估计
        if not dones[-1]:
            next_value = values[-1]  # 使用最后一个状态的价值估计
        else:
            next_value = 0.0
        
        for t in reversed(range(len(rewards))):
            # 计算TD误差
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            
            # 计算GAE
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            
            # 更新next_value
            next_value = values[t]
            
            # 存储优势和回报
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
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
        # 创建本地Actor网络，确保在CPU上，使用与主模型相同的hidden_dim=512
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high
        action_bound = [action_low, action_high]
        
        # 明确指定设备为CPU
        actor = ActorNetwork(state_dim, action_dim, action_bound, hidden_dim=512).to('cpu')
        
        # 加载权重，确保在CPU上
        try:
            actor.load_state_dict(actor_weights)
        except Exception as e:
            logger.error(f"Worker加载测试模型权重时出错: {e}")
            # 重新尝试，确保权重在CPU上
            cpu_actor_weights = {k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in actor_weights.items()}
            actor.load_state_dict(cpu_actor_weights)
        
        # 设置为评估模式
        actor.eval()
        
        # 测试策略
        total_rewards = []
        total_steps = 0
        
        # 检查环境是否使用了奖励转换
        is_reward_wrapped = isinstance(self.env, RewardScaleWrapper)
        if is_reward_wrapped:
            print(f"工作进程 {self.worker_id} 使用了奖励转换包装器，shift={self.env.shift}")
        
        for episode in range(num_episodes):
            episode_reward = 0
            state, _ = self.env.reset()
            done = False
            step_count = 0
            
            while not done:
                if render:
                    self.env.render()
                
                # 选择动作
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, _ = actor.sample_action(state_tensor)
                
                # 执行动作 - 如果需要，记录原始奖励
                if is_reward_wrapped and step_count == 0:
                    # 在第一步直接访问环境以获取原始奖励
                    orig_obs, orig_reward, orig_term, orig_trunc, orig_info = self.env.env.step(action[0])
                    # 使用包装器获取转换后的奖励
                    next_state, reward, terminated, truncated, _ = self.env.step(action[0])
                    print(f"工作进程 {self.worker_id} 第一步奖励对比: 原始={orig_reward:.2f}, 转换后={reward:.2f}")
                    # 使用包装器的状态更新，忽略原始环境的状态
                else:
                    next_state, reward, terminated, truncated, _ = self.env.step(action[0])
                
                done = terminated or truncated
                
                # 更新状态和奖励
                state = next_state
                episode_reward += reward
                step_count += 1
                total_steps += 1
            
            total_rewards.append(episode_reward)
            print(f"工作进程 {self.worker_id} 测试回合 {episode+1}/{num_episodes}, 奖励: {episode_reward:.2f}, 步数: {step_count}")
        
        # 打印统计信息
        if total_rewards:
            avg_reward = sum(total_rewards) / len(total_rewards)
            min_reward = min(total_rewards)
            max_reward = max(total_rewards)
            avg_steps = total_steps / num_episodes
            
            print(f"工作进程 {self.worker_id} 测试完成: {num_episodes}回合, 平均奖励: {avg_reward:.2f}, " +
                  f"最小: {min_reward:.2f}, 最大: {max_reward:.2f}, 平均步数: {avg_steps:.1f}")
        
        return total_rewards
    
    def test_connection(self):
        """测试工作进程连接"""
        return {"status": "ok", "worker_id": self.worker_id, "env_id": self.env.unwrapped.spec.id}

# 示例用法
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='PPO算法训练和测试')
    parser.add_argument('--env_id', type=str, default='HalfCheetah-v4', help='环境ID')
    parser.add_argument('--timesteps', type=int, default=1000000, help='训练总步数')
    parser.add_argument('--num_workers', type=int, default=4, help='并行工作进程数量')
    parser.add_argument('--parallel', action='store_true', help='使用并行训练')
    parser.add_argument('--test', action='store_true', help='测试模型')
    parser.add_argument('--save_path', type=str, default=None, help='模型保存路径')
    parser.add_argument('--load_path', type=str, default=None, help='模型加载路径')
    parser.add_argument('--log_interval', type=int, default=1, help='日志记录间隔')
    parser.add_argument('--save_interval', type=int, default=10, help='模型保存间隔')
    parser.add_argument('--log_dir', type=str, default=None, help='日志目录')
    
    args = parser.parse_args()
    
    # 确保Ray已初始化
    if args.parallel and not ray.is_initialized():
        ray.init()
    
    # 创建环境
    env = make_env(args.env_id)
    
    # 创建PPO实例
    ppo = PPO(env, 
              hidden_dim=1024,
              actor_lr=2e-4,
              critic_lr=1e-3,
              gamma=0.99, 
              gae_lambda=0.97,
              clip_ratio=0.2,
              ent_coef=0.01,
              batch_size=512,
              update_epochs=5,
              policy_update_interval=2048,
              num_workers=args.num_workers,
              log_dir=args.log_dir)
    
    # 加载模型
    if args.load_path:
        ppo.load_model(args.load_path)
        logger.info(f"模型加载自: {args.load_path}")
    
    # 测试模型
    if args.test:
        if args.parallel:
            rewards, mean_reward = ppo.test_parallel(args.env_id, num_episodes=10, render=False)
        else:
            rewards, mean_reward = ppo.test(num_episodes=10, render=False)
        logger.info(f"测试完成。平均奖励: {mean_reward:.2f}")
    else:
        # 训练模型
        logger.info(f"开始{'并行' if args.parallel else '串行'}训练...")
        
        env_id = args.env_id if args.parallel else None
        episode_rewards, mean_rewards = ppo.train(
            total_timesteps=args.timesteps, 
            env_id=env_id,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            save_path=args.save_path
        )
        
        # 保存训练数据
        ppo.save_training_data()
        
        # 测试最终模型
        if args.parallel:
            ppo.test_parallel(args.env_id, num_episodes=10, render=False)
        else:
            ppo.test(num_episodes=10, render=False) 