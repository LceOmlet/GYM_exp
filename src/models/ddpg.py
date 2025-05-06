import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import time
import os
import logging
import matplotlib.pyplot as plt
from collections import deque
import random
import copy
import csv
import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OUNoise:
    """Ornstein-Uhlenbeck过程噪声"""
    
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.1, decay_period=100000):
        """
        初始化OU噪声过程
        
        参数:
            action_space: 动作空间
            mu: OU过程的均值
            theta: OU过程的速度参数
            max_sigma: 初始噪声尺度
            min_sigma: 最终噪声尺度
            decay_period: 噪声衰减的步数
        """
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()
        
    def reset(self):
        """重置噪声过程"""
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        """更新噪声状态"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        """为动作添加噪声，用于探索"""
        # 衰减噪声
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        self.sigma = sigma
        
        # 添加噪声
        noise = self.evolve_state()
        noisy_action = action + noise
        
        # 裁剪到动作空间
        return np.clip(noisy_action, self.low, self.high)

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity):
        """
        初始化经验回放缓冲区
        
        参数:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样经验"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        """返回缓冲区当前大小"""
        return len(self.buffer)

class ActorNetwork(nn.Module):
    """DDPG的Actor网络，输出确定性动作"""
    
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
        self.action_scale = (action_bound[1] - action_bound[0]) / 2.0
        self.action_bias = (action_bound[1] + action_bound[0]) / 2.0
        
        # 网络层定义
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)
        
    def forward(self, state):
        """前向传播，输出确定性动作"""
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        # Tanh输出[-1, 1]，然后缩放和偏移到动作空间
        action = torch.tanh(self.layer3(x))
        
        # 确保action_scale和action_bias在正确的设备上并且是tensor
        action_scale = torch.tensor(self.action_scale, dtype=torch.float32, device=action.device)
        action_bias = torch.tensor(self.action_bias, dtype=torch.float32, device=action.device)
        
        # 缩放到动作空间
        action = action * action_scale + action_bias
        
        return action

class CriticNetwork(nn.Module):
    """DDPG的Critic网络，评估状态-动作对的价值"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        初始化Critic网络
        
        参数:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
        """
        super(CriticNetwork, self).__init__()
        
        # 状态编码层
        self.state_layer = nn.Linear(state_dim, hidden_dim)
        
        # 动作编码层
        self.action_layer = nn.Linear(action_dim, hidden_dim)
        
        # 合并层
        self.layer1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)
        
    def forward(self, state, action):
        """前向传播，输出状态-动作价值"""
        # 处理状态和动作
        state_value = F.relu(self.state_layer(state))
        action_value = F.relu(self.action_layer(action))
        
        # 合并状态和动作表示
        concat = torch.cat([state_value, action_value], dim=1)
        
        # 输出Q值
        x = F.relu(self.layer1(concat))
        x = F.relu(self.layer2(x))
        q_value = self.layer3(x)
        
        return q_value

class DDPG:
    """DDPG算法实现"""
    
    def __init__(self, env, hidden_dim=256, actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, tau=0.005, batch_size=64, buffer_size=1000000,
                 noise_theta=0.15, noise_sigma=0.2):
        """
        初始化DDPG算法
        
        参数:
            env: 环境
            hidden_dim: 隐藏层维度
            actor_lr: Actor网络学习率
            critic_lr: Critic网络学习率
            gamma: 折扣因子
            tau: 目标网络软更新参数
            batch_size: 批次大小
            buffer_size: 经验回放缓冲区大小
            noise_theta: OU噪声的theta参数
            noise_sigma: OU噪声的sigma参数
        """
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self.action_bound = [self.action_low, self.action_high]
        
        # 超参数
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # 初始化网络
        self.actor = ActorNetwork(self.state_dim, self.action_dim, 
                                 self.action_bound, hidden_dim).to(device)
        self.target_actor = copy.deepcopy(self.actor).to(device)
        
        self.critic = CriticNetwork(self.state_dim, self.action_dim, 
                                   hidden_dim).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 噪声过程
        self.noise = OUNoise(env.action_space, theta=noise_theta, max_sigma=noise_sigma)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 训练统计
        self.episode_rewards = []
        self.mean_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        
        # 设置日志目录
        self.setup_logging()
    
    def setup_logging(self, log_dir=None):
        """设置日志目录和文件"""
        # 默认日志目录基于环境名称和时间戳
        if log_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            env_name = self.env.unwrapped.spec.id
            log_dir = f"logs/{env_name}_{timestamp}"
        
        # 创建日志子目录
        self.log_dir = log_dir
        self.models_dir = os.path.join(log_dir, 'models')
        self.plots_dir = os.path.join(log_dir, 'plots')
        self.data_dir = os.path.join(log_dir, 'data')
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 设置CSV日志文件路径
        self.rewards_file = os.path.join(self.data_dir, 'rewards.csv')
        self.updates_file = os.path.join(self.data_dir, 'updates.csv')
        
        # 创建新的CSV文件，并写入标题行
        with open(self.rewards_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Timestep', 'Reward', 'MeanReward', 'PolicyLoss', 'ValueLoss', 'KL', 'Entropy'])
        
        with open(self.updates_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Update', 'Timestep', 'PolicyLoss', 'ValueLoss', 'Entropy', 'KL', 'UpdateTime'])
        
        logger.info(f"日志目录设置完成: {self.log_dir}")
    
    def log_episode_data(self, episode, timestep, reward, mean_reward, policy_loss=None, value_loss=None, kl=None, entropy=None):
        """记录回合数据到CSV文件"""
        with open(self.rewards_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, timestep, reward, mean_reward, 
                             policy_loss if policy_loss is not None else '',
                             value_loss if value_loss is not None else '',
                             kl if kl is not None else '',
                             entropy if entropy is not None else ''])
    
    def select_action(self, state, add_noise=True, timestep=0):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).detach().cpu().numpy().flatten()
        self.actor.train()
        
        if add_noise:
            action = self.noise.get_action(action, timestep)
        
        return action
    
    def train(self, total_timesteps, start_steps=10000, log_interval=10, save_interval=None, save_path=None):
        """
        训练DDPG算法
        
        参数:
            total_timesteps: 总时间步数
            start_steps: 开始训练前的随机动作步数
            log_interval: 日志打印间隔
            save_interval: 模型保存间隔
            save_path: 模型保存路径
        """
        timestep = 0
        state, _ = self.env.reset()
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        
        # 创建保存路径
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
        
        while timestep < total_timesteps:
            # 选择动作
            if timestep < start_steps:
                # 初始阶段随机采样动作
                action = self.env.action_space.sample()
            else:
                # 使用策略和噪声选择动作
                action = self.select_action(state, add_noise=True, timestep=timestep)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # 存储经验
            self.replay_buffer.add(state, action, reward, next_state, done)
            
            # 更新状态
            state = next_state
            episode_reward += reward
            episode_timesteps += 1
            timestep += 1
            
            # 训练网络
            if len(self.replay_buffer) > self.batch_size:
                critic_loss, actor_loss = self.update()
                self.actor_losses.append(actor_loss)
                self.critic_losses.append(critic_loss)
            
            # 如果回合结束，重置环境
            if done:
                self.episode_rewards.append(episode_reward)
                self.mean_rewards.append(np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) > 0 else 0)
                
                # 记录数据到CSV文件
                mean_reward = self.mean_rewards[-1] if self.mean_rewards else 0
                actor_loss = self.actor_losses[-1] if self.actor_losses else 0
                critic_loss = self.critic_losses[-1] if self.critic_losses else 0
                self.log_episode_data(episode_num, timestep, episode_reward, mean_reward, actor_loss, critic_loss, 0, 0)
                
                if episode_num % log_interval == 0:
                    logger.info(f"Episode: {episode_num} | Timestep: {timestep} | Reward: {episode_reward}")
                
                # 重置环境和噪声
                self.noise.reset()
                state, _ = self.env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
                
                # 保存模型
                if save_interval and save_path and episode_num % save_interval == 0:
                    self.save_model(os.path.join(save_path, f"ddpg_model_{timestep}.pt"))
        
        # 训练结束，保存最终模型
        if save_path:
            self.save_model(os.path.join(save_path, "ddpg_model_final.pt"))
        
        return self.episode_rewards, self.mean_rewards
    
    def update(self):
        """更新网络参数"""
        # 从经验回放缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).reshape(-1, 1).to(device)
        
        # 更新Critic网络
        with torch.no_grad():
            # 目标动作
            target_actions = self.target_actor(next_states)
            # 目标Q值
            target_q = self.target_critic(next_states, target_actions)
            # 贝尔曼方程右侧
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # 当前Q值
        current_q = self.critic(states, actions)
        
        # 计算Critic损失
        critic_loss = F.mse_loss(current_q, target_q)
        
        # 更新Critic网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor网络
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self._update_target_networks()
        
        return critic_loss.item(), actor_loss.item()
    
    def _update_target_networks(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'hyperparams': {
                'gamma': self.gamma,
                'tau': self.tau
            }
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # 加载超参数
        hyperparams = checkpoint['hyperparams']
        self.gamma = hyperparams['gamma']
        self.tau = hyperparams['tau']
        
        logger.info(f"Model loaded from {path}")
    
    def test(self, num_episodes=10, render=False):
        """测试训练好的模型"""
        total_rewards = []
        
        for i in range(num_episodes):
            episode_reward = 0
            state, _ = self.env.reset()
            done = False
            
            while not done:
                if render:
                    self.env.render()
                
                # 选择动作（无噪声）
                action = self.select_action(state, add_noise=False)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # 更新状态和回报
                state = next_state
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            logger.info(f"Test Episode: {i+1} | Reward: {episode_reward}")
        
        mean_reward = np.mean(total_rewards)
        logger.info(f"Mean test reward over {num_episodes} episodes: {mean_reward}")
        
        return total_rewards, mean_reward
    
    def plot_rewards(self, save_path=None):
        """绘制回报曲线"""
        plt.figure(figsize=(12, 8))
        
        # 绘制回报曲线
        plt.subplot(2, 1, 1)
        plt.plot(self.episode_rewards, label='Episode Rewards', alpha=0.3)
        plt.plot(self.mean_rewards, label='Mean 100 Rewards', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('DDPG Training Rewards')
        plt.legend()
        plt.grid(True)
        
        # 绘制损失曲线
        plt.subplot(2, 1, 2)
        plt.plot(self.actor_losses, label='Actor Loss', alpha=0.5)
        plt.plot(self.critic_losses, label='Critic Loss', alpha=0.5)
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.title('DDPG Training Losses')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()

def make_env(env_id):
    """创建环境的包装函数"""
    env = gym.make(env_id)
    return env

# 示例用法
if __name__ == "__main__":
    env_id = "HalfCheetah-v4"
    env = make_env(env_id)
    
    # 创建DDPG实例
    ddpg = DDPG(env, 
                hidden_dim=256,
                actor_lr=1e-4,
                critic_lr=1e-3,
                gamma=0.99, 
                tau=0.005,
                batch_size=64,
                buffer_size=1000000)
    
    # 训练模型
    ddpg.train(total_timesteps=1000000, 
              start_steps=10000,
              log_interval=1,
              save_interval=100,
              save_path='models/ddpg')
    
    # 绘制回报曲线
    ddpg.plot_rewards(save_path='plots/ddpg_rewards.png')
    
    # 测试模型
    ddpg.test(num_episodes=10, render=True) 