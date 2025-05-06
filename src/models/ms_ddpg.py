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
from torch.distributions import Normal
import csv

# 导入原始DDPG实现和多尺度零阶优化器
from src.models.ddpg import DDPG, ActorNetwork, CriticNetwork, OUNoise, ReplayBuffer, make_env
from src.optimizers.multi_scale_zero_order_optimizer import MultiScaleZeroOrderAlgorithm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MSZooDDPG(DDPG):
    """使用多尺度零阶优化器的DDPG实现"""
    
    def __init__(self, env, hidden_dim=256, actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, tau=0.005, batch_size=64, buffer_size=1000000,
                 noise_theta=0.15, noise_sigma=0.2, mszoo_config=None):
        """
        初始化多尺度零阶优化的DDPG算法
        
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
            mszoo_config: 多尺度零阶优化器的配置参数
        """
        # 调用父类初始化方法
        super(MSZooDDPG, self).__init__(
            env=env, 
            hidden_dim=hidden_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            tau=tau,
            batch_size=batch_size,
            buffer_size=buffer_size,
            noise_theta=noise_theta,
            noise_sigma=noise_sigma
        )
        
        # 多尺度零阶优化器配置
        self.mszoo_config = mszoo_config or {}
        
        # 创建多尺度零阶优化器，替换标准的Actor优化器
        self.ms_optimizer = self._create_ms_optimizer()
        
        # 记录使用的优化器类型
        logger.info("使用多尺度零阶优化器替代标准Actor优化器")
        
    def _create_ms_optimizer(self):
        """创建多尺度零阶优化器"""
        # 从配置中获取参数，或使用默认值
        perturbation_radii = self.mszoo_config.get(
            "perturbation_radii", [0.01, 0.02, 0.05]
        )
        if isinstance(perturbation_radii, str):
            perturbation_radii = list(map(float, perturbation_radii.split(',')))
            
        weight_update_interval = self.mszoo_config.get("weight_update_interval", 20)
        noise_std = self.mszoo_config.get("noise_std", 0.15)
        noise_decay = self.mszoo_config.get("noise_decay", 0.995)
        
        # 确保actor有环境引用，这对于MultiScaleZeroOrderAlgorithm是必要的
        if not hasattr(self.actor, 'env'):
            self.actor.env = self.env
        
        # 为Actor添加一些MultiScaleZeroOrderAlgorithm需要的属性
        if not hasattr(self.actor, 'state_dim'):
            self.actor.state_dim = self.state_dim
        if not hasattr(self.actor, 'action_dim'):
            self.actor.action_dim = self.action_dim
            
        # 创建多尺度零阶优化器
        optimizer = MultiScaleZeroOrderAlgorithm(
            model=self.actor,
            learning_rate=self.actor_optimizer.param_groups[0]['lr'],
            perturbation_radii=perturbation_radii,
            weight_update_interval=weight_update_interval,
            noise_std=noise_std,
            noise_decay=noise_decay
        )
        
        # 记录优化器配置
        logger.info(f"多尺度零阶优化器配置: 扰动半径={perturbation_radii}, " + 
                   f"权重更新间隔={weight_update_interval}, " + 
                   f"噪声标准差={noise_std}, 噪声衰减={noise_decay}")
        
        return optimizer
    
    def update(self):
        """使用多尺度零阶优化方式更新网络参数"""
        # 从经验回放缓冲区采样
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0, 0, 0  # 如果缓冲区中的样本不足，则跳过更新，返回四个指标
            
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).reshape(-1, 1).to(device)
        
        # 更新Critic网络 - 使用标准梯度下降
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
        
        # === 使用多尺度零阶优化更新Actor网络 ===
        # 生成多个扰动的Actor模型，用于每个扰动半径
        population_size = self.mszoo_config.get("population_size", 10)
        all_populations = []
        all_rewards = []
        all_metrics = []
        
        # 为每个扰动半径生成模型群体
        for radius_idx in range(len(self.ms_optimizer.perturbation_radii)):
            # 生成此半径的模型群体
            population = self.ms_optimizer.generate_population(
                npop=population_size,
                radius_idx=radius_idx
            )
            all_populations.append(population)
            
            # 评估每个模型
            model_rewards = []
            radius_metrics = []
            for model in population:
                model_reward, metrics = self._evaluate_perturbed_actor(model, states)
                model_rewards.append(model_reward)
                radius_metrics.append(metrics)
            
            all_rewards.append(model_rewards)
            all_metrics.append(radius_metrics)
        
        # 更新多尺度优化器的权重
        if hasattr(self.ms_optimizer, '_count') and self.ms_optimizer._count % self.ms_optimizer.weight_update_interval == 0:
            self._update_optimizer_weights(all_rewards)
        
        # 应用每个扰动半径的更新
        for radius_idx in range(len(self.ms_optimizer.perturbation_radii)):
            rewards = np.array(all_rewards[radius_idx])
            
            # 只有当rewards不全为零时才更新
            if np.any(rewards != 0):
                # 获取此半径的权重
                if hasattr(self.ms_optimizer, 'radius_weights'):
                    weight = self.ms_optimizer.radius_weights[radius_idx]
                else:
                    weight = 1.0 / len(self.ms_optimizer.perturbation_radii)  # 简单平均权重
                
                # 更新模型
                self.ms_optimizer.update_population(rewards, radius_idx, weight)
        
        # 软更新目标网络
        self._update_target_networks()
        
        # 计算整体指标用于记录
        avg_policy_loss = 0.0
        avg_entropy = 0.0
        avg_kl = 0.0
        total_models = 0
        
        # 计算所有模型的平均指标
        for radius_metrics in all_metrics:
            for metrics in radius_metrics:
                avg_policy_loss += metrics['policy_loss']
                avg_entropy += metrics['entropy']
                avg_kl += metrics['kl']
                total_models += 1
                
        if total_models > 0:
            avg_policy_loss /= total_models
            avg_entropy /= total_models
            avg_kl /= total_models
        
        return critic_loss.item(), avg_policy_loss, avg_entropy, avg_kl
    
    def _evaluate_perturbed_actor(self, model, states):
        """
        评估扰动Actor模型的表现
        
        参数:
            model: 扰动后的Actor模型
            states: 状态批次
            
        返回:
            reward: 模型的奖励评分（越高越好）
            metrics: 包含policy_loss, entropy, kl等指标的字典
        """
        model.eval()
        with torch.no_grad():
            # 评估当前状态下模型产生的动作
            perturbed_actions = model(states)
            
            # 使用Critic网络评估动作的Q值
            q_values = self.critic(states, perturbed_actions)
            
            # DDPG的Actor是确定性策略，没有概率分布，因此计算一个简单的替代熵值
            # 使用动作的多样性作为熵的简单替代
            entropy = torch.std(perturbed_actions, dim=0).mean().item()
            
            # 计算KL散度 - 对于确定性策略，使用MSE作为距离度量
            current_actions = self.actor(states)  # DDPG的actor直接返回动作
            kl = F.mse_loss(perturbed_actions, current_actions).item()
            
            # 计算policy_loss - 这里是负Q值作为损失
            policy_loss = -q_values.mean().item()
            
            # 返回平均Q值作为奖励（越高越好）
            reward = q_values.mean().item()
            
            # 返回评估指标
            metrics = {
                'policy_loss': policy_loss,
                'entropy': entropy,
                'kl': kl
            }
        
        return reward, metrics
    
    def _update_optimizer_weights(self, all_rewards):
        """
        更新多尺度零阶优化器的权重
        
        参数:
            all_rewards: 所有扰动半径的奖励列表
        """
        # 计算每个扰动半径的平均奖励
        avg_rewards = [np.mean(rewards) for rewards in all_rewards]
        
        # 进入探索或收敛阶段的判断逻辑
        max_steps = 1000  # 估计的一个回合的最大步数
        is_exploration_phase = self.ms_optimizer._count <= 0.6 * max_steps
        
        # 根据不同阶段调整权重分配策略
        if is_exploration_phase:
            # 探索阶段：使用softmax为大扰动半径分配更高权重
            psi = np.random.uniform(5, 20)
            weights = []
            for radius in self.ms_optimizer.perturbation_radii:
                weight = np.exp(psi * radius)
                weights.append(weight)
        else:
            # 收敛阶段：奖励效果好的小扰动半径
            rho = 100  # 半径惩罚系数
            epsilon = 1e-8  # 避免除零
            
            weights = []
            for idx, radius in enumerate(self.ms_optimizer.perturbation_radii):
                # 计算方差或使用平均奖励
                if len(all_rewards[idx]) > 1:
                    variance = np.var(all_rewards[idx])
                else:
                    variance = epsilon
                
                # 小半径在收敛阶段获得更高权重
                if radius == min(self.ms_optimizer.perturbation_radii):
                    weight = np.exp(-rho * 0.5 * (radius ** 2)) / (variance + epsilon)
                else:
                    weight = np.exp(-rho * (radius ** 2)) / (variance + epsilon)
                
                weights.append(weight)
        
        # 归一化权重
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # 更新优化器权重
        if hasattr(self.ms_optimizer, 'radius_weights'):
            self.ms_optimizer.radius_weights = normalized_weights
        
        # 记录权重更新
        # logger.info(f"更新优化器权重: {normalized_weights}, 阶段: {'探索' if is_exploration_phase else '收敛'}")
        
    def train(self, total_timesteps, start_steps=10000, log_interval=10, save_interval=None, save_path=None):
        """
        训练MSZooDDPG算法
        
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
        
        # 记录每个优化步骤的损失，用于调试和监控
        critic_losses = []
        actor_rewards = []  # 记录MSZoo优化中Actor获得的奖励
        
        while timestep < total_timesteps:
            # 选择动作
            if timestep < start_steps:
                # 初始阶段随机采样动作
                action = self.env.action_space.sample()
            else:
                # 使用策略和噪声选择动作
                action = self.select_action(state, add_noise=True)
            
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
                for _ in range(1):  # 可以进行多次更新
                    critic_loss, actor_loss, entropy, kl = self.update()
                    critic_losses.append(critic_loss)
                    
                    # 记录Actor优化信息
                    if hasattr(self.ms_optimizer, 'radius_weights'):
                        actor_rewards.append({
                            'weights': self.ms_optimizer.radius_weights.copy(),
                            'count': self.ms_optimizer._count
                        })
            
            # 如果回合结束，重置环境
            if done:
                self.episode_rewards.append(episode_reward)
                self.mean_rewards.append(np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) > 0 else 0)
                
                # 记录数据到CSV文件
                mean_reward = self.mean_rewards[-1] if self.mean_rewards else 0
                
                # 获取最近的损失值
                critic_loss = critic_losses[-1] if critic_losses else 0
                
                # 确保我们有当前的指标
                # 使用最近计算的指标，如果有的话
                current_policy_loss = actor_loss if 'actor_loss' in locals() else 0
                current_entropy = entropy if 'entropy' in locals() else 0
                current_kl = kl if 'kl' in locals() else 0
                
                # 记录到CSV
                self.log_episode_data(episode_num, timestep, episode_reward, mean_reward, current_policy_loss, critic_loss, current_kl, current_entropy)
                
                if episode_num % log_interval == 0:
                    logger.info(f"Episode: {episode_num} | Timestep: {timestep} | Reward: {episode_reward:.2f} | Mean Reward: {mean_reward:.2f}")
                    
                    # 额外输出多尺度优化器的状态和训练指标
                    if hasattr(self.ms_optimizer, 'radius_weights'):
                        logger.info(f"MSZoo权重: {self.ms_optimizer.radius_weights}, 噪声标准差: {self.ms_optimizer.noise_std}")
                        logger.info(f"策略损失: {current_policy_loss:.6f} | 值函数损失: {critic_loss:.6f} | KL: {current_kl:.6f} | 熵: {current_entropy:.6f}")
                
                # 重置环境和噪声
                self.noise.reset()
                state, _ = self.env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
                
                # 保存模型
                if save_interval and save_path and episode_num % save_interval == 0:
                    self.save_model(os.path.join(save_path, f"ms_ddpg_model_{timestep}.pt"))
        
        # 训练结束，保存最终模型
        if save_path:
            self.save_model(os.path.join(save_path, "ms_ddpg_model_final.pt"))
            
            # 保存MSZoo配置
            with open(os.path.join(save_path, "ms_ddpg_config.json"), 'w') as f:
                import json
                json.dump({
                    'mszoo_config': self.mszoo_config,
                    'final_weights': self.ms_optimizer.radius_weights if hasattr(self.ms_optimizer, 'radius_weights') else None,
                    'noise_std': self.ms_optimizer.noise_std if hasattr(self.ms_optimizer, 'noise_std') else None,
                    'optimizer_count': self.ms_optimizer._count if hasattr(self.ms_optimizer, '_count') else 0
                }, f, indent=4)
        
        return self.episode_rewards, self.mean_rewards
    
    def save_model(self, path):
        """保存模型"""
        # 除了常规的模型参数外，还需要保存MSZoo优化器的状态
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'hyperparams': {
                'gamma': self.gamma,
                'tau': self.tau,
            },
            'mszoo_config': self.mszoo_config,
            'mszoo_state': {
                'radius_weights': self.ms_optimizer.radius_weights if hasattr(self.ms_optimizer, 'radius_weights') else None,
                'noise_std': self.ms_optimizer.noise_std if hasattr(self.ms_optimizer, 'noise_std') else None,
                'count': self.ms_optimizer._count if hasattr(self.ms_optimizer, '_count') else 0
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
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # 加载超参数
        hyperparams = checkpoint['hyperparams']
        self.gamma = hyperparams['gamma']
        self.tau = hyperparams['tau']
        
        # 加载MSZoo配置和状态
        if 'mszoo_config' in checkpoint:
            self.mszoo_config = checkpoint['mszoo_config']
            # 重新创建MSZoo优化器
            self.ms_optimizer = self._create_ms_optimizer()
            
            # 恢复MSZoo状态
            if 'mszoo_state' in checkpoint:
                mszoo_state = checkpoint['mszoo_state']
                if mszoo_state.get('radius_weights') and hasattr(self.ms_optimizer, 'radius_weights'):
                    self.ms_optimizer.radius_weights = mszoo_state['radius_weights']
                if mszoo_state.get('noise_std') is not None:
                    self.ms_optimizer._noise_std = mszoo_state['noise_std']
                if mszoo_state.get('count') is not None:
                    self.ms_optimizer._count = mszoo_state['count']
        
        logger.info(f"Model loaded from {path}")
    
    def log_episode_data(self, episode, timestep, reward, mean_reward, policy_loss=None, value_loss=None, kl=None, entropy=None):
        """重写记录回合数据到CSV文件的方法，增加调试信息"""
        try:
            # 确保self.rewards_file存在
            if not hasattr(self, 'rewards_file') or not self.rewards_file:
                logger.error("没有设置rewards_file路径!")
                return
                
            # 确保目录存在
            log_dir = os.path.dirname(self.rewards_file)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                logger.info(f"创建日志目录: {log_dir}")
            
            # 如果文件不存在，创建并写入标题行
            if not os.path.exists(self.rewards_file):
                with open(self.rewards_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Episode', 'Timestep', 'Reward', 'MeanReward', 'PolicyLoss', 'ValueLoss', 'KL', 'Entropy'])
                logger.info(f"创建新的CSV文件: {self.rewards_file}")
            
            # 写入数据
            with open(self.rewards_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode, timestep, reward, mean_reward, 
                                 policy_loss if policy_loss is not None else '',
                                 value_loss if value_loss is not None else '',
                                 kl if kl is not None else '',
                                 entropy if entropy is not None else ''])
            
            # 记录写入成功的日志
            if episode % 10 == 0:  # 每10个回合记录一次，避免日志过多
                logger.info(f"成功记录回合 {episode} 的数据到 {self.rewards_file}")
                
        except Exception as e:
            # 如果出现异常，记录详细错误信息
            import traceback
            logger.error(f"记录数据到CSV文件时出错: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 记录路径信息进行调试
            logger.error(f"尝试写入的文件路径: {self.rewards_file if hasattr(self, 'rewards_file') else 'rewards_file未设置'}")
            logger.error(f"当前工作目录: {os.getcwd()}")


# 测试代码
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Scale Zero-Order DDPG')
    parser.add_argument('--env_id', type=str, default='HalfCheetah-v4', help='环境ID')
    parser.add_argument('--timesteps', type=int, default=1000000, help='训练总步数')
    parser.add_argument('--batch_size', type=int, default=100, help='批次大小')
    parser.add_argument('--buffer_size', type=int, default=1000000, help='回放缓冲区大小')
    parser.add_argument('--start_steps', type=int, default=10000, help='开始训练前的随机动作步数')
    parser.add_argument('--test', action='store_true', help='测试模型')
    parser.add_argument('--save_path', type=str, default=None, help='模型保存路径')
    parser.add_argument('--load_path', type=str, default=None, help='模型加载路径')
    parser.add_argument('--log_interval', type=int, default=1, help='日志记录间隔')
    parser.add_argument('--save_interval', type=int, default=100, help='模型保存间隔')
    parser.add_argument('--render', action='store_true', help='渲染环境')
    
    args = parser.parse_args()
    
    # 创建环境
    env = make_env(args.env_id)
    
    # 多尺度零阶优化器配置
    mszoo_config = {
        "perturbation_radii": [0.01, 0.02, 0.05],
        "population_size": 10, 
        "weight_update_interval": 20,
        "noise_std": 0.15,
        "noise_decay": 0.995
    }
    
    # 创建MSZooDDPG实例
    ddpg = MSZooDDPG(
        env, 
        hidden_dim=256,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99, 
        tau=0.005,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        mszoo_config=mszoo_config
    )
    
    # 加载模型
    if args.load_path:
        ddpg.load_model(args.load_path)
        logger.info(f"模型加载自: {args.load_path}")
    
    # 测试模型
    if args.test:
        rewards, mean_reward = ddpg.test(num_episodes=10, render=args.render)
        logger.info(f"测试完成。平均奖励: {mean_reward:.2f}")
    else:
        # 训练模型
        logger.info(f"开始训练MSZooDDPG...")
        
        episode_rewards, mean_rewards = ddpg.train(
            total_timesteps=args.timesteps, 
            start_steps=args.start_steps,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            save_path=args.save_path
        )
        
        # 绘制奖励曲线
        ddpg.plot_rewards(save_path=os.path.join(os.path.dirname(args.save_path), 'ms_ddpg_rewards.png') if args.save_path else None)
        
        # 测试最终模型
        rewards, mean_reward = ddpg.test(num_episodes=10, render=args.render)
        logger.info(f"训练完成。最终测试平均奖励: {mean_reward:.2f}") 