import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import gym
import time
import ray
import os
import logging
import matplotlib.pyplot as plt
import datetime
import json
import csv
from copy import deepcopy

from src.models.ppo import PPO, ActorNetwork, CriticNetwork, PPOMemory, make_env
from src.optimizers.multi_scale_zero_order_optimizer import MultiScaleZeroOrderAlgorithm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MSZooPPO(PPO):
    """使用多尺度零阶优化器的PPO实现"""
    
    def __init__(self, env, hidden_dim=1024, actor_lr=2e-4, critic_lr=1e-3,
                 gamma=0.99, gae_lambda=0.97, clip_ratio=0.2, target_kl=0.01,
                 ent_coef=0.01, batch_size=512, update_epochs=5, 
                 max_grad_norm=0.5, policy_update_interval=2048,
                 num_workers=8, log_dir=None, normalize_states=True, 
                 mszoo_config=None):
        """
        初始化多尺度零阶优化的PPO算法
        
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
            batch_size: 批次大小
            update_epochs: 每次策略更新的epoch数
            max_grad_norm: 梯度裁剪的最大范数
            policy_update_interval: 策略更新间隔的环境步数
            num_workers: 并行工作进程数量
            log_dir: 日志目录，如果为None则创建基于时间戳的目录
            normalize_states: 是否对状态进行归一化
            mszoo_config: 多尺度零阶优化器的配置参数
        """
        # 调用父类初始化方法
        super(MSZooPPO, self).__init__(
            env=env, 
            hidden_dim=hidden_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            target_kl=target_kl,
            ent_coef=ent_coef,
            batch_size=batch_size,
            update_epochs=update_epochs,
            max_grad_norm=max_grad_norm,
            policy_update_interval=policy_update_interval,
            num_workers=num_workers,
            log_dir=log_dir,
            normalize_states=normalize_states
        )
        
        # 多尺度零阶优化器配置
        self.mszoo_config = mszoo_config or {}
        
        # 将必要的属性添加到actor模型上，使其与优化器兼容
        self._prepare_actor_for_mszoo()
        
        # 创建多尺度零阶优化器，替换标准的Actor优化器
        self.ms_optimizer = self._create_ms_optimizer()
        
        # 记录使用的优化器类型
        logger.info("使用多尺度零阶优化器替代标准Actor优化器")

    def _prepare_actor_for_mszoo(self):
        """为多尺度零阶优化准备Actor模型"""
        # 添加state_dim和action_dim属性（如果没有的话）
        if not hasattr(self.actor, 'state_dim'):
            self.actor.state_dim = self.env.observation_space.shape[0]
        if not hasattr(self.actor, 'action_dim'):
            self.actor.action_dim = self.env.action_space.shape[0]
        
        # 添加环境引用
        if not hasattr(self.actor, 'env'):
            self.actor.env = self.env
            
        # 添加max_action属性（如果适用）
        if hasattr(self.env.action_space, 'high'):
            self.actor.max_action = float(self.env.action_space.high[0])

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

    def update_policy(self):
        """
        使用多尺度零阶优化方式更新策略
        
        返回:
            avg_policy_loss: 平均策略损失
            avg_value_loss: 平均值函数损失
            avg_entropy: 平均熵
            avg_kl: 平均KL散度
        """
        # 从记忆缓冲区中获取经验
        states, actions, old_log_probs, rewards, dones, values = self.memory.get_all()
        
        # 将NumPy数组转换为PyTorch张量并移动到设备上
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
        
        # 多尺度零阶优化更新策略网络
        # 与标准PPO不同，我们不再使用梯度下降更新Actor，而是使用MSZoo
        
        # 为每个批次生成多个扰动的Actor模型
        population_size = self.mszoo_config.get("population_size", 10)
        
        # 准备模型群体，用于每个扰动半径
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
                model_reward, metrics = self._evaluate_perturbed_model(
                    model, states, actions, advantages, old_log_probs
                )
                model_rewards.append(model_reward)
                radius_metrics.append(metrics)
                
                # 累加指标以计算平均值
                total_policy_loss += metrics['policy_loss']
                total_entropy += metrics['entropy']  
                total_kl += metrics['kl']
            
            all_rewards.append(model_rewards)
            all_metrics.append(radius_metrics)
        
        # 更新多尺度优化器的权重
        if hasattr(self.ms_optimizer, '_count') and self.ms_optimizer._count % self.ms_optimizer.weight_update_interval == 0:
            self._update_optimizer_weights(all_rewards)
        
        # 对每个扰动半径应用更新
        for radius_idx in range(len(self.ms_optimizer.perturbation_radii)):
            rewards = np.array(all_rewards[radius_idx])
            
            # 只有当rewards不全为零时才更新
            if np.any(rewards != 0):
                # 获取此半径的权重
                if hasattr(self.ms_optimizer, 'radius_weights'):
                    weight = self.ms_optimizer.radius_weights[radius_idx]
                else:
                    weight = 1.0 / len(self.ms_optimizer.perturbation_radii)  # 简单的平均权重
                
                # 更新模型
                self.ms_optimizer.update_population(rewards, radius_idx, weight)
        
        # 更新Critic网络（使用标准梯度下降）
        for epoch in range(self.update_epochs):
            # 打乱数据
            np.random.shuffle(indices)
            batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
            
            for batch_idx in batches:
                batch_states = states[batch_idx]
                batch_returns = returns[batch_idx]
                
                # 更新Critic网络
                value_pred = self.critic(batch_states)
                value_loss = F.mse_loss(value_pred, batch_returns)
                
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_value_loss += value_loss.item()
        
        # 清空缓冲区
        self.memory.clear()
        
        # 计算平均值（所有模型和所有半径）
        total_models = sum(len(population) for population in all_populations)
        avg_policy_loss = total_policy_loss / total_models if total_models > 0 else 0
        avg_entropy = total_entropy / total_models if total_models > 0 else 0
        avg_kl = total_kl / total_models if total_models > 0 else 0
        
        # 计算critic的平均损失
        num_batches = len(indices) // batch_size + (1 if len(indices) % batch_size != 0 else 0)
        num_epochs = self.update_epochs  
        avg_value_loss = total_value_loss / (num_batches * num_epochs) if num_batches * num_epochs > 0 else 0
        
        return avg_policy_loss, avg_value_loss, avg_entropy, avg_kl
    
    def _evaluate_perturbed_model(self, model, states, actions, advantages, old_log_probs):
        """
        评估扰动模型的表现
        
        参数:
            model: 扰动后的模型
            states: 状态批次
            actions: 动作批次
            advantages: 优势函数
            old_log_probs: 旧策略的动作对数概率
            
        返回:
            reward: 模型的奖励评分（越高越好）
            metrics: 包含policy_loss, entropy, kl等指标的字典
        """
        model.eval()
        with torch.no_grad():
            # 计算新的动作对数概率
            new_log_probs, entropy = model.evaluate_action(states, actions)
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 计算PPO目标函数
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            
            # 策略损失
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 加入熵正则项
            policy_loss_with_entropy = policy_loss - self.ent_coef * entropy.mean()
            
            # 计算KL散度
            kl = (old_log_probs - new_log_probs).mean().item()
            
            # 返回负损失作为奖励信号
            reward = -policy_loss_with_entropy.item()
            
            # 如果KL散度过大，惩罚奖励
            if kl > 2 * self.target_kl:
                reward = reward * 0.5
            
            # 返回评估指标
            metrics = {
                'policy_loss': policy_loss.item(),
                'entropy': entropy.mean().item(),
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
        max_steps = getattr(self.env, 'max_steps', 500)
        is_exploration_phase = self.ms_optimizer._count <= 0.6 * max_steps
        
        # 根据阶段不同，采用不同的权重分配策略
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
        # 这里我们假设优化器有一个radius_weights属性
        if hasattr(self.ms_optimizer, 'radius_weights'):
            self.ms_optimizer.radius_weights = normalized_weights
        
        # 记录权重更新
        logger.info(f"更新优化器权重: {normalized_weights}, 阶段: {'探索' if is_exploration_phase else '收敛'}")


# 用于测试
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Scale Zero-Order PPO')
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
    
    # 多尺度零阶优化器配置
    mszoo_config = {
        "perturbation_radii": [0.01, 0.02, 0.05],
        "population_size": 10, 
        "weight_update_interval": 20,
        "noise_std": 0.15,
        "noise_decay": 0.995
    }
    
    # 创建MSZooPPO实例
    ppo = MSZooPPO(
        env, 
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
        log_dir=args.log_dir,
        mszoo_config=mszoo_config
    )
    
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