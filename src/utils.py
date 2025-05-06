# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

import numpy as np
import torch
import gym
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def itergroups(items, group_size):
    """Iterates over items by groups of group_size."""
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield group
            group = []
    if group:
        yield group

def batched_weighted_sum(weights, vecs, batch_size):
    """Compute weighted sum of vectors in batches for efficiency.

    Args:
        weights (numpy.ndarray): weights for each vector.
        vecs (iterable): list of vectors to be summed.
        batch_size (int): batch size for each weighted sum computation.

    Returns:
        tuple: weighted sum of vectors and the number of items summed.
    """
    total = 0
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size),
                                         itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float64),
                      np.asarray(batch_vecs, dtype=np.float64))
        num_items_summed += len(batch_weights)
    return total, num_items_summed

def batched_iterable(iterable, batch_size):
    """Split an iterable into batches of a specified size.

    Args:
        iterable (iterable): the iterable to split.
        batch_size (int): the size of each batch.

    Yields:
        list: a batch of items from the iterable.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

class RewardProcessor:
    """
    通用的奖励处理类，用于RL算法的奖励处理。
    
    本类遵循环境原始定义的奖励计算方式，确保不改变环境的基本奖励结构。
    对于HalfCheetah等环境，原始奖励应为正值，表示向前移动的距离。
    """
    
    @staticmethod
    def get_shift_for_env(env_name):
        """
        根据环境名称确定恰当的shift值。
        
        注意：此方法仅用于兼容ARS算法的原始实现。对于大多数环境，
        我们应该遵循环境的原始奖励定义，不做不必要的修改。
        
        Args:
            env_name (str): 环境名称
            
        Returns:
            float: 对应环境的shift值
        """
        # ARS原始实现使用的shift值：
        if any(name in env_name for name in ['Hopper', 'Walker2d', 'Ant']):
            shift = 1.0
        elif 'Humanoid' in env_name:
            shift = 5.0
        else:  # HalfCheetah, Swimmer和其他环境
            shift = 0.0
            
        logger.info(f"环境 {env_name} 的ARS算法shift值: {shift}，但建议使用原始环境奖励定义")
        return shift
    
    @staticmethod
    def process_reward(reward, shift=0.0, algorithm_type=None):
        """
        处理奖励值，默认保持原始奖励不变
        
        Args:
            reward (float): 原始奖励
            shift (float): 偏移值，默认为0，不改变奖励
            algorithm_type (str): 算法类型，用于特定算法的兼容处理
            
        Returns:
            float: 处理后的奖励
        """
        # 默认情况下，我们应该保持环境原始奖励不变
        if algorithm_type == "ARS" and shift != 0:
            # 仅ARS算法且明确指定了非零shift值时应用shift
            return reward - shift
        else:
            # 其他情况保持原始奖励
            return reward


class RewardWrapperBase(gym.Wrapper):
    """
    基础奖励包装器，使用RewardProcessor处理奖励。
    默认情况下保持环境原始奖励不变。
    """
    
    def __init__(self, env, algorithm_type=None, shift=None):
        super(RewardWrapperBase, self).__init__(env)
        
        self.algorithm_type = algorithm_type
        
        # 如果未指定shift，根据环境名自动确定
        if shift is None:
            env_id = env.spec.id
            self.shift = RewardProcessor.get_shift_for_env(env_id)
        else:
            self.shift = shift
        
        if algorithm_type == "ARS" and self.shift != 0:
            logger.info(f"创建ARS专用奖励包装器: shift={self.shift}, env={env.spec.id}")
        else:
            logger.info(f"创建标准奖励包装器: 保持原始奖励不变, env={env.spec.id}")
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 使用RewardProcessor处理奖励
        processed_reward = RewardProcessor.process_reward(
            reward, 
            shift=self.shift,
            algorithm_type=self.algorithm_type
        )
        
        return obs, processed_reward, terminated, truncated, info
