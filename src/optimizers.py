# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# OPTIMIZERS FOR MINIMIZING OBJECTIVES
class Optimizer(object):
    def __init__(self, w_policy, step_size):
        self.w_policy = w_policy
        self.step_size = step_size
        self.dim = w_policy.size
    
    def _compute_step(self, g_hat):
        raise NotImplementedError
        
    def update(self, g_hat):
        step = self._compute_step(g_hat)
        self.w_policy -= step
        return self.w_policy

class SGD(Optimizer):
    def _compute_step(self, g_hat):
        step = self.step_size * g_hat
        return step

class AdamOptimizer(Optimizer):
    def __init__(self, w_policy, step_size, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super(AdamOptimizer, self).__init__(w_policy, step_size)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros_like(w_policy.flatten())
        self.v = np.zeros_like(w_policy.flatten())
        self.t = 0
    
    def _compute_step(self, g_hat):
        self.t += 1
        g_flat = g_hat.flatten()
        self.m = self.beta1 * self.m + (1 - self.beta1) * g_flat
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(g_flat)
        m_hat = self.m / (1 - np.power(self.beta1, self.t))
        v_hat = self.v / (1 - np.power(self.beta2, self.t))
        step_flat = self.step_size * m_hat / (np.sqrt(v_hat) + self.epsilon)
        # Reshape the step to match the shape of w_policy if needed
        if hasattr(self.w_policy, 'shape') and len(self.w_policy.shape) > 1:
            return step_flat.reshape(self.w_policy.shape)
        return step_flat

class ZeroOrderOptimizer(Optimizer):
    def __init__(self, w_policy, step_size, noise_std=0.02, noise_decay=0.999, 
                 lr_decay=0.999, decay_step=10, norm_rewards=True):
        super(ZeroOrderOptimizer, self).__init__(w_policy, step_size)
        self._noise_std = noise_std
        self.noise_decay = noise_decay
        self.lr_decay = lr_decay
        self.decay_step = decay_step
        self.norm_rewards = norm_rewards
        self._count = 0
        
    @property
    def noise_std(self):
        """Get the current noise standard deviation with decay applied."""
        step_decay = np.power(self.noise_decay, np.floor((1 + self._count) / self.decay_step))
        return self._noise_std * step_decay

    @property
    def lr(self):
        """Get the current learning rate with decay applied."""
        step_decay = np.power(self.lr_decay, np.floor((1 + self._count) / self.decay_step))
        return self.step_size * step_decay
    
    def _compute_step(self, g_hat):
        # Apply current learning rate
        step = self.lr * g_hat
        self._count += 1
        # Ensure the step has the same shape as the policy weights
        if hasattr(self.w_policy, 'shape') and len(self.w_policy.shape) > 1:
            if len(g_hat.shape) == 1:
                return step.reshape(self.w_policy.shape)
        return step

class MultiScaleZeroOrderOptimizer(Optimizer):
    def __init__(self, w_policy, step_size, perturbation_radii=None, noise_std=0.02, 
                 noise_decay=0.999, lr_decay=0.999, decay_step=10, norm_rewards=True):
        super(MultiScaleZeroOrderOptimizer, self).__init__(w_policy, step_size)
        if perturbation_radii is None:
            perturbation_radii = [0.01, 0.05, 0.1]
        self.perturbation_radii = perturbation_radii
        self._noise_std = noise_std
        self.noise_decay = noise_decay
        self.lr_decay = lr_decay
        self.decay_step = decay_step
        self.norm_rewards = norm_rewards
        self._count = 0
        
        # Initialize weights for each perturbation radius (equal initially)
        self.radius_weights = [1.0 / len(self.perturbation_radii)] * len(self.perturbation_radii)
        
        # Initialize adaptive radius learning rates
        self.radius_learning_rates = {}
        for idx, radius in enumerate(self.perturbation_radii):
            # Small perturbation radius uses smaller learning rate, large radius uses larger learning rate
            if radius == min(self.perturbation_radii):
                self.radius_learning_rates[idx] = step_size * 0.8
            elif radius == max(self.perturbation_radii):
                self.radius_learning_rates[idx] = step_size * 1.2
            else:
                self.radius_learning_rates[idx] = step_size
    
    @property
    def noise_std(self):
        """Get the current noise standard deviation with decay applied."""
        step_decay = np.power(self.noise_decay, np.floor((1 + self._count) / self.decay_step))
        return self._noise_std * step_decay

    @property
    def lr(self):
        """Get the current learning rate with decay applied."""
        step_decay = np.power(self.lr_decay, np.floor((1 + self._count) / self.decay_step))
        return self.step_size * step_decay
    
    def _compute_step(self, g_hat):
        # Apply learning rate based on adaptive weights
        step = self.lr * g_hat
        self._count += 1
        # Ensure the step has the same shape as the policy weights
        if hasattr(self.w_policy, 'shape') and len(self.w_policy.shape) > 1:
            if len(g_hat.shape) == 1:
                return step.reshape(self.w_policy.shape)
        return step
    
    def update_radius_weights(self, rewards_by_radius):
        """Update weights for different perturbation radii based on rewards.
        
        Args:
            rewards_by_radius: List of rewards for each perturbation radius
        """
        # Simple softmax weighting based on mean rewards
        mean_rewards = [np.mean(rewards) for rewards in rewards_by_radius]
        exp_rewards = np.exp(mean_rewards)
        self.radius_weights = exp_rewards / np.sum(exp_rewards)
        return self.radius_weights

