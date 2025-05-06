import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import time
import json
import torch.nn.functional as F
# Import the registry
from src.registry import BaseOptimizer, OptimizerRegistry

@OptimizerRegistry.register("zero_order")
class ZeroOrderOptimizer(BaseOptimizer):
    """Zero-order optimization algorithm for database parameter tuning."""
    
    def __init__(self, env, model=None, critic=None, learning_rate=1e-3, noise_std=None, noise_decay=None, 
                 lr_decay=None, decay_step=None, norm_rewards=None, train_min_size=32, 
                 size_mem=2000, size_predict_mem=2000, **kwargs):
        """Initialize the zero-order optimizer.
        
        Args:
            env: The environment to optimize
            model: The actor/policy model to optimize (should be provided by the caller)
            critic: The critic model for value estimation (optional)
            learning_rate: Learning rate for optimization
            noise_std: Standard deviation of noise for perturbation
            noise_decay: Decay rate for noise standard deviation
            lr_decay: Decay rate for learning rate
            decay_step: Number of steps after which to decay learning rate and noise
            norm_rewards: Whether to normalize rewards
            train_min_size: Minimum batch size for training
            size_mem: Memory size for experience replay
            size_predict_mem: Memory size for prediction
            **kwargs: Additional optimizer-specific arguments
        """
        super().__init__(env, **kwargs)
        
        self.learning_rate = learning_rate
        self.train_min_size = train_min_size
        self.epsilon = 0.9  # For exploration
        self.epsilon_decay = 0.999
        self.gamma = 0.095
        self.tau = 0.125
        
        # Log the initialization parameters
        self.log_json("init_config", {
            "learning_rate": learning_rate,
            "train_min_size": train_min_size,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "gamma": self.gamma,
            "tau": self.tau
        })
        
        # Memory for experience replay
        self.memory = deque(maxlen=size_mem)
        self.mem_predicted = deque(maxlen=size_predict_mem)
        
        # Store the provided actor and critic models
        self.actor = model
        if self.actor is None:
            raise ValueError("Actor model must be provided")
            
        self.critic = critic
        self.target_actor = None
        self.target_critic = None
        
        # Create target networks if critic is provided
        if self.critic is not None:
            # Make deep copies for target networks
            self.target_actor = type(self.actor)()
            self.target_actor.load_state_dict(self.actor.state_dict())
            
            self.target_critic = type(self.critic)()
            self.target_critic.load_state_dict(self.critic.state_dict())
            
            # Set environment reference for the critic if needed
            if hasattr(self.critic, 'env') and not self.critic.env:
                self.critic.env = env
                self.target_critic.env = env
                
            # Create critic optimizer
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Get configuration values from kwargs or use defaults
        config_dict = kwargs.get("config_dict", {})
        zero_order_config = config_dict.get("zero_order", {})
        
        try:
            self._noise_std = noise_std if noise_std is not None else float(zero_order_config.get("noise_std", 1e-3))
            self.noise_decay = noise_decay if noise_decay is not None else float(zero_order_config.get("noise_decay", 0.99))
            self.lr_decay = lr_decay if lr_decay is not None else float(zero_order_config.get("lr_decay", 0.99))
            self.decay_step = decay_step if decay_step is not None else int(zero_order_config.get("decay_step", 50))
            self.norm_rewards = norm_rewards if norm_rewards is not None else zero_order_config.get("norm_rewards", "true").lower() == "true"
        except Exception as e:
            # If any error occurs reading from config, use default values
            print(f"Error reading zero_order config, using defaults: {e}")
            self._noise_std = 1e-3 if noise_std is None else noise_std
            self.noise_decay = 0.99 if noise_decay is None else noise_decay
            self.lr_decay = 0.99 if lr_decay is None else lr_decay
            self.decay_step = 50 if decay_step is None else decay_step
            self.norm_rewards = True if norm_rewards is None else norm_rewards
        
        # Log the zero-order specific configuration
        self.log_json("zero_order_config", {
            "noise_std": self._noise_std,
            "noise_decay": self.noise_decay,
            "lr_decay": self.lr_decay,
            "decay_step": self.decay_step,
            "norm_rewards": self.norm_rewards
        })
        
        # Set the environment attribute on the model for the optimizer to access
        if hasattr(self.actor, 'env') and not self.actor.env:
            self.actor.env = self.env
        
        # Initialize the zero-order optimization algorithm
        self.zero_order_opt = ZeroOrderAlgorithm(
            model=self.actor,
            learning_rate=learning_rate,
            noise_std=self._noise_std,
            noise_decay=self.noise_decay,
            lr_decay=self.lr_decay,
            decay_step=self.decay_step,
            norm_rewards=self.norm_rewards
        )
        
        # Initialize tracking variables
        self.best_params = None
        self.best_throughput = 0

    def remember(self, cur_state, action, reward, new_state, done):
        """Store experience in memory for replay."""
        self.memory.append([cur_state, action, reward, new_state, done])
        
        # Log the experience
        self.log_state({
            "state_shape": str(cur_state.shape),
            "action_shape": str(action.shape),
            "reward": reward[0] if isinstance(reward, np.ndarray) else reward,
            "done": done
        }, len(self.memory))

    def train(self, i=0):
        """Train the model using zero-order optimization.
        
        Args:
            i: Current iteration (used for epsilon decay)
        """
        if len(self.memory) < self.train_min_size:
            return
        
        # Sample batch from memory
        batch_size = min(len(self.memory), 32)
        indexes = np.random.choice(len(self.memory), batch_size, replace=False)
        samples = [self.memory[i] for i in indexes]
        
        # Log training start
        self.log_training({
            "batch_size": batch_size,
            "memory_size": len(self.memory),
            "epsilon": self.epsilon,
            "noise_std": self.zero_order_opt.noise_std,
            "learning_rate": self.zero_order_opt.lr
        }, i)
        
        # Only train critic if it's provided
        if self.critic is not None:
            self._train_critic(samples, i)
            
        # Train actor using zero-order optimization
        self._train_actor(samples, i)
        
        # Update target networks if they exist
        if self.target_actor is not None and self.target_critic is not None:
            self.update_target()

        # Decay epsilon
        if i > 0:
            self.epsilon = self.epsilon * self.epsilon_decay
            
        # Log training end
        self.log_training({
            "epsilon_after": self.epsilon,
            "training_completed": True
        }, i)

    def update_target(self):
        """Soft update of target networks"""
        if self.target_actor is None or self.target_critic is None:
            return
            
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def _train_critic(self, samples, i):
        """Train the critic network.
        
        Args:
            samples: Batch of experiences from memory
        """
        if self.critic is None or self.target_critic is None:
            return 0.0
            
        total_critic_loss = 0.0
        
        for idx, sample in enumerate(samples):
            cur_state, action, reward, new_state, done = sample
            
            # Normalize state
            if len(cur_state.shape) > 1 and cur_state.shape[0] > 0:
                cur_state = (cur_state - min(cur_state[0])) / (max(cur_state[0]) - min(cur_state[0]) + 1e-10)
            
            if len(new_state.shape) > 1 and new_state.shape[0] > 0:
                new_state = (new_state - min(new_state[0]))/(max(new_state[0])-min(new_state[0]) + 1e-10)

            # Create tensors
            cur_state_tensor = torch.FloatTensor(cur_state).float()
            action_tensor = torch.FloatTensor(action).float()
            reward_tensor = torch.FloatTensor(reward).float()
            new_state_tensor = torch.FloatTensor(new_state).float()
            done_tensor = torch.FloatTensor(np.array([done]).astype(np.float32))
            
            # Ensure tensors have batch dimension
            if len(cur_state_tensor.shape) == 1:
                cur_state_tensor = cur_state_tensor.unsqueeze(0)
            if len(action_tensor.shape) == 1:
                action_tensor = action_tensor.unsqueeze(0)
            if len(reward_tensor.shape) == 1:
                reward_tensor = reward_tensor.unsqueeze(0)
            if len(new_state_tensor.shape) == 1:
                new_state_tensor = new_state_tensor.unsqueeze(0)
            
            # Get Q-value prediction
            self.target_critic.eval()
            with torch.no_grad():
                target_action = self.target_actor(new_state_tensor).detach()
                future_reward = self.target_critic(new_state_tensor, target_action)[0][0].detach().numpy()
            self.target_critic.train()

            reward += self.gamma * future_reward
            
            # 修改target_value的构建方式，确保精度和类型正确
            target_value = torch.tensor([[float(reward)]], dtype=torch.float32, requires_grad=False)
            
            # 确保critic处于训练模式
            self.critic.train()
            
            # Ensure tensors have requires_grad=True for gradient computation
            cur_state_tensor.requires_grad = True
            action_tensor.requires_grad = True
            
            critic_value = self.critic(cur_state_tensor, action_tensor)
            
            critic_loss = nn.MSELoss()(critic_value, target_value)
                
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)  # Add retain_graph=True to keep gradients
            
            # 梯度裁剪，防止梯度消失或爆炸
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            
            self.critic_optimizer.step()
            
            total_critic_loss += critic_loss.item()

            # Log every few samples for debugging
            if idx % 5 == 0:
                self.log_training({
                    "component": "critic",
                    "sample_idx": idx,
                    "critic_loss": float(critic_loss.item()),
                    "future_reward": float(future_reward),
                    "target_value": float(target_value.numpy())
                }, len(self.memory))
                    
        # Log average critic loss
        avg_critic_loss = total_critic_loss / len(samples)
        self.log_training({
            "component": "critic",
            "avg_critic_loss": float(avg_critic_loss)
        }, len(self.memory))
        
        return avg_critic_loss

    def _train_actor(self, samples, i):
        """Train the actor network using zero-order optimization.
        
        Args:
            samples: Batch of experiences from memory
            i: Current iteration
        """
        total_actor_loss = 0.0

        for idx, sample in enumerate(samples):
            cur_state, action, reward, new_state, _ = sample

            # Normalize state
            if len(cur_state.shape) > 1 and cur_state.shape[0] > 0:
                cur_state = (cur_state - min(cur_state[0]))/(max(cur_state[0])-min(cur_state[0]) + 1e-10)

            cur_state_tensor = torch.FloatTensor(cur_state).float()
            
            # Get current actor output and critic value
            action_tensor = self.actor(cur_state_tensor)
            reward_original = 0.0
            
            # If critic exists, get baseline value
            if self.critic is not None:
                reward_original = self.critic(cur_state_tensor, action_tensor)
                
            # Generate population for zero-order optimization
            population = self.zero_order_opt.generate_population(npop=len(samples))
            rewards = []
            
            # Log population generation
            self.log_training({
                "population_size": len(population),
                "stage": "population_generation"
            }, i)
            
            # Evaluate each model in population
            for model_idx, model in enumerate(population):
                # Normalize state
                if len(cur_state.shape) > 1 and cur_state.shape[0] > 0:
                    cur_state = (cur_state - min(cur_state[0]))/(max(cur_state[0])-min(cur_state[0]) + 1e-10)
                
                cur_state_tensor = torch.FloatTensor(cur_state).float()
                
                # Ensure tensor has batch dimension
                if len(cur_state_tensor.shape) == 1:
                    cur_state_tensor = cur_state_tensor.unsqueeze(0)
                
                # Set model to evaluation mode
                model.eval()
                with torch.no_grad():
                    predicted_action = model(cur_state_tensor).detach().numpy()
                # Set model back to training mode
                model.train()
                
                # Calculate critic gradients if critic exists
                critic_value = 0.0
                if self.critic is not None:
                    critic_state_tensor = torch.FloatTensor(cur_state).float()
                    if len(critic_state_tensor.shape) == 1:
                        critic_state_tensor = critic_state_tensor.unsqueeze(0)
                        
                    critic_action_tensor = torch.FloatTensor(predicted_action).float()
                    if len(critic_action_tensor.shape) == 1:
                        critic_action_tensor = critic_action_tensor.unsqueeze(0)
                    
                    critic_action_tensor.requires_grad = True
                    
                    # Set critic to evaluation mode
                    self.critic.eval()
                    critic_value = self.critic(critic_state_tensor, critic_action_tensor)
                    
                    # Set critic back to training mode
                    self.critic.train()
                    
                    # Use improvement over current policy as reward
                    rewards.append((critic_value - reward_original).detach().numpy().item())
                else:
                    # If no critic, use environment reward directly
                    # This is just a fallback and might not be optimal
                    rewards.append(reward[0] if isinstance(reward, np.ndarray) else reward)
                
                # Log model evaluation
                if model_idx % 10 == 0:  # Log every 10th model to avoid excessive logging
                    self.log_training({
                        "model_idx": model_idx,
                        "critic_value": float(critic_value.item()) if isinstance(critic_value, torch.Tensor) else 0.0,
                        "stage": "model_evaluation"
                    }, i)
            
            # Log rewards statistics before update
            self.log_training({
                "rewards_mean": float(np.mean(rewards)),
                "rewards_std": float(np.std(rewards)),
                "rewards_min": float(np.min(rewards)),
                "rewards_max": float(np.max(rewards)),
                "stage": "before_update"
            }, i)
            
            # Update population using zero-order optimization
            self.zero_order_opt.update_population(np.array(rewards))
            
            # Log after update
            self.log_training({
                "stage": "after_update",
                "update_complete": True
            }, i)

        # Calculate average actor loss (this is just for logging)
        avg_actor_loss = total_actor_loss / len(samples) if len(samples) > 0 else 0.0
        self.log_training({
            "component": "actor",
            "avg_actor_loss": float(avg_actor_loss)
        }, len(self.memory))
        
        return avg_actor_loss

    def act(self, state):
        """Choose an action based on the current state using epsilon-greedy strategy.
        
        Args:
            state: Current state
            
        Returns:
            tuple: (action, is_predicted, action_tmp)
        """
        # Normalize state
        if len(state.shape) > 1 and state.shape[0] > 0:
            state = (state - min(state[0]))/(max(state[0])-min(state[0]) + 1e-10)
        
        state_tensor = torch.FloatTensor(state).float()
        
        # Ensure tensor has batch dimension
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        if np.random.random() < self.epsilon:
            # Random action
            is_predicted = 0
            action = np.random.uniform(
                self.env.a_low, 
                self.env.a_high, 
                size=self.env.action_space.shape[0]
            )
            action_tmp = np.zeros_like(action)
            
            # Log random action selection
            self.log_action({
                "action_type": "random",
                "epsilon": self.epsilon,
                "action_shape": str(action.shape)
            }, len(self.memory))
        else:
            # Use actor to predict action
            is_predicted = 1
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(state_tensor).detach().numpy().flatten()
            self.actor.train()
            action_tmp = action.copy()
            
            # Log model-based action selection
            self.log_action({
                "action_type": "model",
                "epsilon": self.epsilon,
                "action_shape": str(action.shape)
            }, len(self.memory))
        
        return action, is_predicted, action_tmp

    def optimize(self, num_trials: int) -> Tuple[Dict[str, float], float]:
        """Run the zero-order optimization algorithm.
        
        Args:
            num_trials: Number of optimization iterations to run
            
        Returns:
            tuple: The best parameters found and the best performance metric achieved
        """
        print("\n------ Starting Database Parameter Tuning with Zero-Order Optimization ------\n")
        
        # Log optimization start
        self.log_json("optimization_start", {
            "num_trials": num_trials,
            "state_dim": self.env.observation_space.shape[0],
            "action_dim": self.env.action_space.shape[0]
        })
        
        # Initialize state
        cur_state = self.env._get_obs()
        cur_state = cur_state.reshape((1, self.env.state.shape[0]))
        
        # Initialize action
        action = self.env.fetch_action()
        action_2 = action.reshape((1, self.env.knob_num))
        action_2 = action_2[:, :self.env.action_space.shape[0]]
        
        # Apply first action and get initial state
        new_state, reward, score, cur_throughput = self.env.step(action, 0, 1)
        new_state = new_state.reshape((1, self.env.state.shape[0]))
        reward_np = np.array([reward])
        
        # Store initial experience
        self.remember(cur_state, action_2, reward_np, new_state, 0)
        
        # Log initial action and result
        self.log_evaluation({
            "iteration": 0,
            "action_type": "initial",
            "reward": reward,
            "score": score,
            "throughput": cur_throughput
        }, 0)
        
        # Initialize best performance tracking
        self.best_throughput = cur_throughput
        self.best_params = {k: v for k, v in zip(self.env.db.knob_names, action)}
        
        # Log initial best performance
        self.log_json("initial_best", {
            "throughput": self.best_throughput,
            "params": self.best_params
        })
        
        # Main optimization loop
        for i in range(1, num_trials + 1):
            # Create new action
            cur_state = new_state
            action, is_predicted, action_tmp = self.act(cur_state)
            
            # Apply action and get new state
            new_state, reward, score, throughput = self.env.step(action, is_predicted, i + 1)
            new_state = new_state.reshape((1, self.env.state.shape[0]))
            
            # Log evaluation results
            self.log_evaluation({
                "is_predicted": is_predicted,
                "reward": reward,
                "score": score,
                "throughput": throughput,
                "action_type": "model" if is_predicted else "random"
            }, i)
            
            # Store experience
            reward_np = np.array([reward])
            action = action.reshape((1, self.env.knob_num))
            action_2 = action[:, :self.env.action_space.shape[0]]
            self.remember(cur_state, action_2, reward_np, new_state, 0)
            
            # Train the model
            self.train(i)
            
            # Update best performance if improved
            if throughput > self.best_throughput:
                improvement = (throughput - self.best_throughput) / self.best_throughput * 100
                self.best_throughput = throughput
                self.best_params = {k: v for k, v in zip(self.env.db.knob_names, action.flatten())}
                
                # Log new best performance
                self.log_json("new_best", {
                    "iteration": i,
                    "throughput": self.best_throughput,
                    "improvement_percentage": improvement,
                    "params": self.best_params
                })
                
                print(f"\nNew best throughput at iteration {i}: {self.best_throughput}")
                print(f"Improvement: {improvement:.2f}%")
            
            # Print progress
            if i % 5 == 0:
                print(f"Iteration {i}/{num_trials} completed. Current throughput: {throughput}, Best so far: {self.best_throughput}")
        
        # Save best parameters to file
        result_file = f'training-results/zero_order_best_params_{self.timestamp}.json'
        with open(result_file, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        print(f"\n------ Optimization Complete ------")
        print(f"Best throughput: {self.best_throughput}")
        print(f"Best parameters saved to: {result_file}")
        
        # Log final results
        self.log_json("optimization_complete", {
            "num_iterations": i,
            "best_throughput": self.best_throughput,
            "best_params": self.best_params,
            "result_file": result_file
        })
        
        return self.best_params, self.best_throughput


class ZeroOrderAlgorithm:
    """Zero-order optimization algorithm implementation."""
    
    def __init__(self, model, learning_rate=1e-3, noise_std=1e-3, 
                 noise_decay=0.99, lr_decay=0.99, decay_step=50, norm_rewards=True):
        """Initialize the zero-order optimization algorithm.
        
        Args:
            model: Model to optimize
            learning_rate: Initial learning rate
            noise_std: Noise standard deviation for perturbation
            noise_decay: Decay rate for noise standard deviation
            lr_decay: Decay rate for learning rate
            decay_step: Number of steps after which to decay parameters
            norm_rewards: Whether to normalize rewards
        """
        self.model = model
        self._lr = learning_rate
        self._noise_std = noise_std
        self.noise_decay = noise_decay
        self.lr_decay = lr_decay
        self.decay_step = decay_step
        self.norm_rewards = norm_rewards
        self._count = 0
        
        # Get environment reference for action range if available
        self.env = None
        if hasattr(model, 'env'):
            self.env = model.env
    
    @property
    def noise_std(self):
        """Current noise standard deviation with decay applied."""
        if self.decay_step > 0:
            decay = self.noise_decay ** (self._count // self.decay_step)
            return self._noise_std * decay
        return self._noise_std
    
    @property
    def lr(self):
        """Current learning rate with decay applied."""
        if self.decay_step > 0:
            decay = self.lr_decay ** (self._count // self.decay_step)
            return self._lr * decay
        return self._lr
    
    def generate_population(self, npop=10):
        """Generate a population of perturbed models.
        
        Args:
            npop: Population size
            
        Returns:
            list: List of perturbed models
        """
        population = []
        
        # Store original parameters
        original_params = [p.data.clone() for p in self.model.parameters()]
        
        for _ in range(npop):
            # Create a model of the same class as the original model
            try:
                # Try to create a model with the same constructor arguments as the original
                if hasattr(self.model, 'state_dim') and hasattr(self.model, 'action_dim'):
                    # For standard RL models
                    kwargs = {}
                    if hasattr(self.model, 'max_action'):
                        kwargs['max_action'] = self.model.max_action
                    new_model = type(self.model)(
                        state_dim=self.model.state_dim,
                        action_dim=self.model.action_dim,
                        **kwargs
                    )
                else:
                    # Generic case
                    new_model = type(self.model)()
            except Exception as e:
                print(f"Error creating model: {e}")
                # Fall back to deep copy
                new_model = type(self.model)()
                for attr_name in dir(self.model):
                    if attr_name.startswith('__') or callable(getattr(self.model, attr_name)):
                        continue
                    try:
                        setattr(new_model, attr_name, getattr(self.model, attr_name))
                    except:
                        pass
            
            # Copy the environment reference if it exists
            if hasattr(self.model, 'env'):
                new_model.env = self.model.env
            
            # Copy the original parameters
            for idx, param in enumerate(new_model.parameters()):
                param.data.copy_(original_params[idx])
            
            # Add parameter perturbations
            new_model.E = []  # Store perturbation directions
            for param in new_model.parameters():
                # Gaussian noise with standard deviation = noise_std
                perturbation = torch.randn_like(param.data) * self.noise_std
                new_model.E.append(perturbation.clone())
                param.data.add_(perturbation)
            
            population.append(new_model)
        
        return population
    
    def update_population(self, rewards):
        """Update the model based on population performance.
        
        Args:
            rewards: Array of rewards for each model in the population
        """
        # Normalize rewards if needed
        if self.norm_rewards and len(rewards) > 1:
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        
        # Update parameters
        for i, param in enumerate(self.model.parameters()):
            # Initialize update
            update = torch.zeros_like(param.data)
            
            # Sum the perturbations weighted by rewards
            for j, model in enumerate(self._population):
                update.add_(model.E[i] * rewards[j])
            
            # Apply update with learning rate
            param.data.add_(self.lr * update / (len(rewards) * self.noise_std))
        
        # Increment counter for decay calculations
        self._count += 1
    
    def get_model(self):
        """Get the optimized model.
        
        Returns:
            The optimized model
        """
        return self.model 