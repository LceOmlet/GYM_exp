from typing import Dict, Type, Any, Callable, Optional
import torch
import torch.optim as optim
import numpy as np
import time
import os
import json
from datetime import datetime


class OptimizerRegistry:
    """Registry for optimization algorithms that can be used for database parameter tuning."""
    
    _optimizers: Dict[str, Type["BaseOptimizer"]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register an optimizer class with the registry.
        
        Args:
            name: The name to register the optimizer under
            
        Returns:
            A decorator function that registers the optimizer class
        """
        def decorator(optimizer_class: Type["BaseOptimizer"]) -> Type["BaseOptimizer"]:
            if name in cls._optimizers:
                raise ValueError(f"Optimizer with name '{name}' already registered")
            cls._optimizers[name] = optimizer_class
            return optimizer_class
        return decorator
    
    @classmethod
    def get_optimizer(cls, name: str, **kwargs) -> "BaseOptimizer":
        """Get an instance of the specified optimizer.
        
        Args:
            name: Name of the optimizer to get
            **kwargs: Arguments to pass to the optimizer constructor
            
        Returns:
            An instance of the requested optimizer
        
        Raises:
            ValueError: If an optimizer with the given name is not registered
        """
        if name not in cls._optimizers:
            available = ", ".join(cls._optimizers.keys())
            raise ValueError(f"Optimizer '{name}' not found. Available optimizers: {available}")
        
        optimizer_class = cls._optimizers[name]
        return optimizer_class(**kwargs)
    
    @classmethod
    def list_optimizers(cls) -> Dict[str, Type["BaseOptimizer"]]:
        """Get all registered optimizers.
        
        Returns:
            A dictionary of registered optimizer names to optimizer classes
        """
        return cls._optimizers.copy()


class BaseOptimizer:
    """Base class for all optimizers."""
    
    def __init__(self, env, **kwargs):
        """Initialize the optimizer.
        
        Args:
            env: The environment to optimize
            **kwargs: Additional optimizer-specific arguments
        """
        self.env = env
        self.kwargs = kwargs
        self.timestamp = int(time.time())
        
        # Initialize log files
        os.makedirs("training-results", exist_ok=True)
        
        # Log base information
        self._log_info = {
            "optimizer_type": self.__class__.__name__,
            "timestamp": self.timestamp,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": kwargs
        }
    
    def log(self, category: str, data: Dict[str, Any], iteration: Optional[int] = None):
        """Log data to a file in the training-results directory.
        
        Args:
            category: Category of the log (e.g., 'training', 'evaluation', 'action')
            data: Dictionary of data to log
            iteration: Optional iteration number to include in the log
        """
        log_file = f'training-results/{category}_{self.__class__.__name__.lower()}_{self.timestamp}.log'
        
        # Format the data
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "iteration": iteration
        }
        log_entry.update(data)
        
        # Write to file in tab-separated format for consistency
        with open(log_file, 'a') as f:
            # Convert the entry to a single line with tab-separated values
            if f.tell() == 0:  # If file is empty, write headers
                headers = "\t".join(log_entry.keys())
                f.write(f"{headers}\n")
            
            values = "\t".join(str(v) for v in log_entry.values())
            f.write(f"{values}\n")
    
    def log_training(self, data: Dict[str, Any], iteration: int):
        """Log training data.
        
        Args:
            data: Dictionary containing training metrics
            iteration: Current training iteration
        """
        self.log("training", data, iteration)
    
    def log_evaluation(self, data: Dict[str, Any], iteration: int):
        """Log evaluation data.
        
        Args:
            data: Dictionary containing evaluation metrics
            iteration: Current evaluation iteration
        """
        self.log("evaluation", data, iteration)
    
    def log_action(self, data: Dict[str, Any], iteration: int):
        """Log action data.
        
        Args:
            data: Dictionary containing action information
            iteration: Current action iteration
        """
        self.log("action", data, iteration)
        
    def log_state(self, data: Dict[str, Any], iteration: int):
        """Log state data.
        
        Args:
            data: Dictionary containing state information
            iteration: Current state iteration
        """
        self.log("state", data, iteration)
    
    def log_json(self, category: str, data: Dict[str, Any]):
        """Log data in JSON format.
        
        Args:
            category: Category of the log
            data: Dictionary of data to log in JSON format
        """
        log_file = f'training-results/{category}_{self.__class__.__name__.lower()}_{self.timestamp}.json'
        
        # Ensure data is JSON serializable
        json_data = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                json_data[k] = v.tolist()
            elif isinstance(v, np.generic):
                json_data[k] = v.item()
            else:
                json_data[k] = v
        
        with open(log_file, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def optimize(self, num_trials: int, **kwargs) -> tuple:
        """Perform optimization.
        
        Args:
            num_trials: Number of optimization iterations to run
            **kwargs: Additional optimizer-specific arguments
            
        Returns:
            tuple: The best parameters found and the best performance metric achieved
        """
        raise NotImplementedError("Subclasses must implement this method") 