"""
Script to run and compare different optimizers for the ARS algorithm.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import ray
import socket
import argparse
from ars import run_ars
import multiprocessing as mp

def main():
    parser = argparse.ArgumentParser(description='Compare different optimizers for ARS')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4',
                        help='OpenAI Gym environment name')
    parser.add_argument('--n_iter', type=int, default=100,
                       help='Number of iterations for each optimizer')
    parser.add_argument('--n_directions', type=int, default=16,
                       help='Number of directions/perturbations per iteration')
    parser.add_argument('--deltas_used', type=int, default=16,
                       help='Number of top-performing directions to use')
    parser.add_argument('--step_size', type=float, default=0.02,
                       help='Step size parameter')
    parser.add_argument('--delta_std', type=float, default=0.03,
                       help='Standard deviation of perturbations')
    parser.add_argument('--n_workers', type=int, default=8,
                       help='Number of parallel workers')
    parser.add_argument('--seed', type=int, default=237,
                       help='Random seed')
    parser.add_argument('--optimizers', type=str, nargs='+',
                        default=['sgd', 'adam', 'zero_order', 'multi_scale'],
                        help='List of optimizers to compare')
    args = parser.parse_args()
    
    # Set up environment variables for MuJoCo
    os.environ["MUJOCO_PY_MUJOCO_PATH"] = "/home/liangchen/.mujoco/mujoco210/"
    if "LD_LIBRARY_PATH" not in os.environ:
        os.environ["LD_LIBRARY_PATH"] = ""
    if "/home/liangchen/.mujoco/mujoco210/bin" not in os.environ["LD_LIBRARY_PATH"]:
        os.environ["LD_LIBRARY_PATH"] += ":/home/liangchen/.mujoco/mujoco210/bin"
    if "/usr/lib/nvidia" not in os.environ["LD_LIBRARY_PATH"]:
        os.environ["LD_LIBRARY_PATH"] += ":/usr/lib/nvidia"
    
    # Create output directory
    os.makedirs('optimizer_comparison', exist_ok=True)
    
    # Initialize Ray if it's not already initialized
    try:
        ray.init(address="auto", ignore_reinit_error=True)
    except:
        local_ip = socket.gethostbyname(socket.gethostname())
        # Set up runtime environment to pass environment variables to all workers
        runtime_env = {
            "env_vars": {
                "MUJOCO_PY_MUJOCO_PATH": "/home/liangchen/.mujoco/mujoco210/",
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "") + ":/home/liangchen/.mujoco/mujoco210/bin:/usr/lib/nvidia"
            }
        }
        ray.init(address=f"{local_ip}:6379", runtime_env=runtime_env)
    
    # Run each optimizer and store results
    results = {}
    timestamp = int(time.time())
    
    for optimizer in args.optimizers:
        print(f"\n{'='*80}")
        print(f"Running ARS with {optimizer} optimizer")
        print(f"{'='*80}\n")
        
        # Create parameters dictionary
        params = {
            'env_name': args.env_name,
            'n_iter': args.n_iter,
            'n_directions': args.n_directions,
            'deltas_used': args.deltas_used,
            'step_size': args.step_size,
            'delta_std': args.delta_std,
            'n_workers': args.n_workers,
            'rollout_length': 1000,  # Default value
            'shift': 0,  # Default for HalfCheetah
            'seed': args.seed,
            'policy_type': 'linear',
            'dir_path': f'optimizer_comparison/{optimizer}_{timestamp}',
            'filter': 'MeanStdFilter',  # Default value
            'optimizer_type': optimizer
        }
        
        # Run ARS with the current optimizer
        run_ars(params)
        
        # Store the directory path for this optimizer's results
        results[optimizer] = params['dir_path']
    
    # Plot comparison of results
    plot_comparison(results, args.n_iter, f'optimizer_comparison/comparison_{timestamp}.png')
    
    # Shutdown Ray
    ray.shutdown()
    print("\nAll optimizers completed. Results saved to optimizer_comparison/")

def plot_comparison(result_dirs, n_iter, output_path):
    """
    Plot a comparison of the different optimizers' performance.
    
    Args:
        result_dirs (dict): Dictionary mapping optimizer names to result directories
        n_iter (int): Number of iterations
        output_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    for optimizer, dir_path in result_dirs.items():
        # Read the average reward data from the progress.txt file
        progress_file = os.path.join(dir_path, 'progress.txt')
        
        if not os.path.exists(progress_file):
            print(f"Warning: Progress file not found for {optimizer} at {progress_file}")
            continue
        
        try:
            # Load the data
            data = np.genfromtxt(progress_file, names=True, skip_header=0,
                                 dtype=None, deletechars='', encoding='utf-8')
            
            # Extract iteration and average reward
            iterations = data['Iteration']
            rewards = data['AverageReward']
            
            # Plot the data
            plt.plot(iterations, rewards, label=f'{optimizer}')
        except Exception as e:
            print(f"Error plotting data for {optimizer}: {e}")
    
    plt.title('Comparison of Optimizer Performance in ARS')
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"Comparison plot saved to {output_path}")

if __name__ == '__main__':
    # Set the main method to spawn to avoid any issues
    mp.set_start_method('spawn', force=True)
    main() 