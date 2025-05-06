#!/bin/bash

# 设置 MuJoCo 环境变量
export MUJOCO_PY_MUJOCO_PATH="/home/liangchen/.mujoco/mujoco210/"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/liangchen/.mujoco/mujoco210/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/nvidia"

# 先停止现有的 Ray 实例
ray stop

# 启动 Ray 并确保它知道 LD_LIBRARY_PATH
ray start --head --redis-port=6379 --num-workers=18

# 运行 ARS
python code/ars.py

# 脚本结束后停止 Ray
ray stop 