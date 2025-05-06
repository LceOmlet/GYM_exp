#!/usr/bin/env python3
"""
修复 Gym 环境检查器中的 np.bool8 引用，因为较新版本的 NumPy 中已移除此类型
"""
import os
import sys

# 指定要修改的文件路径
gym_checker_path = '/home/liangchen/miniconda3/envs/ARS/lib/python3.12/site-packages/gym/utils/passive_env_checker.py'

# 确保文件存在
if not os.path.exists(gym_checker_path):
    print(f"错误: 文件不存在: {gym_checker_path}")
    sys.exit(1)

# 读取文件内容
with open(gym_checker_path, 'r') as f:
    content = f.read()

# 替换 np.bool8 为 bool
modified_content = content.replace('np.bool8', 'bool')

# 写回文件
with open(gym_checker_path, 'w') as f:
    f.write(modified_content)

print(f"成功修复 {gym_checker_path}")
print("现在可以运行 ARS 算法了!") 