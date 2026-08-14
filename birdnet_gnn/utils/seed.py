# birdnet_gnn/utils/seed.py
"""
随机种子设置工具。
保证实验在不同次运行之间结果可复现。
"""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    设置 Python、NumPy 和 PyTorch 的随机种子。

    参数:
        seed: 随机种子整数值。
    """
    random.seed(seed)                       # Python 内置随机模块
    np.random.seed(seed)                    # NumPy 随机模块
    torch.manual_seed(seed)                 # PyTorch CPU 随机种子
    torch.cuda.manual_seed_all(seed)        # PyTorch 所有 GPU 随机种子（如果有 GPU）

    # 让 cuDNN 使用确定性算法，进一步保证可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
