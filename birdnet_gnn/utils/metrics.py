# birdnet_gnn/utils/metrics.py
"""
评价指标工具。
包含:
- calculate_mae():  平均绝对误差 (Mean Absolute Error)
- calculate_rmse(): 均方根误差 (Root Mean Squared Error)
"""

import torch


def calculate_mae(pred, target):
    """
    计算平均绝对误差 MAE。

    参数:
        pred:   预测值张量，形状任意（通常为 [N, 1] 或 [total_nodes, 1]）。
        target: 真实值张量，形状与 pred 相同。

    返回:
        一个 Python float，表示 MAE。
    """
    # torch.abs 逐元素取绝对值，再求平均
    return torch.mean(torch.abs(pred - target)).item()


def calculate_rmse(pred, target):
    """
    计算均方根误差 RMSE。

    参数:
        pred:   预测值张量，形状任意。
        target: 真实值张量，形状与 pred 相同。

    返回:
        一个 Python float，表示 RMSE。
    """
    # 先求均方误差 MSE，再开平方根
    mse = torch.mean((pred - target) ** 2)
    return torch.sqrt(mse).item()
