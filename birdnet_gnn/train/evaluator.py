# birdnet_gnn/train/evaluator.py
"""
评估逻辑。
包含 evaluate()，在验证集或测试集上计算 MAE 和 RMSE。
"""

import torch

from utils.metrics import calculate_mae, calculate_rmse


@torch.no_grad()  # 评估阶段不计算梯度，节省显存与时间
def evaluate(model, loader, device):
    """
    在给定数据集上评估模型，返回 MAE 和 RMSE。

    参数:
        model:  GCN 模型。
        loader: 验证集或测试集的 DataLoader。
        device: 'cpu' 或 'cuda'。

    返回:
        mae:  平均绝对误差 (float)
        rmse: 均方根误差   (float)
    """
    model.eval()  # 切换到评估模式（关闭 Dropout）

    all_preds = []    # 收集所有预测值
    all_targets = []  # 收集所有真实值

    for batch in loader:
        batch = batch.to(device)

        pred = model(batch)   # [num_nodes_in_batch, 1]
        target = batch.y      # [num_nodes_in_batch, 1]

        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

    # 将所有 batch 的结果拼接成一个大张量
    preds = torch.cat(all_preds, dim=0)      # [total_nodes, 1]
    targets = torch.cat(all_targets, dim=0)  # [total_nodes, 1]

    mae = calculate_mae(preds, targets)
    rmse = calculate_rmse(preds, targets)
    return mae, rmse
