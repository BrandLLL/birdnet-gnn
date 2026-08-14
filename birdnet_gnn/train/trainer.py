# birdnet_gnn/train/trainer.py
"""
训练逻辑。
包含 train_one_epoch()，对训练集执行一个完整 epoch 的训练。
"""

import torch


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    训练一个 epoch。

    参数:
        model:     GCN 模型。
        loader:    训练集 DataLoader，每个 batch 是若干时间步图组成的 Batch。
        optimizer: 优化器（如 Adam）。
        criterion: 损失函数（MSELoss）。
        device:    'cpu' 或 'cuda'。

    返回:
        avg_loss: 该 epoch 的平均训练损失（Python float）。
    """
    model.train()  # 切换到训练模式（启用 Dropout）

    total_loss = 0.0   # 累计损失（按样本图数量加权）
    total_graphs = 0   # 累计图数量

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()           # 清空梯度

        pred = model(batch)             # 前向传播, pred 形状 [num_nodes_in_batch, 1]
        target = batch.y                # 真实值, 形状 [num_nodes_in_batch, 1]

        loss = criterion(pred, target)  # MSE 损失（标量）
        loss.backward()                 # 反向传播
        optimizer.step()                # 更新参数

        # batch.num_graphs 表示该 batch 中图（时间步）的数量
        num_graphs = batch.num_graphs
        total_loss += loss.item() * num_graphs
        total_graphs += num_graphs

    avg_loss = total_loss / max(total_graphs, 1)
    return avg_loss
