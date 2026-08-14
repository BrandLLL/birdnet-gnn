# birdnet_gnn/models/gcn_model.py
"""
GCN 迁徙强度预测模型。

对应论文公式 (10)(11)(12):
    H1 = ReLU(A_hat · X · W0)        # 第一层 GCN
    H2 = A_hat · H1 · W1             # 第二层 GCN
    M_next = f(H2)                   # 线性预测层输出下一时间步迁徙强度

结构:
    GCNConv(3, 64) -> ReLU -> Dropout -> GCNConv(64, 64) -> ReLU -> Linear(64, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNMigrationPredictor(nn.Module):
    """
    两层 GCN + 线性输出层，用于预测每个站点下一时间步的迁徙强度。
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        """
        参数:
            input_dim:  输入特征维度（=3: 迁徙强度、风速、温度）。
            hidden_dim: 隐藏层维度（=64）。
            output_dim: 输出维度（=1: 下一时间步迁徙强度）。
            dropout:    Dropout 概率。
        """
        super().__init__()

        # 第一层图卷积: 3 -> 64
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # 第二层图卷积: 64 -> 64
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # 线性预测层: 64 -> 1
        self.linear = nn.Linear(hidden_dim, output_dim)

        self.dropout = dropout

    def forward(self, data):
        """
        前向传播。

        参数:
            data: PyG Data 或 Batch 对象，包含:
                  - x:          节点特征, 形状 [num_nodes, 3]
                  - edge_index: 边索引,   形状 [2, E]
                  （在 batch 中, num_nodes = N * batch_size）

        返回:
            pred: 预测结果, 形状 [num_nodes, 1]
        """
        x = data.x                    # [num_nodes, 3]
        edge_index = data.edge_index  # [2, E]

        # 第一层 GCN + ReLU
        x = self.conv1(x, edge_index)        # [num_nodes, 64]
        x = F.relu(x)                        # [num_nodes, 64]
        # Dropout（仅在训练时生效）
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第二层 GCN + ReLU
        x = self.conv2(x, edge_index)        # [num_nodes, 64]
        x = F.relu(x)                        # [num_nodes, 64]

        # 线性层输出预测
        pred = self.linear(x)                # [num_nodes, 1]
        return pred
