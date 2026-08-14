# birdnet_gnn/datasets/radar_graph_dataset.py
"""
雷达站点图数据集。

将时空数据组织为一系列 PyG 的 Data 对象：
每一个时间步 t 对应一张图（Data），所有时间步共享同一个 edge_index。

每张图 Data 包含:
- x:          节点特征矩阵, 形状 [N, 3]
- edge_index: 边索引,       形状 [2, E]（所有时间步共享）
- y:          预测目标,     形状 [N, 1]
"""

import torch
from torch_geometric.data import Data, Dataset


class RadarGraphDataset(Dataset):
    """
    自定义 PyG 数据集：每个样本是某一时间步的雷达站点图。
    """

    def __init__(self, features, targets, edge_index):
        """
        参数:
            features:   FloatTensor, 形状 [T, N, 3]，所有时间步的节点特征。
            targets:    FloatTensor, 形状 [T, N, 1]，所有时间步的预测目标。
            edge_index: LongTensor,  形状 [2, E]，共享的边索引。
        """
        super().__init__()
        # 统一转换为张量，保证类型正确
        self.features = torch.as_tensor(features, dtype=torch.float)   # [T, N, 3]
        self.targets = torch.as_tensor(targets, dtype=torch.float)     # [T, N, 1]
        self.edge_index = torch.as_tensor(edge_index, dtype=torch.long)  # [2, E]

        self.num_timesteps = self.features.shape[0]  # T

    def len(self):
        """返回数据集中图的数量，即时间步数量 T。"""
        return self.num_timesteps

    def get(self, idx):
        """
        构造第 idx 个时间步对应的图 Data 对象。

        参数:
            idx: 时间步索引。

        返回:
            一个 PyG Data 对象，包含 x [N,3]、edge_index [2,E]、y [N,1]。
        """
        x = self.features[idx]   # [N, 3] 当前时间步所有站点的特征
        y = self.targets[idx]    # [N, 1] 当前时间步的预测目标

        data = Data(x=x, edge_index=self.edge_index, y=y)
        return data
