# birdnet_gnn/main.py
"""
完整训练流程入口。

流程:
1. 设置随机种子
2. 生成 mock data
3. 构建 edge_index（共享图结构）
4. 构建 dataset
5. 按时间顺序划分 train / val / test（不打乱）
6. 创建 DataLoader
7. 创建模型
8. 训练模型，每轮输出 train loss、val MAE、val RMSE
9. 最后输出 test MAE、test RMSE

运行方式:
    python main.py
"""

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from config import Config
from utils.seed import set_seed
from utils.graph_utils import build_edge_index
from data.mock_data import generate_mock_data
from datasets.radar_graph_dataset import RadarGraphDataset
from models.gcn_model import GCNMigrationPredictor
from train.trainer import train_one_epoch
from train.evaluator import evaluate


def main():
    cfg = Config()

    # ---------------- 1. 设置随机种子 ----------------
    set_seed(cfg.seed)

    # 选择运行设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ---------------- 2. 生成 mock data ----------------
    # station_coords: [N, 2], features: [T, N, 3], targets: [T, N, 1]
    station_coords, features, targets = generate_mock_data(
        num_stations=cfg.num_stations,
        num_timesteps=cfg.num_timesteps,
        seed=cfg.seed,
    )
    print(f"station_coords shape: {station_coords.shape}")  # [N, 2]
    print(f"features shape:       {features.shape}")        # [T, N, 3]
    print(f"targets shape:        {targets.shape}")         # [T, N, 1]

    # ---------------- 3. 构建 edge_index ----------------
    # edge_index: [2, E]，所有时间步共享
    edge_index = build_edge_index(station_coords, cfg.threshold_km)
    print(f"edge_index shape:     {tuple(edge_index.shape)}")  # [2, E]

    # ---------------- 4. 构建 dataset ----------------
    dataset = RadarGraphDataset(features, targets, edge_index)
    T = len(dataset)  # 时间步总数

    # ---------------- 5. 按时间顺序划分 train / val / test ----------------
    # 重要: 不打乱时间序列，严格按时间先后切分。
    train_end = int(T * cfg.train_ratio)                       # 训练集结束索引
    val_end = int(T * (cfg.train_ratio + cfg.val_ratio))       # 验证集结束索引

    # 注意: targets 的最后一个时间步是占位值（没有真实的 t+1），
    # 它落在测试集尾部，对整体指标影响很小；如需严格，可在此处排除最后一步。
    train_indices = list(range(0, train_end))                  # [0, train_end)
    val_indices = list(range(train_end, val_end))              # [train_end, val_end)
    test_indices = list(range(val_end, T))                     # [val_end, T)

    # 使用索引切片得到子数据集（PyG Dataset 支持用索引列表切片）
    train_set = dataset[train_indices]
    val_set = dataset[val_indices]
    test_set = dataset[test_indices]

    print(f"训练集时间步数: {len(train_set)}, "
          f"验证集时间步数: {len(val_set)}, "
          f"测试集时间步数: {len(test_set)}")

    # ---------------- 6. 创建 DataLoader ----------------
    # 训练集同样不打乱（shuffle=False），保持时间顺序。
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)

    # ---------------- 7. 创建模型 ----------------
    model = GCNMigrationPredictor(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        dropout=cfg.dropout,
    ).to(device)

    # 优化器与损失函数
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.MSELoss()  # 损失函数使用均方误差

    # ---------------- 8. 训练模型 ----------------
    print("\n开始训练...")
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_mae, val_rmse = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:03d} | "
              f"train loss: {train_loss:.4f} | "
              f"val MAE: {val_mae:.4f} | "
              f"val RMSE: {val_rmse:.4f}")

    # ---------------- 9. 测试集评估 ----------------
    test_mae, test_rmse = evaluate(model, test_loader, device)
    print("\n========== 测试集结果 ==========")
    print(f"test MAE:  {test_mae:.4f}")
    print(f"test RMSE: {test_rmse:.4f}")


if __name__ == "__main__":
    main()
