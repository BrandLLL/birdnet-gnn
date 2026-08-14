# BirdNet-GNN：基于图卷积网络的鸟类迁徙强度预测

本项目是论文 *BirdNet-AI* 第二部分方法的 PyTorch Geometric 实现。

第一部分的 U-Net 已经输出了每个 radar station 在每个时间步的 **migration intensity**。
本部分将各个 radar station 构造成一张图，并使用 **两层 GCN** 预测下一时间步每个站点的 migration intensity。

## 方法对应关系（论文 Section II.D / II.E）

- **图构造**：每个 radar station 是一个 node；用 haversine（大圆）距离计算站点间距离，距离小于 `threshold_km` 则连边（论文公式 3）。所有时间步共享同一个 `edge_index`。
- **节点特征**（论文公式 4/5）：`x_i = [migration_intensity, wind_speed, temperature]`，即 `features.shape = [T, N, 3]`。
- **模型**（论文公式 10/11/12）：`GCNConv(3,64) → ReLU → Dropout → GCNConv(64,64) → ReLU → Linear(64,1)`，输出 `[N, 1]`。
- **损失**（论文公式 13）：`MSELoss`。
- **指标**：`MAE` 与 `RMSE`。

## 文件结构

```
birdnet_gnn/
├── README.md
├── requirements.txt
├── main.py                      # 完整训练流程入口
├── config.py                    # 所有超参数
├── data/
│   └── mock_data.py             # 生成可直接跑通的模拟数据
├── datasets/
│   └── radar_graph_dataset.py   # 每个时间步一张 PyG Data 图
├── models/
│   └── gcn_model.py             # 两层 GCN + 线性输出层
├── utils/
│   ├── graph_utils.py           # haversine_distance / build_edge_index
│   ├── metrics.py               # calculate_mae / calculate_rmse
│   └── seed.py                  # set_seed
└── train/
    ├── trainer.py               # train_one_epoch
    └── evaluator.py             # evaluate
```

## 张量形状速查

| 名称 | 形状 | 含义 |
|------|------|------|
| `station_coords` | `[N, 2]` | 每个站点的 (latitude, longitude) |
| `features` | `[T, N, 3]` | 每步每站点的 [迁徙强度, 风速, 温度] |
| `targets` | `[T, N, 1]` | 每步每站点的下一步迁徙强度 |
| `edge_index` | `[2, E]` | 共享的边索引 |
| 单张图 `x` | `[N, 3]` | 某时间步的节点特征 |
| 单张图 `y` | `[N, 1]` | 某时间步的预测目标 |
| 模型输出 `pred` | `[num_nodes, 1]` | batch 内所有节点的预测 |

## 安装依赖

```bash
pip install -r requirements.txt
```

> 注：`torch-geometric` 的安装可能依赖具体的 torch / CUDA 版本，
> 如遇问题请参考 PyG 官方安装文档选择对应的轮子。

## 运行

```bash
python main.py
```

程序会自动生成模拟数据、构图、按时间顺序（不打乱）划分 train/val/test，
训练并在每个 epoch 输出 `train loss / val MAE / val RMSE`，
最后输出 `test MAE / test RMSE`。
