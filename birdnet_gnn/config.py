# birdnet_gnn/config.py
"""
全局配置文件。
集中保存所有超参数，方便统一修改和管理。
"""


class Config:
    # ---------------- 随机种子 ----------------
    seed = 42                # 随机种子，保证实验可复现

    # ---------------- 数据相关 ----------------
    num_stations = 30        # 雷达站点数量 N
    num_timesteps = 200      # 时间步数量 T
    input_dim = 3            # 每个节点的输入特征维度：[migration_intensity, wind_speed, temperature]
    output_dim = 1           # 预测输出维度：下一个时间步的 migration_intensity

    # ---------------- 图结构相关 ----------------
    threshold_km = 500.0     # 建立边的距离阈值（单位：千米），两站点距离小于该值则连边

    # ---------------- 模型相关 ----------------
    hidden_dim = 64          # GCN 隐藏层维度
    dropout = 0.3            # Dropout 概率

    # ---------------- 训练相关 ----------------
    learning_rate = 0.01     # 学习率
    weight_decay = 5e-4      # 权重衰减（L2 正则）
    epochs = 100             # 训练轮数
    batch_size = 16          # 每个 batch 包含的图（时间步）数量

    # ---------------- 数据集划分比例 ----------------
    # 按时间顺序划分，不打乱
    train_ratio = 0.7        # 训练集占比
    val_ratio = 0.15         # 验证集占比
    # 剩余部分为测试集
