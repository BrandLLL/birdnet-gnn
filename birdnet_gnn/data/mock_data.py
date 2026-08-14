# birdnet_gnn/data/mock_data.py
"""
生成可直接跑通整个项目的 mock（模拟）数据。

生成内容:
- station_coords: 形状 [N, 2]，每个站点的 (latitude, longitude)。
- features:       形状 [T, N, 3]，每个时间步、每个站点的 3 个特征：
                  [migration_intensity, wind_speed, temperature]。
- targets:        形状 [T, N, 1]，每个时间步、每个站点的预测目标：
                  下一个时间步的 migration_intensity。

说明:
本模拟数据在美国本土经纬度范围内随机生成站点位置，
并用带时间趋势 + 噪声的方式生成迁徙强度等特征，
使得相邻时间步之间存在一定的时间相关性，便于模型学习。
"""

import numpy as np


def generate_mock_data(num_stations, num_timesteps, seed=42):
    """
    生成模拟雷达站点时空数据。

    参数:
        num_stations:  站点数量 N。
        num_timesteps: 时间步数量 T。
        seed:          随机种子，保证数据可复现。

    返回:
        station_coords: np.ndarray, 形状 [N, 2]
        features:       np.ndarray, 形状 [T, N, 3]
        targets:        np.ndarray, 形状 [T, N, 1]
    """
    rng = np.random.default_rng(seed)  # 独立随机数生成器，避免影响全局种子

    N = num_stations
    T = num_timesteps

    # ---------------- 1. 生成站点坐标 ----------------
    # 纬度范围大致取美国本土 (25N ~ 49N)，经度范围 (-124 ~ -67)
    lats = rng.uniform(25.0, 49.0, size=N)    # [N]
    lons = rng.uniform(-124.0, -67.0, size=N)  # [N]
    station_coords = np.stack([lats, lons], axis=1)  # [N, 2]

    # ---------------- 2. 生成迁徙强度时间序列 ----------------
    # 思路: 每个站点有一个基础迁徙水平 base，叠加随时间变化的正弦季节信号 + 随机噪声。
    base_intensity = rng.uniform(0.2, 0.8, size=N)        # [N] 每站点基础强度
    phase = rng.uniform(0, 2 * np.pi, size=N)             # [N] 每站点不同相位

    # migration_intensity: [T, N]
    migration_intensity = np.zeros((T, N), dtype=np.float32)
    for t in range(T):
        # 季节性信号: 随时间步缓慢变化的正弦波
        seasonal = 0.3 * np.sin(2 * np.pi * t / 50.0 + phase)  # [N]
        noise = rng.normal(0.0, 0.05, size=N)                  # [N] 小噪声
        migration_intensity[t] = np.clip(base_intensity + seasonal + noise, 0.0, None)

    # ---------------- 3. 生成气象特征 ----------------
    # wind_speed: [T, N]，风速（带噪声的随机过程）
    wind_speed = rng.uniform(0.0, 15.0, size=(T, N)).astype(np.float32)  # [T, N]

    # temperature: [T, N]，温度（带日/季节变化）
    temperature = np.zeros((T, N), dtype=np.float32)
    for t in range(T):
        temp_seasonal = 15.0 + 10.0 * np.sin(2 * np.pi * t / 100.0)  # 标量季节温度
        temperature[t] = temp_seasonal + rng.normal(0.0, 2.0, size=N)

    # ---------------- 4. 组装 features ----------------
    # features: [T, N, 3]，最后一维顺序为 [migration_intensity, wind_speed, temperature]
    features = np.stack([migration_intensity, wind_speed, temperature], axis=2)  # [T, N, 3]
    features = features.astype(np.float32)

    # ---------------- 5. 构造 targets ----------------
    # 预测目标: 下一个时间步的 migration_intensity。
    # targets[t] = migration_intensity[t+1]
    # 最后一个时间步没有 t+1，用当前时间步的值填充（占位，划分数据时会自然落入测试集尾部）。
    targets = np.zeros((T, N, 1), dtype=np.float32)  # [T, N, 1]
    targets[:-1, :, 0] = migration_intensity[1:]     # 前 T-1 步指向下一步
    targets[-1, :, 0] = migration_intensity[-1]      # 最后一步占位

    return station_coords, features, targets
