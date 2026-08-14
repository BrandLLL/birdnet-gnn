# birdnet_gnn/utils/graph_utils.py
"""
图结构构造相关工具。
包含：
- haversine_distance(): 计算两点之间的大圆距离（haversine 公式）。
- build_edge_index(): 根据站点坐标和距离阈值构造 PyG 所需的 edge_index。
"""

import numpy as np
import torch


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    使用 haversine 公式计算地球表面两点之间的大圆距离（单位：千米）。

    参数:
        lat1, lon1: 第一个点的纬度、经度（单位：度）。
        lat2, lon2: 第二个点的纬度、经度（单位：度）。
                    这些参数可以是标量，也可以是可广播的 NumPy 数组。

    返回:
        两点之间的距离（千米），类型与输入广播后一致。
    """
    R = 6371.0  # 地球平均半径，单位：千米

    # 将角度转换为弧度
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # haversine 公式
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))

    return R * c  # 距离 = 半径 * 弧度角


def build_edge_index(station_coords, threshold_km):
    """
    根据站点坐标和距离阈值构造无向图的 edge_index。

    规则（对应论文公式 3）:
        如果两个站点 i, j 之间的 haversine 距离 < threshold_km，则建立一条边。
        图为无向图，因此每条边会同时加入 (i, j) 和 (j, i) 两个方向。

    参数:
        station_coords: 形状为 [N, 2] 的数组，每行是 (latitude, longitude)。
        threshold_km:   距离阈值（千米）。

    返回:
        edge_index: 形状为 [2, E] 的 LongTensor，符合 PyG 的输入格式。
                    第一行为源节点索引，第二行为目标节点索引。
    """
    coords = np.asarray(station_coords)   # [N, 2]
    num_nodes = coords.shape[0]           # N

    src_list = []  # 源节点索引列表
    dst_list = []  # 目标节点索引列表

    # 遍历所有节点对 (i, j)，i < j，避免重复计算
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = haversine_distance(
                coords[i, 0], coords[i, 1],
                coords[j, 0], coords[j, 1],
            )
            # 距离小于阈值则连边（无向，两个方向都加）
            if dist < threshold_km:
                src_list.append(i)
                dst_list.append(j)
                src_list.append(j)
                dst_list.append(i)

    # 如果没有任何边，至少为每个节点添加自环，避免 GCN 出错
    if len(src_list) == 0:
        src_list = list(range(num_nodes))
        dst_list = list(range(num_nodes))

    # 转为 [2, E] 的 LongTensor
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)  # shape: [2, E]
    return edge_index
