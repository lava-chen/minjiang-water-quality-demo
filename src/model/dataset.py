"""
dataset.py
水质预测数据集构建。

关键设计：
1. StandardScaler 只在训练集上 fit（防止数据泄漏）
2. 按时间顺序切分 train/val/test = 70%/15%/15%
3. 【改进A】先在完整数据上创建滑动窗口，再按目标时间点切分
   → 验证/测试集序列保留训练集尾部的历史上下文
4. 使用 4h 间隔数据，lookback=18步(3天)，forecast=6步(24h)
5. 【改进B】输入 19 个特征 = 9水质 + 4时间 + 6滚动趋势
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

# ====== 核心参数 ======
LOOKBACK = 18       # 回看窗口：18 × 4h = 3天
FORECAST = 6        # 预测步数：6 × 4h = 24小时
BATCH_SIZE = 64
TARGET_COL = "溶解氧"

# 9个水质参数
QUALITY_COLS = [
    "水温", "pH", "溶解氧", "电导率", "浊度",
    "高锰酸盐指数", "氨氮", "总磷", "总氮",
]

# 4个周期时间特征（在 prepare_minjiang.py 中已生成）
TIME_COLS = ["hour_sin", "hour_cos", "dayofyear_sin", "dayofyear_cos"]

# 【改进B】6个滚动趋势特征（在 add_rolling_features 中生成）
ROLLING_COLS = [
    "DO_24h_mean",    # 过去24h 溶解氧均值
    "DO_24h_std",     # 过去24h 溶解氧波动幅度
    "DO_diff",        # 溶解氧与上一时刻的差值（变化速率）
    "DO_72h_mean",    # 过去72h 溶解氧均值（长期趋势）
    "temp_24h_mean",  # 过去24h 水温均值（DO最大驱动因素）
    "temp_diff",      # 水温变化速率
]

# 全部输入特征
FEATURE_COLS = QUALITY_COLS + TIME_COLS + ROLLING_COLS   # 19列


class WaterQualityDataset(Dataset):
    """滑动窗口时序数据集"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_station_data(csv_path: str, station_name: str) -> pd.DataFrame:
    """加载单个站点的数据，按时间排序，并剔除传感器故障产生的异常值"""
    df = pd.read_csv(csv_path)
    df = df[df["站点名称"] == station_name].copy()
    df["监测时间"] = pd.to_datetime(df["监测时间"])
    df = df.sort_values("监测时间").reset_index(drop=True)

    # ====== 异常值过滤 ======
    # 河流 DO 的物理合理范围约 1~16 mg/L
    # DO > 16 几乎必定是传感器饱和/故障（如都江堰的 20.0）
    n_before = df["溶解氧"].notna().sum()
    
    if station_name == "姜公堰":
        # 姜公堰“一站一策”：可能有真实低氧情况，取消对 DO < 1 的硬性过滤
        # 只剔除明显大于 16 的传感器故障值
        df.loc[df["溶解氧"] > 16, "溶解氧"] = np.nan
    else:
        # 其他站点维持原有的通用过滤规则
        df.loc[df["溶解氧"] > 16, "溶解氧"] = np.nan
        df.loc[df["溶解氧"] < 1,  "溶解氧"] = np.nan
        
    n_after = df["溶解氧"].notna().sum()
    n_removed = n_before - n_after
    if n_removed > 0:
        print(f"    [异常值过滤] {station_name}: 移除 {n_removed} 条 DO 异常记录")

    # 使用线性插值填补 NaN，保持时间序列的绝对连续性，避免直接删行打断 LSTM 的时间步
    df["溶解氧"] = df["溶解氧"].interpolate(method="linear", limit_direction="both")

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    【改进B】添加滚动统计特征，只使用过去数据（不会泄漏未来信息）。

    pandas .rolling() 默认向后看，min_periods=1 保证窗口不满时也能计算。
    .diff() 计算当前值与上一个值的差，第一行用 0 填充。
    """
    df = df.copy()

    # ---- DO 短期趋势（24h = 6个4h步）----
    df["DO_24h_mean"] = df["溶解氧"].rolling(6, min_periods=1).mean()
    df["DO_24h_std"]  = df["溶解氧"].rolling(6, min_periods=1).std().fillna(0)
    df["DO_diff"]     = df["溶解氧"].diff().fillna(0)

    # ---- DO 长期趋势（72h = 18个4h步）----
    df["DO_72h_mean"] = df["溶解氧"].rolling(18, min_periods=1).mean()

    # ---- 水温趋势（水温是DO最大驱动因素）----
    df["temp_24h_mean"] = df["水温"].rolling(6, min_periods=1).mean()
    df["temp_diff"]     = df["水温"].diff().fillna(0)

    return df


def create_sequences(data: np.ndarray, target_idx: int):
    """
    从二维数组创建滑动窗口序列。

    data: (N, num_features) 的数组
    target_idx: 目标列在 data 中的列索引

    返回:
        X: (num_samples, LOOKBACK, num_features)
        y: (num_samples, FORECAST)
    """
    X, y = [], []
    for i in range(len(data) - LOOKBACK - FORECAST + 1):
        X.append(data[i : i + LOOKBACK])
        y.append(data[i + LOOKBACK : i + LOOKBACK + FORECAST, target_idx])
    return np.array(X), np.array(y)


def prepare_data_for_station(
    csv_path: str,
    station_name: str,
    batch_size: int = BATCH_SIZE,
):
    """
    为单个站点准备 train/val/test DataLoader。

    【改进A】先在完整数据上创建滑动窗口序列，再按"目标起始时间"切分。
    这样验证/测试集的序列可以正确利用训练集尾部的历史数据作为上下文，
    消除了旧方案在切分边界处的"冷启动"缺陷。

    返回:
        train_loader, val_loader, test_loader,
        scaler (StandardScaler), target_idx (int),
        split_info (dict), test_target_times (ndarray of timestamps)
    """
    df = load_station_data(csv_path, station_name)

    # 【改进B】添加滚动趋势特征
    df = add_rolling_features(df)

    # 检查必要列是否存在
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少特征列: {missing}")

    # 提取特征矩阵，丢弃含 NaN 的行
    features = df[FEATURE_COLS].copy()
    valid_mask = features.notna().all(axis=1)
    features = features[valid_mask].values  # (N, 19)
    times = df.loc[valid_mask, "监测时间"].values

    # 目标列在特征列中的索引
    target_idx = FEATURE_COLS.index(TARGET_COL)

    # ====== 按时间顺序确定切分点 ======
    n = len(features)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    # ====== 标准化：只用训练集 fit ======
    scaler = StandardScaler()
    scaler.fit(features[:n_train])
    all_scaled = scaler.transform(features)

    # ====== 【改进A】在完整数据上创建序列，再切分 ======
    X_all, y_all = create_sequences(all_scaled, target_idx)

    # 对于序列 i：
    #   输入窗口: data[i : i+LOOKBACK]
    #   预测目标: data[i+LOOKBACK : i+LOOKBACK+FORECAST]
    # 按"预测目标区间"的位置来分配给 train/val/test
    seq_indices = np.arange(len(X_all))
    target_starts = seq_indices + LOOKBACK       # 目标起始索引
    target_ends   = target_starts + FORECAST     # 目标结束索引（不含）

    # 训练集：整个目标窗口都在训练期内
    train_mask = target_ends <= n_train
    # 验证集：目标起点在训练期之后，整个目标窗口在验证期内
    val_mask = (target_starts >= n_train) & (target_ends <= n_train + n_val)
    # 测试集：目标起点在验证期之后，整个目标窗口在数据范围内
    test_mask = (target_starts >= n_train + n_val) & (target_ends <= n)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val,   y_val   = X_all[val_mask],   y_all[val_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

    # 测试集序列对应的目标起始时间（供前端画图用）
    test_target_times = times[target_starts[test_mask]]

    # ====== 封装 DataLoader ======
    train_loader = DataLoader(
        WaterQualityDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        WaterQualityDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        WaterQualityDataset(X_test, y_test),
        batch_size=batch_size, shuffle=False,
    )

    # 各集的时间范围（基于原始数据行）
    train_times = times[:n_train]
    val_times_arr = times[n_train : n_train + n_val]
    test_times_arr = times[n_train + n_val :]

    split_info = {
        "total_rows": n,
        "train_rows": n_train,
        "val_rows": n_val,
        "test_rows": n - n_train - n_val,
        "train_sequences": int(train_mask.sum()),
        "val_sequences": int(val_mask.sum()),
        "test_sequences": int(test_mask.sum()),
        "train_period": f"{pd.Timestamp(train_times[0]):%Y-%m-%d} ~ {pd.Timestamp(train_times[-1]):%Y-%m-%d}",
        "val_period": f"{pd.Timestamp(val_times_arr[0]):%Y-%m-%d} ~ {pd.Timestamp(val_times_arr[-1]):%Y-%m-%d}",
        "test_period": f"{pd.Timestamp(test_times_arr[0]):%Y-%m-%d} ~ {pd.Timestamp(test_times_arr[-1]):%Y-%m-%d}",
    }

    return train_loader, val_loader, test_loader, scaler, target_idx, split_info, test_target_times


def get_key_stations(csv_path: str) -> list:
    """从站点文件获取7个重点站的名称列表"""
    stations_csv = os.path.join(os.path.dirname(csv_path), "minjiang_stations.csv")
    df = pd.read_csv(stations_csv)
    key = df[df["重点站"] == True]["站点名称"].tolist()
    return key
