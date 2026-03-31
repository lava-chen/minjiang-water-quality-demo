"""
feature_importance.py
基于置换重要性（Permutation Importance）的特征归因分析。

原理：依次打乱某个特征的值，观察预测误差上升了多少。
误差上升越多，说明模型越依赖该特征。

已更新：
- 适配 prepare_data_for_station 新的 7 返回值接口
- 从 train_result.json 读取超参数，不再硬编码 hidden_size
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.lstm_model import WaterQualityLSTM
from src.model.dataset import (
    prepare_data_for_station,
    get_key_stations,
    FEATURE_COLS,
    FORECAST,
    LOOKBACK,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_CSV = PROJECT_ROOT / "data" / "processed" / "minjiang_4h.csv"
MODEL_DIR = PROJECT_ROOT / "models"
N_REPEATS = 5  # 每个特征重复打乱次数，取平均


def compute_permutation_importance(model, test_loader, n_features, device="cpu"):
    """
    计算置换重要性。

    返回: (n_features,) 的数组，值越大越重要。
    """
    model.eval()

    # 基准 MSE
    base_mse = _evaluate_mse(model, test_loader, device)

    importances = np.zeros(n_features)

    for feat_idx in range(n_features):
        mse_increases = []
        for _ in range(N_REPEATS):
            shuffled_mse = _evaluate_mse_shuffled(model, test_loader, feat_idx, device)
            mse_increases.append(shuffled_mse - base_mse)
        importances[feat_idx] = np.mean(mse_increases)

    # 归一化到 0~1 区间（相对重要性）
    total = importances.sum()
    if total > 0:
        importances = importances / total

    return importances, base_mse


def _evaluate_mse(model, loader, device):
    """计算测试集 MSE"""
    total_loss = 0
    total_count = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += ((pred - y) ** 2).sum().item()
            total_count += y.numel()
    return total_loss / total_count


def _evaluate_mse_shuffled(model, loader, feat_idx, device):
    """打乱第 feat_idx 个特征后的 MSE"""
    total_loss = 0
    total_count = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.clone()
            # 在 batch 维度内随机打乱该特征
            perm = torch.randperm(X.size(0))
            X[:, :, feat_idx] = X[perm, :, feat_idx]
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += ((pred - y) ** 2).sum().item()
            total_count += y.numel()
    return total_loss / total_count


def compute_attention_importance(model, test_loader, device="cpu"):
    """
    从注意力权重提取时间步重要性。
    返回: (LOOKBACK,) 的平均注意力权重。
    """
    model.eval()
    all_attn = []
    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(device)
            _, attn_weights = model(X, return_attention=True)
            all_attn.append(attn_weights.cpu().numpy())
    all_attn = np.concatenate(all_attn, axis=0)  # (N, LOOKBACK)
    mean_attn = all_attn.mean(axis=0)  # (LOOKBACK,)
    return mean_attn


def analyze_all_stations():
    """对所有重点站进行特征重要性分析"""
    stations = get_key_stations(str(DATA_CSV))
    print(f"分析 {len(stations)} 个重点站的特征重要性...\n")

    all_importance = {}

    for station in stations:
        print(f"--- {station} ---")
        station_dir = MODEL_DIR / station

        if not (station_dir / "model.pt").exists():
            print(f"  [跳过] 未找到模型文件")
            continue

        # 加载数据（适配新的 7 返回值接口）
        _, _, test_loader, scaler, target_idx, _, _ = \
            prepare_data_for_station(str(DATA_CSV), station)

        # 从 train_result.json 读取超参数（不再硬编码）
        with open(station_dir / "train_result.json", "r", encoding="utf-8") as f:
            train_result = json.load(f)
        hp = train_result["hyperparams"]

        # 加载模型
        model = WaterQualityLSTM(
            input_size=len(FEATURE_COLS),
            hidden_size=hp["hidden_size"],
            num_layers=hp["num_layers"],
            dropout=0.0,           # 推理时不使用 dropout
            forecast_horizon=FORECAST,
        ).to(DEVICE)
        model.load_state_dict(torch.load(station_dir / "model.pt", map_location=DEVICE))

        # 置换重要性
        feat_imp, base_mse = compute_permutation_importance(
            model, test_loader, len(FEATURE_COLS), DEVICE
        )

        # 注意力权重
        attn_imp = compute_attention_importance(model, test_loader, DEVICE)

        # 整理结果
        feat_ranking = sorted(
            zip(FEATURE_COLS, feat_imp.tolist()),
            key=lambda x: x[1], reverse=True
        )

        result = {
            "station": station,
            "base_mse": round(float(base_mse), 6),
            "feature_importance": {name: round(val, 4) for name, val in feat_ranking},
            "attention_weights": {
                f"t-{LOOKBACK - i}({(LOOKBACK - i)*4}h前)": round(float(v), 4)
                for i, v in enumerate(attn_imp)
            },
        }

        print(f"  基准 MSE: {base_mse:.6f}")
        print(f"  Top-5 重要特征:")
        for name, val in feat_ranking[:5]:
            print(f"    {name}: {val:.4f}")
        print(f"  注意力集中度: 最近4h权重={attn_imp[-1]:.3f}, 最远72h权重={attn_imp[0]:.3f}")

        all_importance[station] = result

        # 保存单站结果
        with open(station_dir / "feature_importance.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # 保存全局汇总
    with open(MODEL_DIR / "all_feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(all_importance, f, ensure_ascii=False, indent=2)

    print(f"\n特征重要性分析完成，结果已保存到 {MODEL_DIR}")


if __name__ == "__main__":
    analyze_all_stations()
