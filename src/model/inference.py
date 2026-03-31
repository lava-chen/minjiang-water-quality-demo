"""
inference.py — 模型推理模块

加载训练好的模型和 scaler，对站点最新数据进行前向推理，
输出当前 DO 值和未来 24 小时（6 步 × 4 小时）的预测值。

用法：
  1. 被 Streamlit 页面间接调用（通过 forecasts.json）
  2. 命令行运行：python -m src.model.inference → 为所有站点生成 models/forecasts.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path


# ====== 水质分类（GB 3838 标准）======
def classify_do(do_val):
    """根据 DO 值返回 (水质等级, 状态文字, 颜色)"""
    if do_val is None or (isinstance(do_val, float) and np.isnan(do_val)):
        return "—", "无数据", "#8E8E93"
    if do_val >= 7.5:
        return "Ⅰ", "优", "#34C759"
    if do_val >= 6.0:
        return "Ⅱ", "良", "#007AFF"
    if do_val >= 5.0:
        return "Ⅲ", "一般", "#FF9500"
    if do_val >= 3.0:
        return "Ⅳ", "偏低", "#FF6B35"
    if do_val >= 2.0:
        return "Ⅴ", "差", "#FF3B30"
    return "劣Ⅴ", "极差", "#8B0000"


def get_station_forecast(station_name: str, data_csv_path: str, models_dir: str) -> dict:
    """
    对单个站点进行推理，返回当前状态和未来 24 小时预测。

    返回字典包含:
      - current_do: 当前 DO 值
      - current_time: 最新数据时间
      - current_quality/status/color: 当前水质等级
      - forecasts: 6 个预测步长的列表
      - recent_history: 最近 7 天的历史数据（用于趋势图）
    """
    import torch
    import joblib

    # 延迟导入模型和数据集模块
    from src.model.lstm_model import WaterQualityLSTM
    from src.model.dataset import (
        FEATURE_COLS, LOOKBACK, FORECAST, TARGET_COL,
        load_station_data, add_rolling_features,
    )

    station_dir = Path(models_dir) / station_name

    # ---- 1. 加载并处理站点数据 ----
    df = load_station_data(data_csv_path, station_name)
    df = add_rolling_features(df)

    # ---- 2. 提取当前值 ----
    latest = df.iloc[-1]
    current_do = float(latest["溶解氧"])
    current_time = pd.Timestamp(latest["监测时间"])
    quality, status, color = classify_do(current_do)

    # ---- 3. 最近 7 天历史（42 步 = 7 天 × 6 步/天）----
    recent_n = min(42, len(df))
    recent_df = df.tail(recent_n)[["监测时间", "溶解氧"]].copy()
    recent_history = []
    for _, r in recent_df.iterrows():
        recent_history.append({
            "time": pd.Timestamp(r["监测时间"]).strftime("%Y-%m-%d %H:%M"),
            "do": round(float(r["溶解氧"]), 2),
        })

    # ---- 4. 准备模型输入 ----
    features = df[FEATURE_COLS].copy()
    valid_mask = features.notna().all(axis=1)
    features_arr = features[valid_mask].values  # (N, 19)

    if len(features_arr) < LOOKBACK:
        return {"error": f"数据不足：需要 {LOOKBACK} 步，只有 {len(features_arr)} 步"}

    input_data = features_arr[-LOOKBACK:]  # (18, 19)

    # ---- 5. 加载 scaler 并归一化 ----
    scaler = joblib.load(station_dir / "scaler.pkl")
    input_scaled = scaler.transform(input_data)  # (18, 19)

    # ---- 6. 加载模型 ----
    with open(station_dir / "train_result.json", "r", encoding="utf-8") as f:
        train_result = json.load(f)
    hp = train_result["hyperparams"]

    model = WaterQualityLSTM(
        input_size=len(FEATURE_COLS),
        hidden_size=hp["hidden_size"],
        num_layers=hp["num_layers"],
        dropout=0.0,                # 推理时关闭 dropout
        forecast_horizon=FORECAST,
    )
    model.load_state_dict(torch.load(
        station_dir / "model.pt",
        map_location="cpu",
        weights_only=True,
    ))
    model.eval()

    # ---- 7. 前向推理 ----
    X = torch.FloatTensor(input_scaled).unsqueeze(0)  # (1, 18, 19)
    with torch.no_grad():
        pred_scaled = model(X)  # (1, 6)

    # ---- 8. 反归一化到真实 DO 值 ----
    target_idx = FEATURE_COLS.index(TARGET_COL)
    mean_do = scaler.mean_[target_idx]
    scale_do = scaler.scale_[target_idx]
    pred_real = pred_scaled.numpy().flatten() * scale_do + mean_do

    # ---- 9. 构建预测结果 ----
    forecasts = []
    for step in range(FORECAST):
        hours = (step + 1) * 4
        forecast_time = current_time + pd.Timedelta(hours=hours)
        do_val = float(pred_real[step])
        q, s, c = classify_do(do_val)

        # 与当前 DO 的变化趋势
        delta = do_val - current_do
        if delta > 0.2:
            trend = "↑"
        elif delta < -0.2:
            trend = "↓"
        else:
            trend = "→"

        forecasts.append({
            "hours_ahead": hours,
            "time": forecast_time.strftime("%m-%d %H:%M"),
            "time_full": forecast_time.strftime("%Y-%m-%d %H:%M"),
            "do": round(do_val, 2),
            "delta": round(delta, 2),
            "trend": trend,
            "quality": q,
            "status": s,
            "color": c,
        })

    return {
        "current_do": round(current_do, 2),
        "current_time": current_time.strftime("%Y-%m-%d %H:%M"),
        "current_quality": quality,
        "current_status": status,
        "current_color": color,
        "forecasts": forecasts,
        "recent_history": recent_history,
    }


# ====== 批量生成所有站点预测 ======
def generate_all_forecasts(data_csv_path: str, models_dir: str, stations: list) -> dict:
    """为所有重点站生成预测结果，保存到 models/forecasts.json"""
    results = {}
    for station in stations:
        print(f"  推理: {station} ...", end=" ")
        try:
            result = get_station_forecast(station, data_csv_path, models_dir)
            results[station] = result
            if "error" not in result:
                cur = result["current_do"]
                f4 = result["forecasts"][0]["do"]
                print(f"当前DO={cur} mg/L, 4h后预测={f4} mg/L")
            else:
                print(f"错误: {result['error']}")
        except Exception as e:
            print(f"失败: {e}")
            results[station] = {"error": str(e)}

    # 保存到 JSON
    output_path = Path(models_dir) / "forecasts.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n预测结果已保存到: {output_path}")

    return results


# ====== 命令行入口 ======
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.model.dataset import get_key_stations

    data_csv = PROJECT_ROOT / "data" / "processed" / "minjiang_4h.csv"
    models = PROJECT_ROOT / "models"

    stations = get_key_stations(str(data_csv))
    print(f"{'='*50}")
    print(f"  为 {len(stations)} 个站点生成预测")
    print(f"{'='*50}")
    generate_all_forecasts(str(data_csv), str(models), stations)
