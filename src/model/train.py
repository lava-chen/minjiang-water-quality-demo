"""
train.py
LSTM+Attention 模型训练脚本。

对 7 个重点站逐一训练，每站独立模型。
使用 Early Stopping 防止过拟合（patience=20）。
训练完成后保存模型权重、scaler、训练日志、测试集预测值。

【改进清单】
  A: 数据切分修复（在 dataset.py 中实现）
  B: 滚动趋势特征（在 dataset.py 中实现）
  C: hidden_size 64→96，weight_decay=1e-5，max_epochs=200
  D: MSELoss → SmoothL1Loss（Huber，对异常值更稳健）
  E: 保存 test_predictions.csv 供前端画图
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# 将项目根目录加入 path（优先用环境变量，兼容中文路径）
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

# ====== 训练超参数 ======
HIDDEN_SIZE = 96           # 【改进C】64→96
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5        # 【改进C】L2正则化，防止更大模型过拟合
MAX_EPOCHS = 200           # 【改进C】150→200，给更大模型更多学习时间
PATIENCE = 20              # early stopping 耐心值
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_CSV = PROJECT_ROOT / "data" / "processed" / "minjiang_4h.csv"
OUTPUT_DIR = PROJECT_ROOT / "models"


def train_one_station(station_name: str) -> dict:
    """训练单个站点模型，返回训练结果字典"""
    print(f"\n{'='*60}")
    print(f"  站点: {station_name}")
    print(f"{'='*60}")

    # 准备数据（现在返回7个值，多了 test_target_times）
    train_loader, val_loader, test_loader, scaler, target_idx, split_info, test_target_times = \
        prepare_data_for_station(str(DATA_CSV), station_name)

    print(f"  数据总量: {split_info['total_rows']} 行")
    print(f"  训练集: {split_info['train_sequences']} 个序列 ({split_info['train_period']})")
    print(f"  验证集: {split_info['val_sequences']} 个序列 ({split_info['val_period']})")
    print(f"  测试集: {split_info['test_sequences']} 个序列 ({split_info['test_period']})")

    # 建模
    model = WaterQualityLSTM(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        forecast_horizon=FORECAST,
    ).to(DEVICE)

    # 【改进D】SmoothL1Loss = Huber Loss，对异常值更稳健
    criterion = nn.SmoothL1Loss()

    # 【改进C】加入 weight_decay 做 L2 正则化
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    # ====== 训练循环 ======
    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    best_epoch = 0

    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        # --- 训练 ---
        model.train()
        batch_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)

        # --- 验证 ---
        model.eval()
        batch_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                batch_losses.append(loss.item())
        val_loss = np.mean(batch_losses)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # --- Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epoch % 20 == 0 or epoch == 1 or epochs_no_improve == 0:
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
                  f"{' *best' if epochs_no_improve == 0 else ''}")

        if epochs_no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}, best epoch = {best_epoch}")
            break

    elapsed = time.time() - t0
    print(f"  训练耗时: {elapsed:.1f} 秒 ({epoch} 个 epoch)")

    # 恢复最佳模型
    model.load_state_dict(best_state)

    # ====== 测试集评估 ======
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            pred = model(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(y_batch.numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)

    # 反标准化：只恢复 DO 列（target_idx）
    do_mean = scaler.mean_[target_idx]
    do_std = scaler.scale_[target_idx]
    preds_real = all_preds * do_std + do_mean
    trues_real = all_trues * do_std + do_mean

    # 计算评估指标（对每个预测步）
    step_metrics = []
    for step in range(FORECAST):
        p = preds_real[:, step]
        t = trues_real[:, step]
        mae = np.mean(np.abs(p - t))
        rmse = np.sqrt(np.mean((p - t) ** 2))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        mape = np.mean(np.abs((t - p) / (t + 1e-8))) * 100
        step_metrics.append({
            "step": step + 1,
            "hours_ahead": (step + 1) * 4,
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "R2": round(float(r2), 4),
            "MAPE": round(float(mape), 2),
        })

    # 总体指标（所有步的平均）
    overall_mae = np.mean([m["MAE"] for m in step_metrics])
    overall_rmse = np.mean([m["RMSE"] for m in step_metrics])
    overall_r2 = np.mean([m["R2"] for m in step_metrics])
    overall_mape = np.mean([m["MAPE"] for m in step_metrics])

    print(f"\n  测试集评估（反标准化后的真实 DO 值）:")
    print(f"  {'步':>4s} {'提前':>6s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s} {'MAPE%':>8s}")
    for m in step_metrics:
        print(f"  {m['step']:>4d} {m['hours_ahead']:>4d}h {m['MAE']:>8.4f} "
              f"{m['RMSE']:>8.4f} {m['R2']:>8.4f} {m['MAPE']:>7.2f}%")
    print(f"  {'平均':>4s} {'--':>6s} {overall_mae:>8.4f} "
          f"{overall_rmse:>8.4f} {overall_r2:>8.4f} {overall_mape:>7.2f}%")

    # ====== 保存 ======
    station_dir = OUTPUT_DIR / station_name
    station_dir.mkdir(parents=True, exist_ok=True)

    # 保存模型权重
    torch.save(model.state_dict(), station_dir / "model.pt")

    # 保存 scaler
    import joblib
    joblib.dump(scaler, station_dir / "scaler.pkl")

    # 保存训练结果
    result = {
        "station": station_name,
        "split_info": split_info,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "train_time_sec": round(elapsed, 1),
        "best_val_loss": round(float(best_val_loss), 6),
        "step_metrics": step_metrics,
        "overall": {
            "MAE": round(float(overall_mae), 4),
            "RMSE": round(float(overall_rmse), 4),
            "R2": round(float(overall_r2), 4),
            "MAPE": round(float(overall_mape), 2),
        },
        "hyperparams": {
            "lookback": LOOKBACK,
            "forecast": FORECAST,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "loss_function": "SmoothL1Loss",
            "input_features": FEATURE_COLS,
        },
    }

    with open(station_dir / "train_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 保存 loss 曲线数据
    np.savez(
        station_dir / "loss_curves.npz",
        train_losses=train_losses,
        val_losses=val_losses,
    )

    # 【改进E】保存测试集预测值 CSV，供前端画预测曲线
    pred_rows = []
    for seq_idx in range(len(preds_real)):
        row = {"时间": pd.Timestamp(test_target_times[seq_idx]).strftime("%Y-%m-%d %H:%M")}
        for step in range(FORECAST):
            h = (step + 1) * 4
            row[f"实际DO_{h}h"] = round(float(trues_real[seq_idx, step]), 4)
            row[f"预测DO_{h}h"] = round(float(preds_real[seq_idx, step]), 4)
        pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(station_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"  测试集预测值已保存: {station_dir / 'test_predictions.csv'} ({len(pred_df)} 行)")

    return result


def main():
    print(f"设备: {DEVICE}")
    print(f"模型参数: lookback={LOOKBACK}(3天), forecast={FORECAST}(24h)")
    print(f"输入特征: {len(FEATURE_COLS)} 个 = 9水质 + 4时间 + 6滚动趋势")
    print(f"隐藏层: {HIDDEN_SIZE}, 损失函数: SmoothL1Loss(Huber)")
    print(f"数据文件: {DATA_CSV}")

    stations = get_key_stations(str(DATA_CSV))
    print(f"\n重点站点 ({len(stations)} 个): {stations}")

    all_results = []
    for station in stations:
        try:
            result = train_one_station(station)
            all_results.append(result)
        except Exception as e:
            print(f"\n  [错误] 站点 {station} 训练失败: {e}")
            import traceback
            traceback.print_exc()

    # 输出汇总表
    print(f"\n\n{'='*80}")
    print("  全部站点训练结果汇总")
    print(f"{'='*80}")
    print(f"  {'站点':<12s} {'Best Epoch':>10s} {'耗时(秒)':>8s} "
          f"{'MAE':>8s} {'RMSE':>8s} {'R2':>8s} {'MAPE%':>8s}")
    print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in all_results:
        o = r["overall"]
        print(f"  {r['station']:<12s} {r['best_epoch']:>10d} {r['train_time_sec']:>8.1f} "
              f"{o['MAE']:>8.4f} {o['RMSE']:>8.4f} {o['R2']:>8.4f} {o['MAPE']:>7.2f}%")

    # 保存全局汇总
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n全部模型已保存到: {OUTPUT_DIR}")
    print("训练完成！")


if __name__ == "__main__":
    main()
