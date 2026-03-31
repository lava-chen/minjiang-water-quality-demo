"""
evaluate.py
评估模型性能，生成 RMSE、MAE、R-squared 指标，按站点汇总。
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "results", "predictions.csv")
METRICS_PATH = os.path.join(BASE_DIR, "data", "results", "metrics.json")


def evaluate():
    df = pd.read_csv(PREDICTIONS_PATH, encoding="utf-8-sig")
    print(f"加载预测结果: {len(df)} 行, {df['站点'].nunique()} 个站点\n")

    all_metrics = {}

    for station in sorted(df["站点"].unique()):
        sdf = df[df["站点"] == station]
        actual = sdf["实际DO"].values
        predicted = sdf["预测DO"].values

        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)

        all_metrics[station] = {
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4),
            "样本数": len(sdf),
        }
        print(f"{station}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f} (n={len(sdf)})")

    # 全局指标
    actual_all = df["实际DO"].values
    predicted_all = df["预测DO"].values
    overall = {
        "RMSE": round(np.sqrt(mean_squared_error(actual_all, predicted_all)), 4),
        "MAE": round(mean_absolute_error(actual_all, predicted_all), 4),
        "R2": round(r2_score(actual_all, predicted_all), 4),
        "样本数": len(df),
    }
    all_metrics["全局"] = overall
    print(f"\nOverall: RMSE={overall['RMSE']}, MAE={overall['MAE']}, R2={overall['R2']}")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\n指标已保存到 {METRICS_PATH}")


if __name__ == "__main__":
    evaluate()
