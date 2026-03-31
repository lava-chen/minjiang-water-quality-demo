"""
clean_data.py
读取 merged_raw.csv，进行数据清洗：
  1. 将 '*' 和空字符串替换为 NaN
  2. 数值列类型转换
  3. 删除 100% 缺失的列（叶绿素a、藻密度）
  4. 对小段缺失进行线性插值
输出到 data/processed/cleaned.csv
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "merged_raw.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned.csv")

NUMERIC_COLUMNS = [
    "水温", "pH", "溶解氧", "电导率", "浊度",
    "高锰酸盐指数", "氨氮", "总磷", "总氮",
    "叶绿素a", "藻密度"
]


def main():
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    print(f"读取数据: {len(df)} 行, {len(df.columns)} 列")

    # 将 '*' 和空白替换为 NaN
    df.replace(["*", "", " "], np.nan, inplace=True)

    # 数值列强制转换
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 统计缺失率，删除缺失率 > 90% 的数值列
    drop_cols = []
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            miss_rate = df[col].isna().mean() * 100
            print(f"  {col}: 缺失 {miss_rate:.1f}%")
            if miss_rate > 90:
                drop_cols.append(col)

    if drop_cols:
        print(f"删除高缺失列: {drop_cols}")
        df.drop(columns=drop_cols, inplace=True)

    # 删除溶解氧(目标变量)为空的行
    if "溶解氧" in df.columns:
        before = len(df)
        df.dropna(subset=["溶解氧"], inplace=True)
        print(f"删除溶解氧缺失行: {before} -> {len(df)}")

    # 删除站点情况不为"正常"的行
    if "站点情况" in df.columns:
        before = len(df)
        df = df[df["站点情况"] == "正常"]
        print(f"保留站点情况=正常: {before} -> {len(df)}")

    # 按站点排序后，对数值列进行线性插值（最多填充连续 3 个缺失值）
    remaining_numeric = [c for c in NUMERIC_COLUMNS if c in df.columns]
    df.sort_values(["断面名称", "监测时间"], inplace=True)

    for station in df["断面名称"].unique():
        mask = df["断面名称"] == station
        df.loc[mask, remaining_numeric] = (
            df.loc[mask, remaining_numeric]
            .interpolate(method="linear", limit=3)
        )

    df.reset_index(drop=True, inplace=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"清洗完成，已保存到 {OUTPUT_PATH}，共 {len(df)} 行")


if __name__ == "__main__":
    main()
