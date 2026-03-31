"""
aggregate_daily.py
将 4 小时间隔的数据聚合为日均值。
对每个站点每天计算 mean / min / max / std，
输出到 data/processed/daily_all_stations.csv
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "daily_all_stations.csv")

NUMERIC_COLUMNS = [
    "水温", "pH", "溶解氧", "电导率", "浊度",
    "高锰酸盐指数", "氨氮", "总磷", "总氮"
]


def parse_monitoring_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    解析监测时间列。格式如 '12-28 20:00'，需要补全年份。
    文件名包含年月日信息，但合并后原始监测时间只有 'MM-DD HH:MM'。
    默认全部按 2024 年处理。
    """
    df = df.copy()
    df["监测时间_str"] = "2024-" + df["监测时间"].astype(str).str.strip()
    df["datetime"] = pd.to_datetime(df["监测时间_str"], format="%Y-%m-%d %H:%M", errors="coerce")
    df.drop(columns=["监测时间_str"], inplace=True)
    df.dropna(subset=["datetime"], inplace=True)
    df["日期"] = df["datetime"].dt.date
    return df


def main():
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    print(f"读取数据: {len(df)} 行")

    df = parse_monitoring_time(df)
    print(f"解析时间后: {len(df)} 行")

    available_numeric = [c for c in NUMERIC_COLUMNS if c in df.columns]

    # 按站点+日期聚合
    agg_dict = {}
    for col in available_numeric:
        agg_dict[col] = ["mean", "min", "max", "std"]

    grouped = df.groupby(["省份", "流域", "断面名称", "日期"])
    daily = grouped.agg(agg_dict)

    # 展平多级列名: 如 ('溶解氧', 'mean') -> '溶解氧_mean'
    daily.columns = ["_".join(col).strip() for col in daily.columns]
    daily.reset_index(inplace=True)

    # 同时记录每日的主要水质类别（众数）
    mode_df = grouped["水质类别"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
    mode_df = mode_df.reset_index()
    mode_df.columns = ["省份", "流域", "断面名称", "日期", "水质类别"]

    daily = daily.merge(mode_df, on=["省份", "流域", "断面名称", "日期"], how="left")

    daily.sort_values(["断面名称", "日期"], inplace=True)
    daily.reset_index(drop=True, inplace=True)

    daily.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"日聚合完成: {len(daily)} 行, 已保存到 {OUTPUT_PATH}")
    print(f"站点数: {daily['断面名称'].nunique()}, 日期范围: {daily['日期'].min()} ~ {daily['日期'].max()}")


if __name__ == "__main__":
    main()
