"""
select_stations.py
根据缺失率统计和数据质量，筛选优质站点。
选择标准：溶解氧缺失率 < 5%，且日均数据覆盖天数 >= 200 天。
输出最终用于建模的数据到 data/processed/daily_water_quality.csv
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DAILY_PATH = os.path.join(BASE_DIR, "data", "processed", "daily_all_stations.csv")
MISS_STATS_PATH = os.path.join(BASE_DIR, "Data", "2024年", "长江上游断面_缺失率统计.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "daily_water_quality.csv")

# 溶解氧缺失率阈值(%)
DO_MISS_THRESHOLD = 5.0
# 最少覆盖天数
MIN_DAYS = 200


def main():
    # 读取缺失率统计
    miss_df = pd.read_csv(MISS_STATS_PATH, encoding="utf-8-sig")
    print("缺失率统计表列:", list(miss_df.columns))

    do_col = None
    for c in miss_df.columns:
        if "溶解氧" in c:
            do_col = c
            break

    if do_col:
        miss_df[do_col] = pd.to_numeric(miss_df[do_col], errors="coerce")
        good_stations = miss_df[miss_df[do_col] < DO_MISS_THRESHOLD]["断面名称"].tolist()
        print(f"溶解氧缺失率 < {DO_MISS_THRESHOLD}% 的站点 ({len(good_stations)}): {good_stations}")
    else:
        print("警告: 未找到溶解氧缺失率列，使用全部站点")
        good_stations = None

    # 读取日聚合数据
    daily = pd.read_csv(DAILY_PATH, encoding="utf-8-sig")
    print(f"日聚合数据: {len(daily)} 行, {daily['断面名称'].nunique()} 站点")

    # 按缺失率筛选
    if good_stations:
        daily = daily[daily["断面名称"].isin(good_stations)]
        print(f"缺失率筛选后: {len(daily)} 行, {daily['断面名称'].nunique()} 站点")

    # 按覆盖天数筛选
    station_days = daily.groupby("断面名称")["日期"].nunique()
    enough_stations = station_days[station_days >= MIN_DAYS].index.tolist()
    daily = daily[daily["断面名称"].isin(enough_stations)]
    print(f"覆盖天数 >= {MIN_DAYS} 筛选后: {len(daily)} 行, {daily['断面名称'].nunique()} 站点")

    for s in enough_stations:
        days = station_days[s]
        print(f"  {s}: {days} 天")

    daily.sort_values(["断面名称", "日期"], inplace=True)
    daily.reset_index(drop=True, inplace=True)
    daily.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n最终输出: {OUTPUT_PATH}, 共 {len(daily)} 行")


if __name__ == "__main__":
    main()
