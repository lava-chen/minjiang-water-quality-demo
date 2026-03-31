"""
prepare_minjiang.py
从 1_清洗合并后数据/ 中筛选岷江水系15个站点，
只保留2022年之后的数据，重采样到标准4小时网格，添加时间特征。

15个站全部保留（网站展示+知识图谱），其中7个标记为重点站（模型训练+深入分析）：
  岷江干流5个：渭门桥、都江堰水文站、岳店子下、悦来渡口、月波
  主要支流汇入口2个：李码头（大渡河）、姜公堰（青衣江）
"""

import os
import numpy as np
import pandas as pd

# ============ 路径配置 ============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "1_清洗合并后数据")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

# 只保留此日期之后的数据（之前的数据采样间隔不稳定）
CUTOFF_DATE = "2022-01-01"

# 标准4小时网格：每天6个时间点
RESAMPLE_HOURS = [0, 4, 8, 12, 16, 20]

# ============ 岷江水系15个站点 ============
# (文件名, 所属河流, 上下游序号, 是否重点站)
MINJIANG_STATIONS = {
    "渭门桥":       ("四川省_阿坝藏族羌族自治州_渭门桥.csv",  "岷江干流", 1, True),
    "都江堰水文站": ("四川省_成都市_都江堰水文站.csv",        "岷江干流", 2, True),
    "岳店子下":     ("四川省_成都市_岳店子下.csv",            "岷江干流", 3, True),
    "悦来渡口":     ("四川省_乐山市_悦来渡口.csv",            "岷江干流", 4, True),
    "月波":         ("四川省_宜宾市_月波.csv",                "岷江干流", 5, True),
    "李码头":       ("四川省_乐山市_李码头.csv",              "大渡河",   3, True),
    "姜公堰":       ("四川省_乐山市_姜公堰.csv",              "青衣江",   3, True),
    "大岗山":       ("四川省_雅安市_大岗山.csv",              "大渡河",   1, False),
    "三谷庄":       ("四川省_雅安市_三谷庄.csv",              "大渡河",   2, False),
    "木城镇":       ("四川省_乐山市_木城镇.csv",              "青衣江",   1, False),
    "龟都府":       ("四川省_雅安市_龟都府.csv",              "青衣江",   2, False),
    "黄龙溪":       ("四川省_眉山市_黄龙溪.csv",              "府河",     1, False),
    "二江寺":       ("四川省_成都市_二江寺.csv",              "江安河",   1, False),
    "马边河河口":   ("四川省_乐山市_马边河河口.csv",          "马边河",   1, False),
    "越溪河两河口": ("四川省_宜宾市_越溪河两河口.csv",        "越溪河",   1, False),
}

# 9个水质参数
QUALITY_COLUMNS = [
    "水温", "pH", "溶解氧", "电导率", "浊度",
    "高锰酸盐指数", "氨氮", "总磷", "总氮",
]

DROP_COLUMNS = ["总有机碳", "叶绿素α", "藻密度", "站点"]


def read_and_clean(filepath: str) -> pd.DataFrame:
    """读取CSV，基础清洗"""
    df = pd.read_csv(filepath, encoding="utf-8-sig", skiprows=[1])

    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    df["监测时间"] = pd.to_datetime(df["监测时间"])
    df = df.sort_values("监测时间").drop_duplicates(subset=["监测时间"], keep="first")

    for col in QUALITY_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def resample_to_4h_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    将不规则时间序列重采样到标准4小时网格。

    方法：
    1. 只保留2022年之后的数据
    2. 生成标准4h时间网格（00:00, 04:00, ..., 20:00）
    3. 将原始数据与网格对齐（最近邻匹配，容差±2小时）
    4. 对小间隙（≤24小时=6个连续缺失点）做线性插值
    5. 丢弃仍然缺失的行
    """
    # 只保留2022年之后
    df = df[df["监测时间"] >= CUTOFF_DATE].copy()
    if len(df) == 0:
        return df

    # 生成标准4h时间网格
    t_min = df["监测时间"].min().normalize()  # 当天00:00
    t_max = df["监测时间"].max().normalize() + pd.Timedelta(days=1)
    grid = pd.date_range(start=t_min, end=t_max, freq="4h")

    # 将原始数据的时间戳"吸附"到最近的网格点（容差±2小时）
    df = df.set_index("监测时间")
    grid_df = pd.DataFrame(index=grid)
    grid_df.index.name = "监测时间"

    # merge_asof: 找到网格点对应的最近原始数据（在±2小时内）
    df_sorted = df.sort_index()
    grid_series = pd.DataFrame({"监测时间": grid})

    merged = pd.merge_asof(
        grid_series.sort_values("监测时间"),
        df_sorted.reset_index().sort_values("监测时间"),
        on="监测时间",
        tolerance=pd.Timedelta("2h"),
        direction="nearest",
    )
    merged = merged.set_index("监测时间")

    # 对水质参数列做线性插值（最多填6个连续缺失点=24小时）
    for col in QUALITY_COLUMNS:
        if col in merged.columns:
            merged[col] = merged[col].interpolate(method="linear", limit=6)

    # 丢弃溶解氧仍缺失的行（这些是大段缺失，不可靠）
    merged = merged.dropna(subset=["溶解氧"])

    # 经纬度、省份等元数据是每站固定的，用前后填充补齐
    meta_cols = ["省份", "城市", "流域", "河流", "站点名称", "经度", "纬度", "水质"]
    for col in meta_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill().bfill()

    # 剩余极少量水质参数缺失用前后值填充（不超过总量的1%）
    for col in QUALITY_COLUMNS:
        if col in merged.columns:
            merged[col] = merged[col].ffill(limit=3).bfill(limit=3)

    return merged.reset_index()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加4个循环时间特征"""
    dt = df["监测时间"]

    hour = dt.dt.hour + dt.dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    dayofyear = dt.dt.dayofyear
    df["dayofyear_sin"] = np.sin(2 * np.pi * dayofyear / 365.0)
    df["dayofyear_cos"] = np.cos(2 * np.pi * dayofyear / 365.0)

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_data = []
    station_info_rows = []

    print("=" * 70)
    print(f"岷江水系数据准备（仅保留 {CUTOFF_DATE} 之后，重采样到4h网格）")
    print("=" * 70)

    for name, (filename, river, order, is_key) in MINJIANG_STATIONS.items():
        filepath = os.path.join(RAW_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  [跳过] {name}: 文件不存在")
            continue

        # 读取 + 基础清洗
        df = read_and_clean(filepath)
        raw_after_cut = len(df[df["监测时间"] >= CUTOFF_DATE])

        # 重采样到4h网格
        df = resample_to_4h_grid(df)
        if len(df) == 0:
            print(f"  [跳过] {name}: 2022年后无数据")
            continue

        # 添加时间特征
        df = add_time_features(df)

        # 提取元信息
        first = df.iloc[0]
        station_info_rows.append({
            "站点名称": name,
            "省份": first.get("省份", ""),
            "城市": first.get("城市", ""),
            "河流": river,
            "经度": first.get("经度", 0),
            "纬度": first.get("纬度", 0),
            "上下游序号": order,
            "重点站": is_key,
            "数据条数": len(df),
            "起始时间": df["监测时间"].min().strftime("%Y-%m-%d"),
            "结束时间": df["监测时间"].max().strftime("%Y-%m-%d"),
        })

        # 添加标记列
        df["站点名称"] = name
        df["河流系统"] = river
        df["重点站"] = is_key
        all_data.append(df)

        tag = "★重点" if is_key else "  普通"
        # 计算实际每天平均记录数
        days = (df["监测时间"].max() - df["监测时间"].min()).days + 1
        avg_per_day = len(df) / days if days > 0 else 0

        print(f"  {tag} {name:10s} | {river:6s} | "
              f"原始 {raw_after_cut:5d} -> 重采样 {len(df):5d} 条 | "
              f"平均 {avg_per_day:.1f}条/天 | "
              f"{df['监测时间'].min().strftime('%Y-%m')} ~ "
              f"{df['监测时间'].max().strftime('%Y-%m')}")

    # 合并
    combined = pd.concat(all_data, ignore_index=True)

    output_cols = [
        "站点名称", "河流系统", "重点站", "监测时间",
        *QUALITY_COLUMNS,
        "hour_sin", "hour_cos", "dayofyear_sin", "dayofyear_cos",
        "水质", "经度", "纬度",
    ]
    output_cols = [c for c in output_cols if c in combined.columns]
    combined = combined[output_cols]

    # 保存数据
    data_path = os.path.join(OUTPUT_DIR, "minjiang_4h.csv")
    combined.to_csv(data_path, index=False, encoding="utf-8-sig")

    # 保存站点信息
    station_df = pd.DataFrame(station_info_rows)
    station_path = os.path.join(OUTPUT_DIR, "minjiang_stations.csv")
    station_df.to_csv(station_path, index=False, encoding="utf-8-sig")

    # 汇总
    key_data = combined[combined["重点站"] == True]
    print(f"\n{'='*70}")
    print(f"输出文件: {data_path}")
    print(f"  全部站点: {combined['站点名称'].nunique()} 个, {len(combined)} 条")
    print(f"  重点站:   {key_data['站点名称'].nunique()} 个, {len(key_data)} 条")

    # 验证时间间隔一致性（抽查一个重点站）
    print(f"\n{'='*70}")
    print("时间间隔验证（渭门桥）:")
    check = combined[combined["站点名称"] == "渭门桥"].copy()
    check = check.sort_values("监测时间")
    diffs = check["监测时间"].diff().dt.total_seconds() / 3600
    diffs = diffs.dropna()
    print(f"  间隔分布: {diffs.value_counts().sort_index().to_dict()}")
    print(f"  4h占比: {(diffs == 4.0).sum() / len(diffs) * 100:.1f}%")
    print(f"  非4h记录数: {(diffs != 4.0).sum()}")

    # 数据质量
    print(f"\n{'='*70}")
    print("重点站数据质量:")
    for col in QUALITY_COLUMNS:
        if col in key_data.columns:
            m = key_data[col].isna().sum()
            print(f"  {col:12s} | 缺失 {m} / {len(key_data)} ({m/len(key_data)*100:.2f}%)")


if __name__ == "__main__":
    main()
