"""
merge_data.py
遍历 Data/2024年/ 下所有 CSV 文件，统一列名后合并为单一 DataFrame，
按 (断面名称, 监测时间) 去重，输出到 data/processed/merged_raw.csv
"""

import os
import glob
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(BASE_DIR, "Data", "2024年")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "merged_raw.csv")

EXPECTED_COLUMNS = [
    "省份", "流域", "断面名称", "监测时间", "水质类别",
    "水温", "pH", "溶解氧", "电导率", "浊度",
    "高锰酸盐指数", "氨氮", "总磷", "总氮",
    "叶绿素a", "藻密度", "站点情况"
]

COLUMN_RENAME_MAP = {
    "pH无量纲": "pH",
    "溶解氧mgL": "溶解氧",
    "电导率μScm": "电导率",
    "浊度NTU": "浊度",
    "高锰酸盐指数mgL": "高锰酸盐指数",
    "氨氮mgL": "氨氮",
    "总磷mgL": "总磷",
    "总氮mgL": "总氮",
    "叶绿素αmgL": "叶绿素a",
    "藻密度cellsL": "藻密度",
}


def load_and_rename(filepath: str) -> pd.DataFrame | None:
    """读取单个 CSV 并统一列名，失败时返回 None"""
    try:
        try:
            df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding="gbk", on_bad_lines="skip")
    except Exception as e:
        print(f"  跳过文件 {os.path.basename(filepath)}: {e}")
        return None

    df.rename(columns=COLUMN_RENAME_MAP, inplace=True)

    # 必须包含"断面名称"和"监测时间"列
    if "断面名称" not in df.columns or "监测时间" not in df.columns:
        return None

    available = [c for c in EXPECTED_COLUMNS if c in df.columns]
    return df[available]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 找到所有 Filtered_*.csv 数据文件（排除缺失率统计文件）
    pattern = os.path.join(RAW_DATA_DIR, "Filtered_*.csv*")
    csv_files = sorted(glob.glob(pattern))
    print(f"找到 {len(csv_files)} 个数据文件")

    frames = []
    skipped = 0
    for i, fp in enumerate(csv_files):
        df = load_and_rename(fp)
        if df is not None and len(df) > 0:
            frames.append(df)
        else:
            skipped += 1
        if (i + 1) % 200 == 0:
            print(f"  已读取 {i + 1}/{len(csv_files)} ...")
    print(f"成功读取 {len(frames)} 个文件, 跳过 {skipped} 个")

    merged = pd.concat(frames, ignore_index=True)
    print(f"合并后总行数: {len(merged)}")

    # 按 (断面名称, 监测时间) 去重，保留最后出现的记录
    before = len(merged)
    merged.drop_duplicates(subset=["断面名称", "监测时间"], keep="last", inplace=True)
    print(f"去重: {before} -> {len(merged)} 行")

    merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"已保存到 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
