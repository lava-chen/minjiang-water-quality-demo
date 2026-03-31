"""
build_kg.py
岷江水系知识图谱构建脚本

从以下数据源读取信息，构建领域知识图谱，导出 JSON 供前端"驱动因素解释卡片"调用：
  - data/processed/minjiang_stations.csv  → 15 个站点元信息
  - data/processed/minjiang_4h.csv        → 4 小时水质监测数据
  - models/all_results.json               → 7 个重点站的模型训练结果
  - models/all_feature_importance.json     → 7 个重点站的特征重要性
  - models/{站名}/test_predictions.csv    → 测试集预测值（用于识别低 DO 事件）
"""

import os
import json
import pandas as pd
import networkx as nx

# ====================== 路径配置 ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATION_CSV = os.path.join(BASE_DIR, "data", "processed", "minjiang_stations.csv")
WATER_DATA = os.path.join(BASE_DIR, "data", "processed", "minjiang_4h.csv")
RESULTS_JSON = os.path.join(BASE_DIR, "models", "all_results.json")
IMPORTANCE_JSON = os.path.join(BASE_DIR, "models", "all_feature_importance.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_PATH = os.path.join(BASE_DIR, "src", "knowledge_graph", "kg_data.json")

# ====================== 水质参数定义 ======================
# 9 个原始监测参数
PARAMETERS = {
    "溶解氧": {"unit": "mg/L", "category": "物理化学", "description": "水中溶解氧含量，反映水体自净能力"},
    "水温": {"unit": "°C", "category": "物理", "description": "水体温度，影响溶氧和生化过程"},
    "pH": {"unit": "无量纲", "category": "化学", "description": "酸碱度，影响水生生物生存"},
    "电导率": {"unit": "uS/cm", "category": "物理", "description": "水体导电能力，反映离子浓度"},
    "浊度": {"unit": "NTU", "category": "物理", "description": "水体透明度，反映悬浮物含量"},
    "高锰酸盐指数": {"unit": "mg/L", "category": "化学", "description": "有机污染综合指标"},
    "氨氮": {"unit": "mg/L", "category": "化学", "description": "含氮有机物分解产物，水体富营养化指标"},
    "总磷": {"unit": "mg/L", "category": "化学", "description": "水体富营养化关键控制因子"},
    "总氮": {"unit": "mg/L", "category": "化学", "description": "氮素总量，富营养化指标"},
}

# ====================== 水质类别及 DO 标准 (mg/L) ======================
WATER_QUALITY_CLASSES = {
    "I": {"level": 1, "do_threshold": 7.5, "description": "适用于源头水、国家自然保护区"},
    "II": {"level": 2, "do_threshold": 6.0, "description": "适用于集中式饮用水源地一级保护区等"},
    "III": {"level": 3, "do_threshold": 5.0, "description": "适用于集中式饮用水源地二级保护区等"},
    "IV": {"level": 4, "do_threshold": 3.0, "description": "适用于一般工业用水及人体非直接接触的娱乐用水"},
    "V": {"level": 5, "do_threshold": 2.0, "description": "适用于农业用水及一般景观用水"},
}

# ====================== 岷江水系河流拓扑 ======================
# 每条河流上的站点，按上游→下游排列（序号对应 minjiang_stations.csv 中的"上下游序号"）
RIVER_TOPOLOGY = {
    "岷江干流": ["渭门桥", "都江堰水文站", "岳店子下", "悦来渡口", "月波"],
    "大渡河": ["大岗山", "三谷庄", "李码头"],
    "青衣江": ["木城镇", "龟都府", "姜公堰"],
    "府河": ["黄龙溪"],
    "江安河": ["二江寺"],
    "马边河": ["马边河河口"],
    "越溪河": ["越溪河两河口"],
}

# 支流汇入关系：{支流末端站点: 汇入的干流站点}
TRIBUTARY_CONFLUENCES = {
    "李码头": "悦来渡口",      # 大渡河 → 岷江干流
    "姜公堰": "悦来渡口",      # 青衣江 → 岷江干流
    "黄龙溪": "岳店子下",      # 府河   → 岷江干流
    "二江寺": "岳店子下",      # 江安河 → 岷江干流
    "马边河河口": "悦来渡口",  # 马边河 → 岷江干流
    "越溪河两河口": "月波",    # 越溪河 → 岷江干流
}

# ====================== 特征对 DO 影响的领域解释 ======================
# 包含原始监测参数 + 时间特征 + 滚动工程特征
FEATURE_EXPLANATIONS = {
    # --- 原始监测参数 ---
    "水温": "水温升高导致溶解氧饱和浓度下降，是DO最直接的物理驱动因素。夏季高温期常出现DO低谷。",
    "pH": "pH影响水体中碳酸平衡和生物活性。偏碱性条件下藻类光合作用旺盛，可短期提升DO。",
    "电导率": "反映水体总离子浓度，间接指示排污或地质背景变化，与DO存在协变关系。",
    "浊度": "高浊度减弱光照穿透，抑制藻类光合产氧，同时悬浮有机物分解消耗DO。",
    "高锰酸盐指数": "反映水体有机污染程度，有机物好氧分解直接消耗溶解氧。",
    "氨氮": "氨氮硝化过程消耗大量溶解氧，是耗氧型污染的重要指示。",
    "总磷": "磷是富营养化限制因子，高磷促进藻类繁殖，影响DO的昼夜和季节波动。",
    "总氮": "氮素过量加剧富营养化，间接通过生物过程影响DO浓度。",
    "溶解氧": "DO自身具有强时序自相关性，历史DO值是预测未来趋势的重要基准。",
    # --- 时间周期特征 ---
    "hour_sin": "一天内的时间周期（正弦分量），捕捉DO的昼夜节律——白天光合产氧，夜间呼吸耗氧。",
    "hour_cos": "一天内的时间周期（余弦分量），与hour_sin配合编码24小时周期变化。",
    "dayofyear_sin": "一年内的季节周期（正弦分量），捕捉DO的季节性变化——冬高夏低。",
    "dayofyear_cos": "一年内的季节周期（余弦分量），与dayofyear_sin配合编码365天周期。",
    # --- 滚动工程特征 ---
    "DO_24h_mean": "过去24小时DO平均值，反映近期DO整体水平，帮助模型识别稳态趋势。",
    "DO_24h_std": "过去24小时DO标准差，衡量近期DO波动幅度，高波动通常预示水质变化。",
    "DO_diff": "当前DO与上一个时间步的差值，捕捉DO的瞬时变化趋势（上升/下降）。",
    "DO_72h_mean": "过去72小时DO平均值，提供更长时间尺度的基线参考，平滑短期波动。",
    "temp_24h_mean": "过去24小时水温平均值，反映近期温度水平对DO饱和浓度的影响。",
    "temp_diff": "当前水温与上一个时间步的差值，捕捉温度的急剧变化对DO的冲击。",
}


def build_graph():
    """构建岷江水系知识图谱"""
    G = nx.DiGraph()

    # ========== 1. 加载站点元信息 ==========
    print("  加载站点信息...")
    stations_df = pd.read_csv(STATION_CSV, encoding="utf-8-sig")
    station_info = {}
    for _, row in stations_df.iterrows():
        station_info[row["站点名称"]] = {
            "province": row["省份"],
            "city": row["城市"],
            "river": row["河流"],
            "lon": float(row["经度"]),
            "lat": float(row["纬度"]),
            "order": int(row["上下游序号"]),
            "is_key": bool(row["重点站"]),
            "data_count": int(row["数据条数"]),
            "start_time": row["起始时间"],
            "end_time": row["结束时间"],
        }

    # ========== 2. 加载 4 小时水质数据，统计每站水质等级 ==========
    print("  加载水质数据...")
    water_df = pd.read_csv(WATER_DATA, encoding="utf-8-sig", low_memory=False)

    # 全角→半角罗马数字映射
    qclass_map = {"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV", "Ⅴ": "V"}
    water_df["水质_std"] = water_df["水质"].map(qclass_map).fillna("II")

    # 每站最常见的水质类别
    station_main_class = water_df.groupby("站点名称")["水质_std"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "II"
    ).to_dict()

    # 每站水质类别的分布（前端可用于饼图等）
    station_class_dist = {}
    for name, grp in water_df.groupby("站点名称"):
        dist = grp["水质_std"].value_counts(normalize=True).round(4).to_dict()
        station_class_dist[name] = dist

    # 每站最新一条DO值
    station_latest_do = {}
    for name, grp in water_df.groupby("站点名称"):
        latest = grp.sort_values("监测时间").iloc[-1]
        station_latest_do[name] = round(float(latest["溶解氧"]), 2) if pd.notna(latest["溶解氧"]) else None

    # ========== 3. 加载模型结果 ==========
    print("  加载模型结果...")
    model_results = {}
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            results_list = json.load(f)
            for r in results_list:
                model_results[r["station"]] = r

    # ========== 4. 加载特征重要性 ==========
    print("  加载特征重要性...")
    feature_importance = {}
    if os.path.exists(IMPORTANCE_JSON):
        with open(IMPORTANCE_JSON, "r", encoding="utf-8") as f:
            feature_importance = json.load(f)

    # ========== 5. 扫描低 DO 事件（从 test_predictions.csv） ==========
    print("  扫描低DO事件...")
    events = []
    DO_LOW_THRESHOLD = 5.0  # III 类水质标准线
    for station_name in model_results:
        pred_path = os.path.join(MODELS_DIR, station_name, "test_predictions.csv")
        if not os.path.exists(pred_path):
            continue
        pred_df = pd.read_csv(pred_path, encoding="utf-8-sig")
        # 检查 4h 步长的实际 DO（最近的预测步）
        if "实际DO_4h" in pred_df.columns:
            low_rows = pred_df[pred_df["实际DO_4h"] < DO_LOW_THRESHOLD]
            for _, row in low_rows.iterrows():
                events.append({
                    "type": "DO_LOW",
                    "station": station_name,
                    "time": str(row["时间"]),
                    "value": round(float(row["实际DO_4h"]), 2),
                    "description": f"溶解氧={row['实际DO_4h']:.2f}mg/L，低于III类水质标准({DO_LOW_THRESHOLD}mg/L)",
                })

    # ==================== 构建图节点 ====================

    # --- 数据源节点 ---
    G.add_node("DS_国家地表水", type="DataSource",
               name="国家地表水水质自动监测实时数据发布系统",
               url="https://szzdjc.cnemc.cn:8070/GJZ/Business/Publish/Main.html",
               institution="中国环境监测总站")

    # --- 河流节点 ---
    rivers_info = {
        "岷江干流": {"full_name": "岷江干流", "level": "干流", "description": "岷江主河道，发源于松潘，汇入长江"},
        "大渡河": {"full_name": "大渡河", "level": "一级支流", "description": "岷江最大支流，在乐山汇入岷江"},
        "青衣江": {"full_name": "青衣江", "level": "一级支流", "description": "岷江支流，在乐山汇入岷江"},
        "府河": {"full_name": "府河", "level": "二级支流", "description": "成都市区主要河流，在岳店子下游汇入岷江"},
        "江安河": {"full_name": "江安河", "level": "二级支流", "description": "成都市区河流，在岳店子下游汇入岷江"},
        "马边河": {"full_name": "马边河", "level": "一级支流", "description": "岷江支流，在悦来渡口附近汇入岷江"},
        "越溪河": {"full_name": "越溪河", "level": "一级支流", "description": "岷江支流，在月波附近汇入岷江"},
    }
    for river_name, rinfo in rivers_info.items():
        G.add_node(f"R_{river_name}", type="River", name=river_name, **rinfo)

    # 支流汇入关系（River → River）
    for river_name in ["大渡河", "青衣江", "马边河", "越溪河"]:
        G.add_edge(f"R_{river_name}", "R_岷江干流", relation="tributary_of")
    for river_name in ["府河", "江安河"]:
        G.add_edge(f"R_{river_name}", "R_岷江干流", relation="tributary_of")

    # --- 城市/区域节点 ---
    cities = set()
    for info in station_info.values():
        cities.add(info["city"])
    for city in cities:
        G.add_node(f"C_{city}", type="City", name=city, province="四川省")

    # --- 水质参数节点 ---
    for pname, pinfo in PARAMETERS.items():
        G.add_node(f"P_{pname}", type="Parameter", name=pname, **pinfo)

    # --- 水质类别节点 ---
    for cname, cinfo in WATER_QUALITY_CLASSES.items():
        G.add_node(f"WQ_{cname}", type="WaterQualityClass", name=cname, **cinfo)

    # --- 站点节点 ---
    for station_name, info in station_info.items():
        G.add_node(f"S_{station_name}", type="Station",
                   name=station_name,
                   city=info["city"],
                   province=info["province"],
                   river=info["river"],
                   lon=info["lon"],
                   lat=info["lat"],
                   order=info["order"],
                   is_key=info["is_key"],
                   data_count=info["data_count"],
                   data_period=f"{info['start_time']} ~ {info['end_time']}",
                   latest_do=station_latest_do.get(station_name),
                   main_quality_class=station_main_class.get(station_name, "II"),
                   quality_class_dist=station_class_dist.get(station_name, {}))

        # 关系：站点 → 河流
        G.add_edge(f"S_{station_name}", f"R_{info['river']}", relation="located_at")
        # 关系：站点 → 城市
        G.add_edge(f"S_{station_name}", f"C_{info['city']}", relation="belongs_to")
        # 关系：站点 → 数据源
        G.add_edge(f"S_{station_name}", "DS_国家地表水", relation="sourced_from")
        # 关系：站点 → 水质类别
        qclass = station_main_class.get(station_name, "II")
        if f"WQ_{qclass}" in G.nodes:
            G.add_edge(f"S_{station_name}", f"WQ_{qclass}", relation="classified_as")
        # 关系：站点 → 监测参数
        for pname in PARAMETERS:
            G.add_edge(f"S_{station_name}", f"P_{pname}", relation="monitors")

    # --- 站点上下游关系（同一条河内） ---
    for river_name, station_list in RIVER_TOPOLOGY.items():
        # 只处理图中存在的站点
        ordered = [s for s in station_list if f"S_{s}" in G.nodes]
        for i in range(len(ordered) - 1):
            G.add_edge(f"S_{ordered[i]}", f"S_{ordered[i+1]}", relation="upstream_of")

    # --- 支流汇入关系（站点→站点，跨河流） ---
    for tributary_end, main_station in TRIBUTARY_CONFLUENCES.items():
        if f"S_{tributary_end}" in G.nodes and f"S_{main_station}" in G.nodes:
            G.add_edge(f"S_{tributary_end}", f"S_{main_station}", relation="flows_into",
                       description=f"{tributary_end}所在支流汇入岷江干流({main_station})")

    # --- 模型结果节点（仅 7 个重点站） ---
    for station_name, result in model_results.items():
        overall = result.get("overall", {})
        step_metrics = result.get("step_metrics", [])
        hyperparams = result.get("hyperparams", {})
        split_info = result.get("split_info", {})

        # 从特征重要性中提取 TOP 5
        imp_data = feature_importance.get(station_name, {})
        fi = imp_data.get("feature_importance", {})
        # 按绝对值降序排列，取 TOP 5
        sorted_feats = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = [
            {"name": f, "importance": round(v, 4), "explanation": FEATURE_EXPLANATIONS.get(f, "")}
            for f, v in sorted_feats[:5]
        ]

        # 注意力权重
        attn = imp_data.get("attention_weights", {})

        mr_id = f"MR_{station_name}"
        G.add_node(mr_id, type="ModelResult",
                   model_type="LSTM+Attention",
                   station=station_name,
                   # 总体指标
                   rmse=overall.get("RMSE", 0),
                   mae=overall.get("MAE", 0),
                   r2=overall.get("R2", 0),
                   mape=overall.get("MAPE", 0),
                   # 各步长指标
                   step_metrics=step_metrics,
                   # TOP5 驱动特征（含解释）
                   top_features=top_features,
                   # 注意力权重
                   attention_weights=attn,
                   # 训练信息
                   best_epoch=result.get("best_epoch", 0),
                   total_epochs=result.get("total_epochs", 0),
                   train_time_sec=result.get("train_time_sec", 0),
                   # 数据划分
                   train_period=split_info.get("train_period", ""),
                   val_period=split_info.get("val_period", ""),
                   test_period=split_info.get("test_period", ""),
                   # 超参数
                   hyperparams=hyperparams)

        # 关系：DO 参数 ← 模型预测
        G.add_edge("P_溶解氧", mr_id, relation="predicted_by")
        # 关系：模型结果 → 站点
        G.add_edge(mr_id, f"S_{station_name}", relation="model_for")
        # 关系：模型结果 → 驱动特征（TOP5）
        for feat in top_features:
            feat_name = feat["name"]
            if f"P_{feat_name}" in G.nodes:
                G.add_edge(mr_id, f"P_{feat_name}", relation="driven_by",
                           importance=feat["importance"])

    # --- 水质事件节点 ---
    for i, evt in enumerate(events):
        evt_id = f"E_{i}"
        G.add_node(evt_id, type="Event", **evt)
        if f"S_{evt['station']}" in G.nodes:
            G.add_edge(evt_id, f"S_{evt['station']}", relation="occurred_at")
        G.add_edge(evt_id, "P_溶解氧", relation="affects")

    return G


def export_to_json(G: nx.DiGraph):
    """将 NetworkX 图导出为前端可用的 JSON 格式"""
    nodes = []
    for nid, data in G.nodes(data=True):
        node = {"id": nid}
        # 序列化所有属性（包括 list/dict 类型）
        for k, v in data.items():
            node[k] = v
        nodes.append(node)

    edges = []
    for u, v, data in G.edges(data=True):
        edge = {"source": u, "target": v}
        edge.update(data)
        edges.append(edge)

    # 附加领域知识
    result = {
        "metadata": {
            "project": "岷江水系水质预测系统",
            "total_stations": 15,
            "key_stations": 7,
            "rivers": list(RIVER_TOPOLOGY.keys()),
            "data_source": "国家地表水水质自动监测实时数据发布系统",
            "model": "LSTM+Attention",
            "target": "溶解氧(DO)",
            "forecast_horizon": "4h / 8h / 12h / 16h / 20h / 24h",
        },
        "nodes": nodes,
        "edges": edges,
        "feature_explanations": FEATURE_EXPLANATIONS,
        "water_quality_standards": WATER_QUALITY_CLASSES,
        "river_topology": RIVER_TOPOLOGY,
        "tributary_confluences": TRIBUTARY_CONFLUENCES,
    }
    return result


def main():
    print("=" * 50)
    print("  岷江水系知识图谱构建")
    print("=" * 50)

    G = build_graph()

    print(f"\n  图谱统计:")
    print(f"    节点数: {G.number_of_nodes()}")
    print(f"    边数:   {G.number_of_edges()}")

    # 按类型统计节点
    type_counts = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"\n  节点类型分布:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}")

    # 按类型统计边
    rel_counts = {}
    for _, _, data in G.edges(data=True):
        r = data.get("relation", "unknown")
        rel_counts[r] = rel_counts.get(r, 0) + 1
    print(f"\n  关系类型分布:")
    for r, c in sorted(rel_counts.items()):
        print(f"    {r}: {c}")

    # 导出 JSON
    kg_json = export_to_json(G)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(kg_json, f, ensure_ascii=False, indent=2)
    print(f"\n  知识图谱已保存到: {OUTPUT_PATH}")

    # 打印 TOP5 特征摘要
    print(f"\n  各重点站 TOP5 驱动因素:")
    for nid, data in G.nodes(data=True):
        if data.get("type") == "ModelResult":
            station = data["station"]
            r2 = data["r2"]
            feats = data.get("top_features", [])
            feat_str = " | ".join([f"{f['name']}({f['importance']:.3f})" for f in feats])
            print(f"    {station} (R2={r2:.4f}): {feat_str}")

    print(f"\n  低DO事件数: {type_counts.get('Event', 0)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
