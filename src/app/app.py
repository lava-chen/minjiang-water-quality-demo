"""
app.py — 岷江水系水质监测系统 · 总览仪表盘

一眼看全局：顶部指标卡片 → 地图 → 站点状态表。
所有数据统一从 kg_data.json 读取，无需直接访问 CSV / 模型文件。
"""

import os
import json
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# ====================== 页面配置 ======================
st.set_page_config(
    page_title="岷江水系水质监测系统",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ====================== 路径 ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KG_PATH = os.path.join(BASE_DIR, "src", "knowledge_graph", "kg_data.json")

# ====================== 常量 ======================
# 河流配色方案（柔和且可区分）
RIVER_COLORS = {
    "岷江干流": "#3478F6",   # 蓝
    "大渡河":   "#30B065",   # 绿
    "青衣江":   "#F5A623",   # 琥珀
    "府河":     "#8B5CF6",   # 紫
    "江安河":   "#E84393",   # 粉
    "马边河":   "#0ABDE3",   # 青
    "越溪河":   "#F97316",   # 橙
}

# 河流排序（干流在前、支流在后，与 kg_data.json 一致）
RIVER_ORDER = ["岷江干流", "大渡河", "青衣江", "府河", "江安河", "马边河", "越溪河"]

# 水质等级标签
QUALITY_LABELS = {"I": "Ⅰ类", "II": "Ⅱ类", "III": "Ⅲ类", "IV": "Ⅳ类", "V": "Ⅴ类"}
QUALITY_COLORS = {"I": "#34C759", "II": "#007AFF", "III": "#FF9500", "IV": "#FF6B35", "V": "#FF3B30"}


def get_do_status(do_val):
    """根据 DO 值返回 (状态文本, 颜色)，对应 GB3838 标准"""
    if do_val is None:
        return "无数据", "#8E8E93"
    if do_val >= 7.5:
        return "优",  "#34C759"
    if do_val >= 6.0:
        return "良",  "#007AFF"
    if do_val >= 5.0:
        return "一般", "#FF9500"
    if do_val >= 3.0:
        return "偏低", "#FF6B35"
    return "风险", "#FF3B30"


# ====================== 数据加载（缓存） ======================
@st.cache_data
def load_kg():
    with open(KG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


kg = load_kg()

# 从知识图谱中提取站点和模型结果
stations = [n for n in kg["nodes"] if n["type"] == "Station"]
model_results = {n["station"]: n for n in kg["nodes"] if n["type"] == "ModelResult"}

# ====================== 全局样式 ======================
st.markdown("""
<style>
    /* 隐藏 Streamlit 默认页脚 */
    footer {visibility: hidden;}

    /* ========== 侧边栏 · 深色主题 ========== */
    [data-testid="stSidebar"] {
        background: #1C1C1E !important;
    }
    /* 侧边栏内所有文字默认白色 */
    [data-testid="stSidebar"] * {
        color: #F5F5F7 !important;
    }
    /* 导航链接 */
    [data-testid="stSidebarNav"] a,
    [data-testid="stSidebar"] a {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 1rem !important;
        border-radius: 12px !important;
        margin: 4px 8px !important;
        color: #F5F5F7 !important;
        transition: background 0.2s;
    }
    [data-testid="stSidebarNav"] a:hover,
    [data-testid="stSidebar"] a:hover {
        background: #2C2C2E !important;
    }
    [data-testid="stSidebarNav"] a[aria-selected="true"] {
        background: #007AFF !important;
        color: #FFFFFF !important;
    }
    /* 导航文字 */
    [data-testid="stSidebarNav"] span {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #F5F5F7 !important;
    }
    /* 侧边栏标签文字（"选择站点" 等） */
    [data-testid="stSidebar"] label {
        color: #A1A1A6 !important;
        font-size: 0.95rem !important;
    }
    /* 侧边栏 selectbox / radio 控件 */
    [data-testid="stSidebar"] [data-baseweb="select"] {
        background: #2C2C2E !important;
        border-radius: 10px !important;
        border: 1px solid #3A3A3C !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #F5F5F7 !important;
    }
    /* 侧边栏分隔线 */
    [data-testid="stSidebar"] hr {
        border-color: #3A3A3C !important;
    }
    /* 侧边栏 caption */
    [data-testid="stSidebar"] .stCaption, 
    [data-testid="stSidebar"] small {
        color: #8E8E93 !important;
    }

    /* ========== 全局基础字号放大 ========== */
    html, body, [class*="css"] {
        font-size: 17px !important;
    }
    /* Streamlit 正文区域 */
    .stMarkdown, .stText, .stCaption, p, span, td, th, li {
        font-size: 1.05rem !important;
    }
    /* 表格字号 */
    [data-testid="stDataFrame"] {
        font-size: 1rem !important;
    }

    /* ========== 主标题 ========== */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1D1D1F;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        font-size: 1.15rem;
        color: #86868B;
        margin-bottom: 1.5rem;
    }

    /* ========== 指标卡片 ========== */
    .kpi-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        border: 1px solid #E5E5E7;
    }
    .kpi-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1D1D1F;
    }
    .kpi-label {
        font-size: 1rem;
        color: #86868B;
        margin-top: 4px;
    }

    /* ========== 图例行 ========== */
    .legend-row {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 16px;
        margin: 0.5rem 0 0.8rem;
        font-size: 1rem;
        color: #1D1D1F;
    }
    .legend-dot {
        width: 12px; height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }
    .legend-sep {
        color: #D1D1D6;
    }

    /* ========== 状态徽章 ========== */
    .badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 10px;
        font-size: 0.9rem;
        font-weight: 600;
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)


# ====================== 标题区域 ======================
st.markdown('<div class="hero-title">💧 岷江水系水质监测系统</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">基于 LSTM + Attention 深度学习模型的溶解氧 (DO) 预测与智能分析平台</div>',
    unsafe_allow_html=True,
)


# ====================== 顶部指标卡片 ======================
key_count = sum(1 for s in stations if s.get("is_key"))
river_count = len(kg["metadata"]["rivers"])
total_data = sum(s.get("data_count", 0) for s in stations)
avg_r2 = 0
if model_results:
    avg_r2 = sum(m["r2"] for m in model_results.values()) / len(model_results)

c1, c2, c3, c4 = st.columns(4)
for col, val, label in [
    (c1, "15",                "监测站点"),
    (c2, str(key_count),      "重点建模站"),
    (c3, str(river_count),    "监测河流"),
    (c4, f"{avg_r2:.2f}",     "平均模型 R²"),
]:
    col.markdown(f'''
    <div class="kpi-card">
        <div class="kpi-val">{val}</div>
        <div class="kpi-label">{label}</div>
    </div>''', unsafe_allow_html=True)

st.markdown("")


# ====================== 地图 ======================
st.markdown("### 🗺️ 站点分布")

# 图例
legend_items = "".join(
    f'<span><span class="legend-dot" style="background:{RIVER_COLORS[r]};"></span>{r}</span>'
    for r in RIVER_ORDER
)
st.markdown(
    f'<div class="legend-row">{legend_items}'
    f'<span class="legend-sep">|</span>'
    f'<span style="color:#86868B;">● 大圆 = 重点站 &nbsp; ● 小圆 = 普通站</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# 计算地图中心（排除明显异常的经纬度）
valid = [(s["lat"], s["lon"]) for s in stations
         if 28 < s.get("lat", 0) < 33 and 101 < s.get("lon", 0) < 106]
if valid:
    center_lat = sum(v[0] for v in valid) / len(valid)
    center_lon = sum(v[1] for v in valid) / len(valid)
else:
    center_lat, center_lon = 30.0, 103.8

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=8,
    tiles="CartoDB positron",
)

for s in stations:
    name   = s["name"]
    lat    = s.get("lat", 0)
    lon    = s.get("lon", 0)
    river  = s.get("river", "")
    is_key = s.get("is_key", False)
    latest_do = s.get("latest_do")
    quality   = s.get("main_quality_class", "")

    color  = RIVER_COLORS.get(river, "#8E8E93")
    radius = 10 if is_key else 6

    # 模型信息
    mr = model_results.get(name)
    r2_str = f"{mr['r2']:.4f}" if mr else "—"

    status, status_color = get_do_status(latest_do)
    do_str = f"{latest_do:.2f}" if latest_do is not None else "N/A"

    popup_html = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;min-width:210px;">
        <h4 style="margin:0 0 8px;color:#1D1D1F;">{name}</h4>
        <table style="font-size:13px;line-height:1.8;">
            <tr><td style="color:#86868B;width:72px;">河流</td>
                <td><b style="color:{color};">{river}</b></td></tr>
            <tr><td style="color:#86868B;">城市</td><td>{s.get('city','')}</td></tr>
            <tr><td style="color:#86868B;">最新 DO</td>
                <td><b>{do_str} mg/L</b></td></tr>
            <tr><td style="color:#86868B;">水质等级</td>
                <td>{QUALITY_LABELS.get(quality, quality)}</td></tr>
            <tr><td style="color:#86868B;">状态</td>
                <td><b style="color:{status_color};">{status}</b></td></tr>
            <tr><td style="color:#86868B;">模型 R²</td><td>{r2_str}</td></tr>
            <tr><td style="color:#86868B;">类型</td>
                <td>{'⭐ 重点站' if is_key else '普通站'}</td></tr>
        </table>
    </div>
    """

    folium.CircleMarker(
        location=[lat, lon],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        weight=2,
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=f"{name}（{river}）",
    ).add_to(m)

st_folium(m, width=None, height=520, use_container_width=True)


# ====================== 站点状态表 ======================
st.markdown("### 📊 站点状态总览")

# 构建表格数据，按河流排序
table_rows = []
for s in stations:
    name    = s["name"]
    river   = s.get("river", "")
    is_key  = s.get("is_key", False)
    latest_do = s.get("latest_do")
    quality = s.get("main_quality_class", "")

    mr = model_results.get(name)
    r2  = mr["r2"]  if mr else None
    mae = mr["mae"] if mr else None

    status, _ = get_do_status(latest_do)

    table_rows.append({
        "_river_order": RIVER_ORDER.index(river) if river in RIVER_ORDER else 99,
        "_station_order": s.get("order", 99),
        "站点":     f"⭐ {name}" if is_key else f"　{name}",
        "河流":     river,
        "城市":     s.get("city", ""),
        "最新DO":   f"{latest_do:.2f}" if latest_do is not None else "—",
        "水质等级": QUALITY_LABELS.get(quality, ""),
        "状态":     status,
        "模型 R²":  f"{r2:.4f}" if r2 is not None else "—",
        "模型 MAE": f"{mae:.4f}" if mae is not None else "—",
    })

df_table = pd.DataFrame(table_rows)
df_table = df_table.sort_values(["_river_order", "_station_order"])
df_table = df_table.drop(columns=["_river_order", "_station_order"])

st.dataframe(
    df_table,
    use_container_width=True,
    hide_index=True,
    height=560,
)


# ====================== 底部信息 ======================
st.markdown("---")
st.caption(
    "数据来源：中国环境监测总站 · 国家地表水水质自动监测实时数据发布系统 "
    "| 数据时间：2022.01 — 2025.02 "
    "| 👈 点击左侧菜单进入「站点分析」页面查看详细预测"
)
