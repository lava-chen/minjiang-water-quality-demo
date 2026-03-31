"""
1_📊_站点分析.py — 水质预报与智能分析

天气预报风格：当前状态 → 未来 24h 预报 → DO 趋势图 → 驱动因素 → 技术详情
数据来源：kg_data.json（知识图谱）+ forecasts.json（模型推理）+ test_predictions.csv（历史回测）
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ====================== 页面配置 ======================
st.set_page_config(page_title="站点分析", page_icon="📊", layout="wide")

# ====================== 路径 ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
KG_PATH = os.path.join(BASE_DIR, "src", "knowledge_graph", "kg_data.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FORECAST_PATH = os.path.join(MODELS_DIR, "forecasts.json")


# ====================== 数据加载（缓存） ======================
@st.cache_data
def load_kg():
    with open(KG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_forecasts():
    """加载预计算的预测结果"""
    if os.path.exists(FORECAST_PATH):
        with open(FORECAST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_predictions(station_name: str):
    """加载测试集历史回测数据"""
    path = os.path.join(MODELS_DIR, station_name, "test_predictions.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["时间"] = pd.to_datetime(df["时间"])
        return df
    return None


# ====================== 加载数据 ======================
kg = load_kg()
forecasts_data = load_forecasts()

stations = {n["name"]: n for n in kg["nodes"] if n["type"] == "Station"}
model_results = {n["station"]: n for n in kg["nodes"] if n["type"] == "ModelResult"}
feature_explanations = kg.get("feature_explanations", {})

# 重点站列表
key_station_names = sorted(
    [name for name, s in stations.items() if s.get("is_key")],
    key=lambda x: (stations[x].get("river", ""), stations[x].get("order", 0)),
)


# ====================== 全局样式 ======================
st.markdown("""
<style>
    footer {visibility: hidden;}

    /* ========== 侧边栏 · 深色主题 ========== */
    [data-testid="stSidebar"] {
        background: #1C1C1E !important;
    }
    [data-testid="stSidebar"] * {
        color: #F5F5F7 !important;
    }
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
    [data-testid="stSidebarNav"] span {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #F5F5F7 !important;
    }
    [data-testid="stSidebar"] label {
        color: #A1A1A6 !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] {
        background: #2C2C2E !important;
        border-radius: 10px !important;
        border: 1px solid #3A3A3C !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #F5F5F7 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #3A3A3C !important;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        color: #8E8E93 !important;
    }

    /* ========== 全局字号 ========== */
    html, body, [class*="css"] {
        font-size: 17px !important;
    }
    .stMarkdown, .stText, .stCaption, p, span, td, th, li {
        font-size: 1.05rem !important;
    }

    /* ========== 当前状态卡片 ========== */
    .status-hero {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 40%, #0f3460 100%);
        border-radius: 24px;
        padding: 2rem 2.5rem;
        color: #fff;
        position: relative;
        overflow: hidden;
        margin-bottom: 0.5rem;
    }
    .status-hero::after {
        content: '';
        position: absolute;
        top: -40px;
        right: -40px;
        width: 180px;
        height: 180px;
        border-radius: 50%;
        background: rgba(255,255,255,0.03);
    }
    .status-station {
        font-size: 1.3rem;
        font-weight: 600;
        color: #A1A1A6;
        margin-bottom: 4px;
    }
    .status-river {
        font-size: 0.95rem;
        color: #636366;
        margin-bottom: 1rem;
    }
    .status-do-row {
        display: flex;
        align-items: baseline;
        gap: 12px;
        margin-bottom: 4px;
    }
    .status-do-value {
        font-size: 4.5rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1;
    }
    .status-do-unit {
        font-size: 1.4rem;
        font-weight: 400;
        color: #8E8E93;
    }
    .status-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 12px;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 8px;
    }
    .status-time {
        font-size: 0.9rem;
        color: #636366;
        margin-top: 12px;
    }

    /* ========== 预报卡片区 ========== */
    .forecast-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1D1D1F;
        margin: 1.5rem 0 0.8rem;
    }
    .forecast-container {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 10px;
    }
    .forecast-card {
        background: #1C1C1E;
        border-radius: 18px;
        padding: 18px 10px;
        text-align: center;
        color: #fff;
        position: relative;
        transition: transform 0.15s;
    }
    .forecast-card:hover {
        transform: translateY(-2px);
    }
    .fc-label {
        font-size: 0.82rem;
        color: #8E8E93;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .fc-time {
        font-size: 0.82rem;
        color: #636366;
        margin-top: 2px;
    }
    .fc-do {
        font-size: 2rem;
        font-weight: 700;
        margin: 10px 0 4px;
        line-height: 1;
    }
    .fc-trend {
        font-size: 0.85rem;
        color: #8E8E93;
        margin-bottom: 8px;
    }
    .fc-quality {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #fff;
    }

    /* ========== 区块标题 ========== */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1D1D1F;
        margin: 1.5rem 0 0.3rem;
    }
    .section-desc {
        font-size: 0.95rem;
        color: #86868B;
        margin-bottom: 0.8rem;
    }

    /* ========== 驱动因素卡片 ========== */
    .driver-card {
        background: #FAFAFA;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .driver-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
    }
    .driver-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1D1D1F;
    }
    .driver-tag {
        font-size: 0.82rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 8px;
        color: #fff;
    }
    .driver-text {
        color: #3D3D3D;
        margin: 0.4rem 0 0;
        font-size: 0.98rem;
        line-height: 1.65;
    }

    /* ========== 技术详情卡片 ========== */
    .info-card {
        background: #F5F5F7;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
    }
    .info-title {
        font-weight: 600;
        color: #1D1D1F;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    .info-row {
        font-size: 0.95rem;
        color: #3D3D3D;
        line-height: 2;
    }
    .info-label {
        color: #86868B;
        display: inline-block;
        width: 88px;
    }

    /* ========== 水质标准线图例 ========== */
    .legend-inline {
        display: flex;
        gap: 14px;
        flex-wrap: wrap;
        font-size: 0.88rem;
        color: #636366;
        margin-top: 2px;
    }
    .legend-inline span {
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .legend-dot-sm {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ====================== 侧边栏 ======================
st.sidebar.markdown("### ⚙️ 分析设置")

selected_station = st.sidebar.selectbox(
    "选择站点",
    key_station_names,
    index=0,
    format_func=lambda x: f"⭐ {x}（{stations[x]['river']}）",
)

st.sidebar.markdown("---")
st.sidebar.caption("💡 选择站点后可查看实时预报、趋势分析和驱动因素")


# ====================== 获取数据 ======================
sd = stations[selected_station]
mr = model_results.get(selected_station)
fc = forecasts_data.get(selected_station, {})
pred_df = load_predictions(selected_station)


# ====================== 区块 1：当前水质状态（大卡片）======================
if fc and "error" not in fc:
    current_do = fc["current_do"]
    current_time = fc["current_time"]
    current_quality = fc["current_quality"]
    current_status = fc["current_status"]
    current_color = fc["current_color"]

    st.markdown(f"""
    <div class="status-hero">
        <div class="status-station">📍 {selected_station}</div>
        <div class="status-river">{sd.get('river','')} · {sd.get('city','')} · ({sd.get('lat',0):.4f}°N, {sd.get('lon',0):.4f}°E)</div>
        <div style="display: flex; align-items: flex-end; gap: 24px; flex-wrap: wrap;">
            <div>
                <div style="font-size: 0.85rem; color: #8E8E93; margin-bottom: 4px;">当前溶解氧</div>
                <div class="status-do-row">
                    <span class="status-do-value" style="color: {current_color};">{current_do}</span>
                    <span class="status-do-unit">mg/L</span>
                </div>
                <div class="status-badge" style="background: {current_color};">{current_quality}类水质 · {current_status}</div>
            </div>
            <div style="flex:1;"></div>
            <div style="text-align: right;">
                <div style="font-size: 0.85rem; color: #8E8E93;">模型精度 R²</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: #F5F5F7;">{mr['r2']:.2f}</div>
                <div style="font-size: 0.85rem; color: #636366;">MAE {mr['mae']:.3f} mg/L</div>
            </div>
        </div>
        <div class="status-time">🕐 最新数据时间：{current_time}</div>
    </div>
    """, unsafe_allow_html=True)
else:
    # 降级：无预测数据时显示简版
    st.markdown(f"## 📊 {selected_station}")
    st.markdown(f"*{sd.get('river','')} · {sd.get('city','')}*")
    if fc.get("error"):
        st.warning(f"预测数据加载失败：{fc['error']}")


# ====================== 区块 2：未来 24 小时预报卡片 ======================
if fc and "error" not in fc and fc.get("forecasts"):
    st.markdown('<div class="forecast-title">⏱ 未来 24 小时预报</div>', unsafe_allow_html=True)

    cards_html = '<div class="forecast-container">'
    for f_item in fc["forecasts"]:
        hours = f_item["hours_ahead"]
        cards_html += f"""
        <div class="forecast-card">
            <div class="fc-label">+{hours}h</div>
            <div class="fc-time">{f_item['time']}</div>
            <div class="fc-do" style="color: {f_item['color']};">{f_item['do']}</div>
            <div class="fc-trend">{f_item['trend']} {f_item['delta']:+.2f}</div>
            <div class="fc-quality" style="background: {f_item['color']};">{f_item['quality']}类 · {f_item['status']}</div>
        </div>
        """
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

st.markdown("")


# ====================== 区块 3：DO 趋势图（近 7 天 + 未来 24h）======================
if fc and "error" not in fc:
    st.markdown('<div class="section-title">📈 DO 趋势（近 7 天实测 + 未来 24h 预测）</div>', unsafe_allow_html=True)

    fig_trend = go.Figure()

    # —— 近 7 天实测 ——
    history = fc.get("recent_history", [])
    if history:
        hist_times = [h["time"] for h in history]
        hist_dos = [h["do"] for h in history]

        fig_trend.add_trace(go.Scatter(
            x=hist_times, y=hist_dos,
            mode="lines",
            name="实测 DO",
            line=dict(color="#007AFF", width=2.5),
            hovertemplate="%{x}<br>DO: %{y:.2f} mg/L<extra>实测</extra>",
        ))

    # —— "当前"标记点 ——
    fig_trend.add_trace(go.Scatter(
        x=[fc["current_time"]],
        y=[fc["current_do"]],
        mode="markers",
        name="当前",
        marker=dict(color=fc["current_color"], size=14, symbol="circle",
                    line=dict(color="#fff", width=2)),
        hovertemplate="当前: %{y:.2f} mg/L<extra></extra>",
    ))

    # —— 未来 24h 预测 ——
    forecast_times = [fc["current_time"]] + [f_item["time_full"] for f_item in fc["forecasts"]]
    forecast_dos = [fc["current_do"]] + [f_item["do"] for f_item in fc["forecasts"]]
    forecast_colors = [fc["current_color"]] + [f_item["color"] for f_item in fc["forecasts"]]

    fig_trend.add_trace(go.Scatter(
        x=forecast_times, y=forecast_dos,
        mode="lines+markers",
        name="预测 DO",
        line=dict(color="#FF9500", width=2.5, dash="dot"),
        marker=dict(color="#FF9500", size=8),
        hovertemplate="%{x}<br>预测DO: %{y:.2f} mg/L<extra>预测</extra>",
    ))

    # —— "现在"垂直线 ——
    fig_trend.add_vline(
        x=fc["current_time"],
        line_dash="dash",
        line_color="#636366",
        line_width=1,
        annotation_text="现在",
        annotation_position="top",
        annotation_font_color="#636366",
    )

    # —— 水质标准区域 ——
    y_min = min(hist_dos + forecast_dos) - 1 if history else 5
    y_max = max(hist_dos + forecast_dos) + 1.5 if history else 15

    for val, label, clr in [
        (7.5, "Ⅰ类 (7.5)", "#34C759"),
        (6.0, "Ⅱ类 (6.0)", "#007AFF"),
        (5.0, "Ⅲ类 (5.0)", "#FF3B30"),
    ]:
        if y_min - 1 < val < y_max + 1:
            fig_trend.add_hline(
                y=val, line_dash="dash", line_color=clr,
                annotation_text=label,
                annotation_position="top left",
                annotation_font_size=11,
                opacity=0.4,
            )

    fig_trend.update_layout(
        hovermode="x unified",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=40, b=50),
        yaxis_title="溶解氧 (mg/L)",
        yaxis_range=[max(y_min, 0), y_max],
        xaxis_title="",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # 图例说明
    st.markdown("""
    <div class="legend-inline">
        <span><span class="legend-dot-sm" style="background:#007AFF;"></span>蓝色实线 = 近7天实测</span>
        <span><span class="legend-dot-sm" style="background:#FF9500;"></span>橙色虚线 = 未来24h预测</span>
        <span>|</span>
        <span>虚线标准：<span style="color:#34C759;">绿=Ⅰ类</span> / <span style="color:#007AFF;">蓝=Ⅱ类</span> / <span style="color:#FF3B30;">红=Ⅲ类</span></span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ====================== 区块 4：驱动因素 TOP 5 ======================
st.markdown('<div class="section-title">🔬 为什么这样预测？— 驱动因素 TOP 5</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">基于排列重要性分析：打乱某个特征后模型误差增大越多，说明该特征对 DO 预测越关键</div>',
    unsafe_allow_html=True,
)

if mr and mr.get("top_features"):
    top5 = mr["top_features"]

    # 柱状图
    names  = [f["name"] for f in top5]
    values = [f["importance"] for f in top5]

    bar_colors = []
    for v in values:
        if abs(v) > 0.2:
            bar_colors.append("#FF3B30")
        elif abs(v) > 0.05:
            bar_colors.append("#FF9500")
        else:
            bar_colors.append("#007AFF")

    fig_imp = go.Figure()
    fig_imp.add_trace(go.Bar(
        y=names[::-1],
        x=values[::-1],
        orientation="h",
        marker_color=bar_colors[::-1],
        text=[f"{v:+.4f}" for v in values[::-1]],
        textposition="outside",
    ))
    fig_imp.update_layout(
        template="plotly_white",
        height=240,
        margin=dict(l=130, r=80, t=10, b=30),
        xaxis_title="重要性（MSE 增量）",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # 解释卡片
    for i, feat in enumerate(top5):
        imp_val = feat["importance"]
        explanation = feat.get("explanation", "")
        if not explanation:
            explanation = feature_explanations.get(feat["name"], "暂无领域解释。")

        if abs(imp_val) > 0.2:
            icon, tag_text, tag_bg, border = "🔴", "高影响", "#FF3B30", "#FF3B30"
        elif abs(imp_val) > 0.05:
            icon, tag_text, tag_bg, border = "🟡", "中等影响", "#FF9500", "#FF9500"
        else:
            icon, tag_text, tag_bg, border = "🔵", "轻微影响", "#007AFF", "#007AFF"

        direction = "正向驱动 ↑" if imp_val > 0 else "负向抑制 ↓"

        st.markdown(f"""
        <div class="driver-card" style="border-left: 4px solid {border};">
            <div class="driver-header">
                <span class="driver-name">{icon} #{i+1} {feat['name']}</span>
                <span class="driver-tag" style="background: {tag_bg};">
                    {tag_text} · {direction} · {imp_val:+.4f}
                </span>
            </div>
            <p class="driver-text">{explanation}</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("该站点暂无特征重要性数据。")

st.markdown("---")


# ====================== 区块 5：技术详情（可折叠）======================
with st.expander("📋 技术详情（模型精度 · 注意力分布 · 历史回测 · 站点信息）", expanded=False):

    # ---- 模型精度指标 ----
    st.markdown("#### 🎯 模型精度（测试集）")
    if mr and mr.get("step_metrics"):
        metrics_data = []
        for sm in mr["step_metrics"]:
            metrics_data.append({
                "预测步长": f"+{sm['hours_ahead']}h",
                "MAE (mg/L)": f"{sm['MAE']:.4f}",
                "RMSE (mg/L)": f"{sm['RMSE']:.4f}",
                "R²": f"{sm['R2']:.4f}",
                "MAPE (%)": f"{sm['MAPE']:.2f}",
            })
        st.dataframe(
            pd.DataFrame(metrics_data),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # ---- 注意力分布 ----
    st.markdown("#### 🎯 时间注意力分布")
    st.caption("模型对过去 72 小时（18 个时间步）各时刻的关注程度 · 权重越高 = 该时刻对预测越关键")

    if mr and mr.get("attention_weights"):
        attn = mr["attention_weights"]
        attn_labels = list(attn.keys())
        attn_values = list(attn.values())

        max_attn = max(attn_values) if attn_values else 1
        attn_colors = [
            f"rgba(0,122,255,{0.3 + 0.7 * (v / max_attn)})" for v in attn_values
        ]

        fig_attn = go.Figure()
        fig_attn.add_trace(go.Bar(
            x=attn_labels, y=attn_values,
            marker_color=attn_colors,
            text=[f"{v:.3f}" for v in attn_values],
            textposition="outside",
            textfont=dict(size=9),
        ))
        fig_attn.update_layout(
            template="plotly_white",
            height=280,
            margin=dict(l=50, r=20, t=10, b=70),
            xaxis_title="时间步",
            yaxis_title="注意力权重",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_attn, use_container_width=True)

        st.info(
            "💡 **解读：** 模型赋予 **最近时间步（t-1，即 4 小时前）** 最高权重，"
            "权重随时间距离增大而逐步衰减。这符合河流水质的物理特性——近期水质状态"
            "对短期预测最有参考价值，远期信息的影响逐渐减弱。"
        )

    st.markdown("---")

    # ---- 历史回测曲线 ----
    st.markdown("#### 📈 历史回测曲线（测试集）")
    test_period = mr.get("test_period", "N/A") if mr else "N/A"
    st.caption(f"测试区间：{test_period}")

    if pred_df is not None:
        step_sel = st.radio(
            "选择预测步长",
            [4, 8, 12, 16, 20, 24],
            format_func=lambda x: f"+{x}h",
            horizontal=True,
            key="backtest_step",
        )

        actual_col = f"实际DO_{step_sel}h"
        pred_col = f"预测DO_{step_sel}h"

        if actual_col in pred_df.columns and pred_col in pred_df.columns:
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=pred_df["时间"], y=pred_df[actual_col],
                mode="lines", name="实测 DO",
                line=dict(color="#007AFF", width=2),
            ))
            fig_bt.add_trace(go.Scatter(
                x=pred_df["时间"], y=pred_df[pred_col],
                mode="lines", name=f"预测 DO（+{step_sel}h）",
                line=dict(color="#FF9500", width=2, dash="dot"),
            ))

            for val, label, clr in [
                (7.5, "Ⅰ类", "#34C759"),
                (6.0, "Ⅱ类", "#007AFF"),
                (5.0, "Ⅲ类", "#FF3B30"),
            ]:
                fig_bt.add_hline(y=val, line_dash="dash", line_color=clr,
                                 annotation_text=label, annotation_position="top left",
                                 opacity=0.35)

            fig_bt.update_layout(
                hovermode="x unified",
                template="plotly_white",
                height=360,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=50, r=20, t=30, b=40),
                xaxis_title="时间",
                yaxis_title="溶解氧 (mg/L)",
            )
            st.plotly_chart(fig_bt, use_container_width=True)

    st.markdown("---")

    # ---- 站点与模型信息 ----
    st.markdown("#### 📋 站点与模型信息")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        geo_rows = "".join([
            f'<div class="info-row"><span class="info-label">河流</span>{sd.get("river","")}</div>',
            f'<div class="info-row"><span class="info-label">城市</span>{sd.get("city","")}</div>',
            f'<div class="info-row"><span class="info-label">经度</span>{sd.get("lon",0):.6f}</div>',
            f'<div class="info-row"><span class="info-label">纬度</span>{sd.get("lat",0):.6f}</div>',
            f'<div class="info-row"><span class="info-label">类型</span>'
            f'{"⭐ 重点建模站" if sd.get("is_key") else "普通监测站"}</div>',
        ])
        st.markdown(f'<div class="info-card"><div class="info-title">🌍 地理信息</div>{geo_rows}</div>',
                    unsafe_allow_html=True)

    with col_b:
        quality_label = {"I": "Ⅰ", "II": "Ⅱ", "III": "Ⅲ", "IV": "Ⅳ", "V": "Ⅴ"}.get(
            sd.get("main_quality_class", ""), sd.get("main_quality_class", ""))
        data_rows = "".join([
            f'<div class="info-row"><span class="info-label">数据条数</span>{sd.get("data_count",0):,}</div>',
            f'<div class="info-row"><span class="info-label">时间范围</span>{sd.get("data_period","N/A")}</div>',
            f'<div class="info-row"><span class="info-label">最新 DO</span>{sd.get("latest_do","N/A")} mg/L</div>',
            f'<div class="info-row"><span class="info-label">主要水质</span>{quality_label} 类</div>',
        ])
        st.markdown(f'<div class="info-card"><div class="info-title">📊 数据概况</div>{data_rows}</div>',
                    unsafe_allow_html=True)

    with col_c:
        if mr:
            hp = mr.get("hyperparams", {})
            model_rows = "".join([
                f'<div class="info-row"><span class="info-label">模型</span>LSTM + Attention</div>',
                f'<div class="info-row"><span class="info-label">隐藏层</span>{hp.get("hidden_size","N/A")}</div>',
                f'<div class="info-row"><span class="info-label">学习率</span>{hp.get("lr","N/A")}</div>',
                f'<div class="info-row"><span class="info-label">损失函数</span>{hp.get("loss_function","N/A")}</div>',
                f'<div class="info-row"><span class="info-label">训练轮数</span>'
                f'{mr.get("best_epoch",0)} / {mr.get("total_epochs",0)}</div>',
                f'<div class="info-row"><span class="info-label">训练耗时</span>'
                f'{mr.get("train_time_sec",0):.1f} 秒</div>',
            ])
            st.markdown(
                f'<div class="info-card"><div class="info-title">🤖 模型信息</div>{model_rows}</div>',
                unsafe_allow_html=True,
            )


# ====================== 底部 ======================
st.markdown("---")
st.caption(
    "数据来源：中国环境监测总站 · 国家地表水水质自动监测实时数据发布系统 "
    "| 模型：LSTM + Attention | 目标：溶解氧 (DO) | 预测范围：4~24 小时"
)
