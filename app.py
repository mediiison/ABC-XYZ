import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from abc_xyz import RECOMMENDATIONS, analyze, load_sales, segment_matrix

DEFAULT_PATH = Path(__file__).parent / "data" / "sales.csv"
ABC_COLORS = {"A": "#1b6ca8", "B": "#f2a541", "C": "#9aa5b1"}
PERIODS = {"Неделя": "W", "Месяц": "M"}

st.set_page_config(page_title="ABC/XYZ анализ ассортимента", layout="wide")


@st.cache_data
def read_default(path):
    return load_sales(path)


@st.cache_data
def read_upload(data):
    return load_sales(io.BytesIO(data))


st.title("ABC/XYZ анализ ассортимента")

with st.sidebar:
    st.header("Данные")
    upload = st.file_uploader("CSV с продажами", type="csv")
    st.caption("Колонки: date, article, qty, revenue, category (необязательно)")

if upload is not None:
    sales = read_upload(upload.getvalue())
elif DEFAULT_PATH.exists():
    sales = read_default(str(DEFAULT_PATH))
else:
    st.info("Загрузите CSV с продажами или положите файл в data/sales.csv")
    st.stop()

with st.sidebar:
    st.header("Фильтры")
    d_min, d_max = sales["date"].min().date(), sales["date"].max().date()
    date_range = st.date_input("Период", value=(d_min, d_max), min_value=d_min, max_value=d_max)
    categories = sorted(sales["category"].unique())
    chosen = st.multiselect("Категории", categories, default=categories)

    st.header("Параметры")
    period_label = st.radio("Шаг для XYZ", list(PERIODS), index=1, horizontal=True)
    abc_a = st.slider("Граница A, % выручки", 50, 95, 80)
    abc_b = st.slider("Граница B, % выручки", abc_a + 1, 99, max(95, abc_a + 1))
    xyz_x = st.slider("Граница X, коэффициент вариации", 0.05, 0.5, 0.25, 0.05)
    xyz_y = st.slider("Граница Y, коэффициент вариации", round(xyz_x + 0.05, 2), 1.5, max(0.5, round(xyz_x + 0.05, 2)), 0.05)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
else:
    start, end = d_min, d_max

mask = (
    (sales["date"].dt.date >= start)
    & (sales["date"].dt.date <= end)
    & (sales["category"].isin(chosen))
)
filtered = sales[mask]

if filtered.empty:
    st.warning("Нет данных для выбранных фильтров")
    st.stop()

try:
    result = analyze(filtered, PERIODS[period_label], abc_a / 100, abc_b / 100, xyz_x, xyz_y)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

total_revenue = result["revenue"].sum()
a_share = result.loc[result["abc"] == "A", "revenue"].sum() / total_revenue if total_revenue else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Артикулов", f"{len(result)}")
k2.metric("Выручка", f"{total_revenue:,.0f}".replace(",", " "))
k3.metric("Доля выручки класса A", f"{a_share * 100:.1f}%")
k4.metric("Артикулов класса Z", f"{(result['xyz'] == 'Z').sum()}")

tab_matrix, tab_pareto, tab_demand, tab_table = st.tabs(["Матрица", "Парето", "Спрос", "Артикулы"])

with tab_matrix:
    left, right = st.columns(2)
    counts = segment_matrix(result, "count")
    shares = segment_matrix(result, "revenue") * 100
    fig_c = px.imshow(
        counts,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="XYZ", y="ABC", color="Артикулов"),
        title="Число артикулов",
    )
    fig_r = px.imshow(
        shares,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="Oranges",
        labels=dict(x="XYZ", y="ABC", color="% выручки"),
        title="Доля выручки, %",
    )
    left.plotly_chart(fig_c, width="stretch")
    right.plotly_chart(fig_r, width="stretch")

    st.subheader("Что делать с сегментами")
    seg_counts = result["segment"].value_counts()
    guide = pd.DataFrame(
        {
            "Сегмент": list(RECOMMENDATIONS),
            "Артикулов": [int(seg_counts.get(s, 0)) for s in RECOMMENDATIONS],
            "Рекомендация": list(RECOMMENDATIONS.values()),
        }
    )
    st.dataframe(guide, hide_index=True, width="stretch")

with tab_pareto:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for cls, color in ABC_COLORS.items():
        part = result[result["abc"] == cls]
        fig.add_bar(x=part["article"], y=part["revenue"], name=f"Класс {cls}", marker_color=color)
    fig.add_trace(
        go.Scatter(
            x=result["article"],
            y=result["cum_share"] * 100,
            name="Накопленная доля, %",
            mode="lines",
            line=dict(color="#444444"),
        ),
        secondary_y=True,
    )
    fig.update_xaxes(categoryorder="array", categoryarray=result["article"].tolist(), showticklabels=False)
    fig.update_yaxes(title_text="Выручка", secondary_y=False)
    fig.update_yaxes(title_text="Накопленная доля, %", range=[0, 100], secondary_y=True)
    fig.update_layout(barmode="stack", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

with tab_demand:
    scatter = px.scatter(
        result,
        x="cv",
        y="revenue",
        color="abc",
        color_discrete_map=ABC_COLORS,
        hover_name="article",
        hover_data={"category": True, "segment": True, "cv": ":.2f", "revenue": ":,.0f"},
        log_y=True,
        labels={"cv": "Коэффициент вариации спроса", "revenue": "Выручка", "abc": "Класс"},
        title="Выручка и стабильность спроса",
    )
    scatter.add_vline(x=xyz_x, line_dash="dash", line_color="gray")
    scatter.add_vline(x=xyz_y, line_dash="dash", line_color="gray")
    st.plotly_chart(scatter, width="stretch")

    classes = result[["article", "abc"]]
    monthly = (
        filtered.merge(classes, on="article")
        .assign(month=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["month", "abc"], as_index=False)["revenue"]
        .sum()
    )
    bars = px.bar(
        monthly,
        x="month",
        y="revenue",
        color="abc",
        color_discrete_map=ABC_COLORS,
        category_orders={"abc": ["A", "B", "C"]},
        labels={"month": "Месяц", "revenue": "Выручка", "abc": "Класс"},
        title="Выручка по месяцам и классам ABC",
    )
    st.plotly_chart(bars, width="stretch")

with tab_table:
    segments = st.multiselect("Сегменты", list(RECOMMENDATIONS), default=list(RECOMMENDATIONS))
    table = result[result["segment"].isin(segments)].copy()
    table["share"] = (table["share"] * 100).round(2)
    table["cum_share"] = (table["cum_share"] * 100).round(2)
    table["cv"] = table["cv"].round(3)
    table["mean_qty"] = table["mean_qty"].round(1)
    table = table[
        ["article", "category", "revenue", "share", "cum_share", "abc", "mean_qty", "cv", "xyz", "segment", "recommendation"]
    ].rename(
        columns={
            "article": "Артикул",
            "category": "Категория",
            "revenue": "Выручка",
            "share": "Доля, %",
            "cum_share": "Накопл. доля, %",
            "abc": "ABC",
            "mean_qty": "Средний спрос за период",
            "cv": "Коэф. вариации",
            "xyz": "XYZ",
            "segment": "Сегмент",
            "recommendation": "Рекомендация",
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button(
        "Скачать CSV",
        table.to_csv(index=False).encode("utf-8-sig"),
        file_name="abc_xyz_result.csv",
        mime="text/csv",
    )
