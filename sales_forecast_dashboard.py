"""
Sales Forecast Dashboard
-------------------------
A Streamlit app for uploading sales CSV data, exploring it with interactive
charts, and forecasting future sales using ARIMA.

Run with:
    streamlit run sales_forecast_dashboard.py

Requirements (install first):
    pip install streamlit pandas numpy plotly statsmodels scikit-learn
"""

import io
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLE
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Overall background */
    .stApp {
        background: linear-gradient(180deg, #0f1116 0%, #131722 100%);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1f2e 0%, #1f2937 100%);
        border: 1px solid #2d3548;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetricLabel"] { color: #9ca3af !important; }
    div[data-testid="stMetricValue"] { color: #f3f4f6 !important; }

    /* Headings */
    h1, h2, h3 { color: #f3f4f6 !important; }
    h1 { font-weight: 800 !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #11141d;
        border-right: 1px solid #232838;
    }

    /* Cards / containers */
    .dash-card {
        background: #161a26;
        border: 1px solid #232838;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }

    .accent-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid #6366f1;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        box-shadow: 0 0 14px rgba(139,92,246,0.5);
        transform: translateY(-1px);
    }

    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT = "#8b5cf6"
ACCENT2 = "#22d3ee"
ACCENT3 = "#34d399"


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def guess_date_column(df: pd.DataFrame):
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            return col
    return df.columns[0]


def guess_value_column(df: pd.DataFrame, date_col: str):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        if col != date_col and any(k in col.lower() for k in ["sale", "revenue", "amount", "qty", "quantity", "total"]):
            return col
    return numeric_cols[0] if numeric_cols else None


def make_demo_data() -> pd.DataFrame:
    rng = pd.date_range("2021-01-01", periods=48, freq="MS")
    rs = np.random.RandomState(42)
    trend = np.linspace(8000, 18000, len(rng))
    season = 2500 * np.sin(np.linspace(0, 8 * np.pi, len(rng)))
    noise = rs.normal(0, 700, len(rng))
    sales = trend + season + noise
    return pd.DataFrame({
        "Date": rng,
        "Sales": sales.round(2),
        "Region": rs.choice(["North", "South", "East", "West"], len(rng)),
        "Product": rs.choice(["Widget A", "Widget B", "Widget C"], len(rng)),
    })


def aggregate_series(df: pd.DataFrame, date_col: str, value_col: str, freq: str) -> pd.Series:
    s = df[[date_col, value_col]].dropna()
    s = s.set_index(date_col).resample(freq)[value_col].sum()
    s = s.asfreq(freq).ffill()
    return s


def check_stationarity(series: pd.Series) -> dict:
    try:
        result = adfuller(series.dropna())
        return {"adf_stat": result[0], "p_value": result[1], "stationary": result[1] < 0.05}
    except Exception:
        return {"adf_stat": np.nan, "p_value": np.nan, "stationary": None}


@st.cache_data(show_spinner=False)
def run_arima_forecast(series_values, series_index_iso, order, steps, freq):
    series = pd.Series(series_values, index=pd.to_datetime(list(series_index_iso)))
    model = ARIMA(series, order=order)
    fitted = model.fit()
    forecast_res = fitted.get_forecast(steps=steps)
    mean_forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.2)  # 80% CI

    last_date = series.index[-1]
    future_index = pd.date_range(last_date, periods=steps + 1, freq=freq)[1:]
    mean_forecast.index = future_index
    conf_int.index = future_index

    return fitted, mean_forecast, conf_int


def kpi_card(label, value, delta=None):
    st.metric(label, value, delta)


# --------------------------------------------------------------------------
# SIDEBAR — DATA INPUT
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 Data Source")
    uploaded_file = st.file_uploader("Upload sales CSV", type=["csv"])
    use_demo = st.checkbox("Use demo dataset instead", value=uploaded_file is None)

    st.markdown("---")
    st.markdown("### ⚙️ Forecast Settings")
    freq_label = st.selectbox("Aggregation period", ["Daily", "Weekly", "Monthly"], index=2)
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}
    freq = freq_map[freq_label]

    horizon = st.slider("Forecast horizon (periods ahead)", 3, 36, 12)

    st.markdown("**ARIMA order (p, d, q)**")
    c1, c2, c3 = st.columns(3)
    p = c1.number_input("p", 0, 5, 2)
    d = c2.number_input("d", 0, 2, 1)
    q = c3.number_input("q", 0, 5, 2)

    auto_order = st.checkbox("Auto-pick a simple order if ARIMA fails", value=True)

    st.markdown("---")
    st.caption("Tip: CSV needs at least one date column and one numeric sales column.")


# --------------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------------
if uploaded_file is not None and not use_demo:
    raw_df = load_csv(uploaded_file.getvalue())
    source_label = uploaded_file.name
elif use_demo:
    raw_df = make_demo_data()
    source_label = "Demo dataset"
else:
    raw_df = None
    source_label = None

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:14px; margin-bottom: 6px;">
        <div style="font-size:38px;">📈</div>
        <div>
            <h1 style="margin:0;">Sales Forecast Dashboard</h1>
            <p style="color:#9ca3af; margin:0;">Upload your data, explore trends, and forecast what's next with ARIMA.</p>
            <p style="color:#6366f1; margin:0; font-size:13px; font-weight:600;">by Rizwana Masood</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if raw_df is None:
    st.info("👈 Upload a CSV from the sidebar, or check **'Use demo dataset'** to explore the dashboard.")
    st.stop()

st.markdown(f'<span class="accent-badge">Source: {source_label}</span>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# COLUMN MAPPING
# --------------------------------------------------------------------------
with st.expander("🔧 Column mapping (auto-detected — adjust if needed)", expanded=False):
    cols = raw_df.columns.tolist()
    date_col = st.selectbox("Date column", cols, index=cols.index(guess_date_column(raw_df)))
    numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
    default_val_col = guess_value_column(raw_df, date_col)
    value_col = st.selectbox(
        "Sales / value column",
        numeric_cols if numeric_cols else cols,
        index=numeric_cols.index(default_val_col) if default_val_col in numeric_cols else 0,
    )
    cat_cols = [c for c in cols if c not in [date_col, value_col] and raw_df[c].dtype == object]
    segment_col = st.selectbox("Optional segment column (for breakdown chart)", ["None"] + cat_cols)

# Clean & parse
df = raw_df.copy()
try:
    df[date_col] = pd.to_datetime(df[date_col])
except Exception:
    st.error(f"Couldn't parse '{date_col}' as dates. Please pick a valid date column.")
    st.stop()

df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)

if df.empty:
    st.error("No valid rows after cleaning. Check your date/value column selection.")
    st.stop()

# --------------------------------------------------------------------------
# KPI ROW
# --------------------------------------------------------------------------
total_sales = df[value_col].sum()
avg_sales = df[value_col].mean()
date_span = (df[date_col].max() - df[date_col].min()).days
last_30 = df[df[date_col] >= df[date_col].max() - timedelta(days=30)][value_col].sum()
growth = None
series_for_growth = df.set_index(date_col)[value_col].resample("MS").sum()
if len(series_for_growth) >= 2:
    growth = (series_for_growth.iloc[-1] / series_for_growth.iloc[-2] - 1) * 100 if series_for_growth.iloc[-2] != 0 else None

k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi_card("💰 Total Sales", f"{total_sales:,.0f}")
with k2:
    kpi_card("📊 Average per Record", f"{avg_sales:,.2f}")
with k3:
    kpi_card("🗓️ Date Range (days)", f"{date_span:,}")
with k4:
    kpi_card("📈 MoM Growth", f"{growth:,.1f}%" if growth is not None else "N/A")

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------------
tab_overview, tab_explore, tab_forecast, tab_data = st.tabs(
    ["🏠 Overview", "🔍 Explore", "🔮 Forecast", "🗂️ Raw Data"]
)

# ---- OVERVIEW TAB ----
with tab_overview:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        head_l, head_r = st.columns([3, 2])
        with head_l:
            st.subheader("Sales Over Time")
        with head_r:
            overview_freq_label = st.selectbox(
                "View by", ["Daily", "Weekly", "Monthly", "Quarterly"],
                index=2, key="overview_freq", label_visibility="collapsed",
            )
        overview_freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "MS", "Quarterly": "QS"}
        ts = df.set_index(date_col)[value_col].resample(overview_freq_map[overview_freq_label]).sum().reset_index()
        fig = px.area(ts, x=date_col, y=value_col, template=PLOTLY_TEMPLATE)
        fig.update_traces(line_color=ACCENT, fillcolor="rgba(139,92,246,0.25)")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Showing totals aggregated by **{overview_freq_label.lower()}** period — switch above for more or less detail.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.subheader("Distribution")
        fig2 = px.histogram(df, x=value_col, nbins=30, template=PLOTLY_TEMPLATE, color_discrete_sequence=[ACCENT2])
        fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if segment_col != "None":
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.subheader(f"Breakdown by {segment_col}")
        seg = df.groupby(segment_col)[value_col].sum().reset_index().sort_values(value_col, ascending=False)
        fig3 = px.bar(seg, x=segment_col, y=value_col, template=PLOTLY_TEMPLATE, color=segment_col,
                       color_discrete_sequence=px.colors.qualitative.Bold)
        fig3.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---- EXPLORE TAB ----
with tab_explore:
    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    st.subheader("Custom Trend Explorer")
    c1, c2 = st.columns(2)
    with c1:
        roll_window = st.slider("Rolling average window", 1, 30, 7)
    with c2:
        date_min, date_max = df[date_col].min(), df[date_col].max()
        date_range = st.date_input("Filter date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)

    filtered = df.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        filtered = filtered[(filtered[date_col] >= pd.Timestamp(date_range[0])) & (filtered[date_col] <= pd.Timestamp(date_range[1]))]

    ts2 = filtered.groupby(date_col)[value_col].sum().reset_index()
    ts2["Rolling Avg"] = ts2[value_col].rolling(roll_window, min_periods=1).mean()

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=ts2[date_col], y=ts2[value_col], mode="lines", name="Actual",
                               line=dict(color=ACCENT2, width=1.5)))
    fig4.add_trace(go.Scatter(x=ts2[date_col], y=ts2["Rolling Avg"], mode="lines", name=f"{roll_window}-period Rolling Avg",
                               line=dict(color=ACCENT, width=3)))
    fig4.update_layout(template=PLOTLY_TEMPLATE, height=420, margin=dict(l=10, r=10, t=30, b=10),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if segment_col != "None":
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        seg_head_l, seg_head_r = st.columns([3, 2])
        with seg_head_l:
            st.subheader(f"Trend by {segment_col}")
        with seg_head_r:
            seg_freq_label = st.selectbox(
                "View by", ["Daily", "Weekly", "Monthly", "Quarterly"],
                index=2, key="segment_freq", label_visibility="collapsed",
            )
        seg_freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "MS", "Quarterly": "QS"}
        ts3 = (
            filtered.set_index(date_col)
            .groupby(segment_col)[value_col]
            .resample(seg_freq_map[seg_freq_label])
            .sum()
            .reset_index()
        )
        fig5 = px.line(ts3, x=date_col, y=value_col, color=segment_col, template=PLOTLY_TEMPLATE,
                        color_discrete_sequence=px.colors.qualitative.Bold)
        fig5.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---- FORECAST TAB ----
with tab_forecast:
    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    st.subheader("🔮 ARIMA Forecast")

    series = aggregate_series(df, date_col, value_col, freq)

    if len(series) < 8:
        st.warning("Need at least ~8 periods of aggregated data for a meaningful ARIMA forecast. Try a finer aggregation (e.g. Daily/Weekly) or upload more data.")
    else:
        stat_check = check_stationarity(series)
        sc1, sc2 = st.columns(2)
        with sc1:
            st.caption(f"ADF p-value: **{stat_check['p_value']:.4f}**" if not np.isnan(stat_check['p_value']) else "ADF test unavailable")
        with sc2:
            if stat_check["stationary"] is True:
                st.caption("✅ Series looks stationary")
            elif stat_check["stationary"] is False:
                st.caption("ℹ️ Series is non-stationary — differencing (d≥1) recommended")

        order = (int(p), int(d), int(q))
        fitted_model, mean_forecast, conf_int = None, None, None
        series_index_iso = tuple(series.index.strftime("%Y-%m-%d"))
        try:
            fitted_model, mean_forecast, conf_int = run_arima_forecast(
                series.values, series_index_iso, order, horizon, freq
            )
        except Exception as e:
            if auto_order:
                try:
                    fallback_order = (1, 1, 1)
                    fitted_model, mean_forecast, conf_int = run_arima_forecast(
                        series.values, series_index_iso, fallback_order, horizon, freq
                    )
                    st.info(f"Order {order} failed ({e}). Used fallback order {fallback_order} instead.")
                    order = fallback_order
                except Exception as e2:
                    st.error(f"ARIMA fitting failed even with fallback order: {e2}")
            else:
                st.error(f"ARIMA fitting failed with order {order}: {e}")

        if mean_forecast is not None:
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name="Historical",
                                       line=dict(color=ACCENT2, width=2)))
            fig6.add_trace(go.Scatter(x=mean_forecast.index, y=mean_forecast.values, mode="lines+markers",
                                       name="Forecast", line=dict(color=ACCENT3, width=3, dash="dash")))
            fig6.add_trace(go.Scatter(
                x=list(conf_int.index) + list(conf_int.index[::-1]),
                y=list(conf_int.iloc[:, 1]) + list(conf_int.iloc[:, 0][::-1]),
                fill="toself", fillcolor="rgba(52,211,153,0.15)", line=dict(color="rgba(0,0,0,0)"),
                name="80% Confidence Interval", showlegend=True,
            ))
            fig6.update_layout(template=PLOTLY_TEMPLATE, height=460, margin=dict(l=10, r=10, t=30, b=10),
                                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                                title=f"ARIMA{order} Forecast — next {horizon} {freq_label.lower()} periods")
            st.plotly_chart(fig6, use_container_width=True)

            st.markdown("#### Forecast Summary")
            f1, f2, f3 = st.columns(3)
            with f1:
                kpi_card("Next period forecast", f"{mean_forecast.iloc[0]:,.0f}")
            with f2:
                kpi_card(f"Total forecast ({horizon}p)", f"{mean_forecast.sum():,.0f}")
            with f3:
                kpi_card("AIC (model fit)", f"{fitted_model.aic:,.1f}")

            forecast_df = pd.DataFrame({
                "Date": mean_forecast.index,
                "Forecast": mean_forecast.values,
                "Lower 80%": conf_int.iloc[:, 0].values,
                "Upper 80%": conf_int.iloc[:, 1].values,
            })
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)

            csv_bytes = forecast_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download forecast as CSV", csv_bytes, "sales_forecast.csv", "text/csv")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- RAW DATA TAB ----
with tab_data:
    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    st.subheader("Raw Data")
    st.dataframe(df, use_container_width=True, height=480)
    st.caption(f"{len(df):,} rows × {len(df.columns)} columns")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><hr style='border-color:#232838;'><br>", unsafe_allow_html=True)
st.caption("Built with Streamlit, Plotly & statsmodels ARIMA · Sales Forecast Dashboard")
