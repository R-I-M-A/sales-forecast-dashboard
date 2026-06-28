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


@st.cache_data(show_spinner=False)
def backtest_arima(series_values, series_index_iso, order, holdout_size, freq):
    """
    Train ARIMA on all but the last `holdout_size` periods, forecast those
    held-out periods, and compare against actuals to compute error metrics.
    Returns (actual_holdout, predicted_holdout, metrics_dict).
    """
    full_index = pd.to_datetime(list(series_index_iso))
    full_series = pd.Series(series_values, index=full_index)

    train = full_series.iloc[:-holdout_size]
    test = full_series.iloc[-holdout_size:]

    model = ARIMA(train, order=order)
    fitted = model.fit()
    forecast_res = fitted.get_forecast(steps=holdout_size)
    predicted = forecast_res.predicted_mean
    predicted.index = test.index  # align exactly with the holdout dates

    errors = test.values - predicted.values
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    nonzero_mask = test.values != 0
    mape = np.mean(np.abs(errors[nonzero_mask] / test.values[nonzero_mask])) * 100 if nonzero_mask.any() else np.nan

    metrics = {"mae": mae, "rmse": rmse, "mape": mape}
    return test, predicted, metrics


def kpi_card(label, value, delta=None):
    st.metric(label, value, delta)


# --------------------------------------------------------------------------
# RULE-BASED INSIGHTS ENGINE (no API key needed)
# --------------------------------------------------------------------------
def _fmt(n):
    """Format a number compactly for natural-language sentences."""
    if pd.isna(n):
        return "N/A"
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:,.0f}"


def explain_trend_chart(ts: pd.DataFrame, date_col: str, value_col: str, freq_label: str) -> str:
    """Generate a plain-English summary of a time-series trend chart."""
    if len(ts) < 2:
        return "Not enough data points yet to describe a trend."

    vals = ts[value_col].values
    first, last = vals[0], vals[-1]
    change_pct = ((last - first) / first * 100) if first != 0 else None
    peak_idx = int(np.argmax(vals))
    trough_idx = int(np.argmin(vals))
    peak_date = ts[date_col].iloc[peak_idx]
    trough_date = ts[date_col].iloc[trough_idx]

    direction = "risen" if (change_pct or 0) > 2 else ("fallen" if (change_pct or 0) < -2 else "stayed roughly flat")

    sentence = f"Over this {freq_label.lower()} view, sales have {direction}"
    if change_pct is not None and abs(change_pct) > 2:
        sentence += f" by about **{abs(change_pct):.0f}%**"
    sentence += f", from **{_fmt(first)}** to **{_fmt(last)}**. "
    sentence += f"The highest point was **{_fmt(vals[peak_idx])}** around **{pd.Timestamp(peak_date).strftime('%b %Y')}**, "
    sentence += f"while the lowest was **{_fmt(vals[trough_idx])}** around **{pd.Timestamp(trough_date).strftime('%b %Y')}**."

    # Volatility note
    cv = np.std(vals) / np.mean(vals) if np.mean(vals) != 0 else 0
    if cv > 0.4:
        sentence += " The series is quite volatile, with large swings between periods."
    elif cv < 0.1:
        sentence += " The series is fairly stable, with only minor period-to-period swings."

    return sentence


def explain_distribution(df: pd.DataFrame, value_col: str) -> str:
    """Generate a plain-English summary of a value distribution / histogram."""
    vals = df[value_col].dropna()
    mean, median, std = vals.mean(), vals.median(), vals.std()
    skew = vals.skew()

    shape = "fairly symmetric"
    if skew > 0.5:
        shape = "right-skewed — most records are on the lower end, with a few large outliers pulling the average up"
    elif skew < -0.5:
        shape = "left-skewed — most records are on the higher end, with a few unusually low values"

    return (
        f"The typical (median) record is **{_fmt(median)}**, while the average is **{_fmt(mean)}** "
        f"— {'close to each other' if abs(mean-median) < 0.1*mean else 'noticeably different'}, "
        f"meaning the distribution is {shape}. Values typically vary by about **±{_fmt(std)}** around the average."
    )


def explain_breakdown(seg: pd.DataFrame, segment_col: str, value_col: str) -> str:
    """Generate a plain-English summary of a category breakdown bar chart."""
    seg_sorted = seg.sort_values(value_col, ascending=False).reset_index(drop=True)
    total = seg_sorted[value_col].sum()
    top_name = seg_sorted[segment_col].iloc[0]
    top_val = seg_sorted[value_col].iloc[0]
    top_share = (top_val / total * 100) if total else 0

    sentence = f"**{top_name}** leads with **{_fmt(top_val)}** ({top_share:.0f}% of the total shown). "
    if len(seg_sorted) > 1:
        bottom_name = seg_sorted[segment_col].iloc[-1]
        bottom_val = seg_sorted[value_col].iloc[-1]
        sentence += f"**{bottom_name}** trails at **{_fmt(bottom_val)}**. "
        gap = (top_val / bottom_val) if bottom_val else None
        if gap and gap > 2:
            sentence += f"That's roughly **{gap:.1f}x** the difference between the top and bottom {segment_col.lower()}."
    return sentence


def explain_backtest(test: pd.Series, predicted: pd.Series, metrics: dict, freq_label: str) -> str:
    """Generate a plain-English summary of backtest accuracy."""
    avg_actual = test.mean()
    mape = metrics["mape"]

    if np.isnan(mape):
        accuracy_word = "unclear (test period contains zero values, so percentage error can't be computed)"
    elif mape < 10:
        accuracy_word = "quite accurate"
    elif mape < 20:
        accuracy_word = "reasonably accurate"
    elif mape < 35:
        accuracy_word = "moderately accurate, with room for improvement"
    else:
        accuracy_word = "not very accurate for this data — consider a different ARIMA order or model"

    sentence = (
        f"On the last **{len(test)}** {freq_label.lower()} periods held out for testing, "
        f"the model's predictions were **{accuracy_word}**. "
        f"On average, forecasts were off by **{_fmt(metrics['mae'])}**"
    )
    if not np.isnan(mape):
        sentence += f" ({mape:.1f}% of the actual value)"
    sentence += f", with a root-mean-squared error of **{_fmt(metrics['rmse'])}**. "
    sentence += f"For comparison, the average actual value in this test period was **{_fmt(avg_actual)}**."
    return sentence



def explain_forecast(series: pd.Series, mean_forecast: pd.Series, conf_int: pd.DataFrame,
                      order: tuple, freq_label: str, stat_check: dict) -> str:
    """Generate a plain-English summary of an ARIMA forecast chart."""
    last_actual = series.iloc[-1]
    next_forecast = mean_forecast.iloc[0]
    final_forecast = mean_forecast.iloc[-1]
    change_pct = ((next_forecast - last_actual) / last_actual * 100) if last_actual != 0 else None

    direction = "rise" if (change_pct or 0) > 1 else ("fall" if (change_pct or 0) < -1 else "stay roughly flat")
    sentence = f"The model expects sales to **{direction}**"
    if change_pct is not None and abs(change_pct) > 1:
        sentence += f" by about **{abs(change_pct):.0f}%**"
    sentence += f" in the next {freq_label.lower()} period, from **{_fmt(last_actual)}** to **{_fmt(next_forecast)}**. "
    sentence += f"By the end of the forecast window, the projected level is **{_fmt(final_forecast)}**. "

    band_width = (conf_int.iloc[-1, 1] - conf_int.iloc[-1, 0])
    relative_band = band_width / final_forecast if final_forecast else None
    if relative_band is not None:
        if relative_band > 0.5:
            sentence += "The confidence band is quite wide at the end of the horizon, meaning there's significant uncertainty that far out — treat long-range values as rough estimates. "
        else:
            sentence += "The confidence band stays reasonably tight, suggesting the model is fairly confident in this forecast. "

    sentence += f"This uses an ARIMA{order} model"
    if stat_check.get("stationary") is False:
        sentence += ", chosen partly because the raw data wasn't stationary (had a trend), so differencing was applied."
    else:
        sentence += "."

    return sentence


# Keyword-based question answering for the interactive "ask about this chart" box
def answer_question(question: str, context: dict) -> str:
    """
    Very simple rule-based Q&A: matches keywords in the question to canned,
    but data-driven, explanations using the same stats computed for the charts.
    """
    q = question.lower()

    if any(k in q for k in ["accura", "confiden", "trust", "reliable", "backtest", "error", "mae", "rmse", "mape"]):
        if "backtest_explanation" in context:
            return context["backtest_explanation"]
        if "forecast_explanation" in context:
            return context["forecast_explanation"] + " (Tip: scroll down to the 'Model Accuracy (Backtest)' section in the Forecast tab for a real accuracy check against held-out data.)"
        return "Run a forecast first in the Forecast tab, then ask me about its accuracy."

    if any(k in q for k in ["why", "spike", "drop", "increase", "decrease", "peak", "low"]):
        if "trend_explanation" in context:
            return context["trend_explanation"] + " (Note: this is based on overall patterns in your data — for a specific cause like a promotion or event, you'd need to cross-reference your own records.)"
        return "Upload data and view the Overview tab first so I have a trend to explain."

    if any(k in q for k in ["trend", "growing", "declining", "overall"]):
        return context.get("trend_explanation", "I don't have a trend computed yet — check the Overview tab.")

    if any(k in q for k in ["distribut", "spread", "histogram", "average", "median", "typical"]):
        return context.get("distribution_explanation", "I don't have distribution stats yet — check the Overview tab.")

    if any(k in q for k in ["region", "product", "breakdown", "category", "best", "worst", "top", "compare"]):
        return context.get("breakdown_explanation", "Pick a segment column (like Region or Product) in the column mapping to see a breakdown I can explain.")

    if any(k in q for k in ["arima", "model", "order", "p,d,q", "parameter"]):
        return context.get(
            "forecast_explanation",
            "ARIMA stands for AutoRegressive Integrated Moving Average. The (p,d,q) values control how much it "
            "looks at past values (p), how much trend-removal/differencing it does (d), and how much it corrects "
            "based on past forecast errors (q). Run a forecast in the Forecast tab and I can explain your specific result."
        )

    return (
        "I can explain the trend chart, the distribution, category breakdowns, or the ARIMA forecast. "
        "Try asking things like *'what's the overall trend?'*, *'why did sales spike?'*, *'which region is best?'*, "
        "or *'how accurate is the forecast?'*"
    )


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

if "insights" not in st.session_state:
    st.session_state.insights = {}

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
tab_overview, tab_explore, tab_forecast, tab_data, tab_chat = st.tabs(
    ["🏠 Overview", "🔍 Explore", "🔮 Forecast", "🗂️ Raw Data", "💬 Ask About This Data"]
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
        st.plotly_chart(fig, width='stretch')
        st.caption(f"Showing totals aggregated by **{overview_freq_label.lower()}** period — switch above for more or less detail.")
        trend_explanation = explain_trend_chart(ts, date_col, value_col, overview_freq_label)
        st.session_state.insights["trend_explanation"] = trend_explanation
        st.info(f"💡 **What this shows:** {trend_explanation}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.subheader("Distribution")
        fig2 = px.histogram(df, x=value_col, nbins=30, template=PLOTLY_TEMPLATE, color_discrete_sequence=[ACCENT2])
        fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, width='stretch')
        distribution_explanation = explain_distribution(df, value_col)
        st.session_state.insights["distribution_explanation"] = distribution_explanation
        st.info(f"💡 **What this shows:** {distribution_explanation}")
        st.markdown("</div>", unsafe_allow_html=True)

    if segment_col != "None":
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.subheader(f"Breakdown by {segment_col}")
        seg = df.groupby(segment_col)[value_col].sum().reset_index().sort_values(value_col, ascending=False)
        fig3 = px.bar(seg, x=segment_col, y=value_col, template=PLOTLY_TEMPLATE, color=segment_col,
                       color_discrete_sequence=px.colors.qualitative.Bold)
        fig3.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig3, width='stretch')
        breakdown_explanation = explain_breakdown(seg, segment_col, value_col)
        st.session_state.insights["breakdown_explanation"] = breakdown_explanation
        st.info(f"💡 **What this shows:** {breakdown_explanation}")
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
    st.plotly_chart(fig4, width='stretch')
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
        st.plotly_chart(fig5, width='stretch')
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
            st.plotly_chart(fig6, width='stretch')
            forecast_explanation = explain_forecast(series, mean_forecast, conf_int, order, freq_label, stat_check)
            st.session_state.insights["forecast_explanation"] = forecast_explanation
            st.info(f"💡 **What this shows:** {forecast_explanation}")

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
            st.dataframe(forecast_df, width='stretch', hide_index=True)

            csv_bytes = forecast_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download forecast as CSV", csv_bytes, "sales_forecast.csv", "text/csv")

            # ---- BACKTEST / MODEL ACCURACY ----
            st.markdown("---")
            st.markdown("#### 📏 Model Accuracy (Backtest)")
            st.caption(
                "To check how trustworthy this model is, we hide the most recent periods, "
                "retrain on the rest, forecast those hidden periods, and compare against what actually happened."
            )

            max_holdout = max(1, min(12, len(series) - 5))
            holdout_size = st.slider(
                "Holdout periods to test on", 1, max_holdout, min(6, max_holdout), key="holdout_slider"
            )

            if len(series) - holdout_size < 4:
                st.warning("Not enough historical data to backtest with this holdout size. Try a smaller holdout or upload more data.")
            else:
                try:
                    test_actual, test_predicted, metrics = backtest_arima(
                        series.values, series_index_iso, order, holdout_size, freq
                    )

                    bt1, bt2, bt3 = st.columns(3)
                    with bt1:
                        kpi_card("MAE (avg error)", f"{metrics['mae']:,.0f}")
                    with bt2:
                        kpi_card("RMSE", f"{metrics['rmse']:,.0f}")
                    with bt3:
                        kpi_card("MAPE", f"{metrics['mape']:.1f}%" if not np.isnan(metrics['mape']) else "N/A")

                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(
                        x=series.index[:-holdout_size], y=series.values[:-holdout_size],
                        mode="lines", name="Training data", line=dict(color="#6b7280", width=1.5)
                    ))
                    fig_bt.add_trace(go.Scatter(
                        x=test_actual.index, y=test_actual.values,
                        mode="lines+markers", name="Actual (held out)", line=dict(color=ACCENT2, width=3)
                    ))
                    fig_bt.add_trace(go.Scatter(
                        x=test_predicted.index, y=test_predicted.values,
                        mode="lines+markers", name="Predicted", line=dict(color=ACCENT3, width=3, dash="dash")
                    ))
                    fig_bt.update_layout(
                        template=PLOTLY_TEMPLATE, height=380, margin=dict(l=10, r=10, t=30, b=10),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        title="Backtest: Predicted vs Actual on held-out periods",
                    )
                    st.plotly_chart(fig_bt, width='stretch')

                    backtest_explanation = explain_backtest(test_actual, test_predicted, metrics, freq_label)
                    st.session_state.insights["backtest_explanation"] = backtest_explanation
                    st.info(f"💡 **What this shows:** {backtest_explanation}")

                    comparison_df = pd.DataFrame({
                        "Date": test_actual.index,
                        "Actual": test_actual.values,
                        "Predicted": test_predicted.values,
                        "Error": test_actual.values - test_predicted.values,
                    })
                    st.dataframe(comparison_df, width='stretch', hide_index=True)

                except Exception as bt_error:
                    st.error(f"Backtest failed: {bt_error}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- RAW DATA TAB ----
with tab_data:
    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    st.subheader("Raw Data")
    st.dataframe(df, width='stretch', height=480)
    st.caption(f"{len(df):,} rows × {len(df.columns)} columns")
    st.markdown("</div>", unsafe_allow_html=True)

# ---- CHATBOT TAB ----
with tab_chat:
    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    st.subheader("💬 Ask About This Data")
    st.caption(
        "This is a free, rule-based assistant — it answers using the real numbers computed from your charts "
        "(no AI model or API key involved). Visit the Overview and Forecast tabs first so it has things to explain."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    suggestion_cols = st.columns(4)
    suggestions = [
        "What's the overall trend?",
        "Which region is best?",
        "How accurate is the model (backtest)?",
        "Why did sales spike?",
    ]
    clicked_suggestion = None
    for col, suggestion in zip(suggestion_cols, suggestions):
        with col:
            if st.button(suggestion, width='stretch', key=f"sugg_{suggestion}"):
                clicked_suggestion = suggestion

    user_question = st.chat_input("Ask a question about your charts...")
    final_question = clicked_suggestion or user_question

    if final_question:
        answer = answer_question(final_question, st.session_state.insights)
        st.session_state.chat_history.append(("user", final_question))
        st.session_state.chat_history.append(("assistant", answer))

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    if st.session_state.chat_history and st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><hr style='border-color:#232838;'><br>", unsafe_allow_html=True)
st.caption("Built with Streamlit, Plotly & statsmodels ARIMA · Sales Forecast Dashboard")
