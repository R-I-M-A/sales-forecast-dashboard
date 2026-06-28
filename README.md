# Sales Forecast Dashboard

An interactive Streamlit dashboard for sales data: upload a CSV, explore trends
visually, and generate ARIMA-based forecasts.

## Setup

```bash
pip install -r requirements.txt
streamlit run sales_forecast_dashboard.py
```

Your browser will open automatically at `http://localhost:8501`.

## How it works

1. **Upload a CSV** from the sidebar, or check "Use demo dataset" to try it
   instantly with generated sample data.
2. **Column mapping** (in an expander on the main page) auto-detects your date
   and sales columns — adjust if the guess is wrong. You can also pick an
   optional category column (e.g. Region, Product) for breakdown charts.
3. **Overview tab**: KPIs, total sales trend, distribution, and category
   breakdown.
4. **Explore tab**: rolling averages, date-range filtering, per-segment trend
   lines.
5. **Forecast tab**: pick an aggregation period (daily/weekly/monthly), a
   forecast horizon, and an ARIMA (p,d,q) order. The app fits the model,
   plots history + forecast + 80% confidence band, and lets you download the
   forecast as CSV.
6. **Raw Data tab**: full table view of the cleaned data.

## CSV format

Minimum requirement: one date-like column and one numeric sales column, e.g.:

```csv
Date,Sales,Region,Product
2024-01-01,1200,North,Widget A
2024-01-02,980,South,Widget B
...
```

Extra columns are fine — you'll be able to pick which ones to use.

## Notes on ARIMA settings

- **d (differencing)**: if the ADF test shown in the Forecast tab says the
  series is non-stationary, increase `d` (try 1).
- **p, q**: start with small values (1–2) and increase only if the forecast
  looks too simplistic; higher orders risk overfitting on short series.
- If your chosen order fails to converge, the app automatically falls back to
  ARIMA(1,1,1) when "Auto-pick a simple order if ARIMA fails" is checked.

---

## Ideas to extend this further

- **🤖 AI chatbot for data Q&A** — embed a chat panel (using the Anthropic API
  or LangChain) where users ask things like *"Which region grew fastest last
  quarter?"* and the model answers using the loaded dataframe (e.g. via
  pandas-based tool calls or a text-to-pandas-query layer).
- **📦 Smarter forecasting** — auto-select ARIMA orders with `pmdarima.auto_arima`,
  or add Facebook Prophet / `statsmodels` ETS as alternative models with a
  model-comparison view (MAE/RMSE backtesting).
- **🔔 Anomaly detection & alerts** — flag days/weeks where actual sales
  deviate significantly from the rolling average or forecast band, and surface
  them as a "Watchlist" panel.
- **📧 Scheduled reports** — use `streamlit` + a cron job or GitHub Action to
  regenerate forecasts daily/weekly and email a PDF/CSV summary (combine with
  the PDF skill to auto-generate a styled report).
- **🗄️ Database backend** — swap CSV upload for a live connection (Postgres,
  BigQuery, Google Sheets) so the dashboard always reflects current data
  instead of a static upload.
- **🔐 Multi-user support** — add login (e.g. `streamlit-authenticator`) so
  different sales teams see only their own data/forecasts.
- **📊 What-if simulator** — sliders to simulate promotions/price changes and
  see a projected impact on the forecast curve.
- **🌍 Multi-series forecasting** — forecast each Region/Product separately
  and show them on one comparative chart instead of only the aggregate total.
- **📱 Deploy it** — host for free on Streamlit Community Cloud, or containerize
  with Docker for internal company deployment.
