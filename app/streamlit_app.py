"""
War Predictive Dashboard
Geopolitical Risk Index (GPR) Analysis like Bridgewater Macro Model

Inspired by Ray Dalio’s "Principles," I built this project to function as a "machine"—a systematic approach to processing data into actionable reality. By mapping out cause-effect relationships within the data, this tool predicts future outcomes and provides a front-end interface to translate complex insights into clear, human-centric narratives

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="War Risk Predictive Dashboard",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #4a9eff;
    }
    .metric-card.red   { border-left-color: #ff4b4b; }
    .metric-card.orange{ border-left-color: #ffa726; }
    .metric-card.green { border-left-color: #26c281; }
    .metric-card.blue  { border-left-color: #4a9eff; }
    .risk-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

EVENTS = {
    "2001-09-01": "9/11 Attacks",
    "2002-08-01": "Iraq War Buildup",
    "2003-03-01": "Iraq War Begins",
    "2006-07-01": "Lebanon War",
    "2008-08-01": "Russia–Georgia War",
    "2008-11-01": "Mumbai Attacks",
    "2009-06-01": "Iran Election Crisis",
    "2010-06-01": "Arab Spring Onset",
    "2011-03-01": "Libya Intervention",
    "2013-01-01": "Mali / Syria Escalation",
    "2014-03-01": "Crimea Annexation",
    "2014-12-01": "N. Korea / ISIS Peak",
    "2017-06-01": "N. Korea ICBM Tensions",
    "2018-05-01": "JCPOA Withdrawal",
    "2019-07-01": "US–Iran Gulf Crisis",
    "2020-01-01": "Soleimani Strike",
    "2021-01-01": "Capitol Storming",
    "2022-03-01": "Russia–Ukraine War",
    "2023-09-01": "Multi-Front Escalation",
    "2023-10-01": "Israel–Hamas War",
}

REGIME_COLORS = {
    "Low Risk":   "#26c281",
    "Normal":     "#4a9eff",
    "Elevated":   "#ffa726",
    "High Alert": "#ff4b4b",
    "Extreme":    "#9c27b0",
}

@st.cache_data
def load_gpr_data() -> pd.DataFrame:
    data_dir = Path(__file__).parent.parent / "data"
    csv_path = data_dir / "geopolitical_risk_index.csv"
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    df.columns = ["GPR"]

    # Try loading the XLS for additional columns
    xls_path = data_dir / "data_gpr_export.xls"
    if xls_path.exists():
        try:
            xls = pd.read_excel(xls_path, header=0, index_col=0, parse_dates=True)
            xls.index = pd.to_datetime(xls.index, errors="coerce")
            xls = xls.dropna(how="all")
            # Keep only numeric columns and merge
            xls_num = xls.select_dtypes(include="number")
            if not xls_num.empty:
                xls_num.index.name = "Date"
                df = df.join(xls_num, how="left")
        except Exception:
            pass
    return df


@st.cache_data
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    gpr = d["GPR"]

    # Rolling stats
    d["MA_3"]    = gpr.rolling(3).mean()
    d["MA_6"]    = gpr.rolling(6).mean()
    d["MA_12"]   = gpr.rolling(12).mean()
    d["MA_24"]   = gpr.rolling(24).mean()
    d["Std_12"]  = gpr.rolling(12).std()
    d["Std_6"]   = gpr.rolling(6).std()

    # Bollinger bands
    d["BB_Upper"] = d["MA_12"] + 2 * d["Std_12"]
    d["BB_Lower"] = d["MA_12"] - 2 * d["Std_12"]

    # Z-scores
    d["Z_12"] = (gpr - d["MA_12"]) / d["Std_12"]
    d["Z_6"]  = (gpr - d["MA_6"])  / d["Std_6"]

    # Momentum / rate of change
    d["MoM"]  = gpr.pct_change(1)   * 100   # month-on-month %
    d["QoQ"]  = gpr.pct_change(3)   * 100   # quarter-on-quarter %
    d["YoY"]  = gpr.pct_change(12)  * 100   # year-on-year %

    # Absolute level shift
    d["Delta_1"]  = gpr.diff(1)
    d["Delta_3"]  = gpr.diff(3)
    d["Delta_12"] = gpr.diff(12)

    # Percentile rank across full history
    d["Percentile"] = gpr.rank(pct=True) * 100

    # Expanding stats (lifetime)
    d["Hist_Mean"] = gpr.expanding().mean()
    d["Hist_Std"]  = gpr.expanding().std()
    d["Z_Hist"]    = (gpr - d["Hist_Mean"]) / d["Hist_Std"]

    # Regime classification (based on 12M z-score)
    def classify(z):
        if pd.isna(z):
            return "Normal"
        if z > 2.0:
            return "Extreme"
        if z > 1.0:
            return "High Alert"
        if z > 0.5:
            return "Elevated"
        if z < -1.0:
            return "Low Risk"
        return "Normal"

    d["Regime"] = d["Z_12"].apply(classify)

    # War Risk Score (0–100 composite)
    z_norm    = d["Z_12"].clip(-3, 3) / 3          # -1 to +1
    mom_norm  = (d["QoQ"] / 20).clip(-1, 1)         # normalise % change
    pct_norm  = (d["Percentile"] - 50) / 50         # -1 to +1
    d["War_Risk_Score"] = ((z_norm * 0.5 + mom_norm * 0.2 + pct_norm * 0.3) * 50 + 50).clip(0, 100)

    return d


def sidebar(df: pd.DataFrame):
    st.sidebar.title("🌐 War Risk Dashboard")
    st.sidebar.markdown("**Geopolitical Risk Index (GPR)**  \n*Monthly, 2000 – present*")
    st.sidebar.divider()

    min_date = df.index.min().date()
    max_date = df.index.max().date()

    start = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    end   = st.sidebar.date_input("End Date",   value=max_date, min_value=min_date, max_value=max_date)

    st.sidebar.divider()
    z_thresh   = st.sidebar.slider("Z-Score Alert Threshold", 0.5, 3.0, 1.0, 0.1,
                                   help="Months above this z-score are flagged as elevated risk")
    show_events = st.sidebar.checkbox("Show Key Events", value=True)
    show_bands  = st.sidebar.checkbox("Show Bollinger Bands", value=True)

    st.sidebar.divider()
    st.sidebar.caption("Data source: Geopolitical Risk Index (Caldara & Iacoviello)")

    return pd.Timestamp(start), pd.Timestamp(end), z_thresh, show_events, show_bands


def kpi_row(df: pd.DataFrame, feat: pd.DataFrame):
    latest = feat.iloc[-1]
    prev   = feat.iloc[-2] if len(feat) > 1 else latest

    gpr_val    = latest["GPR"]
    gpr_delta  = latest["Delta_1"]
    z_val      = latest["Z_12"]
    pct_val    = latest["Percentile"]
    regime     = latest["Regime"]
    risk_score = latest["War_Risk_Score"]

    regime_color = REGIME_COLORS.get(regime, "#4a9eff")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current GPR", f"{gpr_val:.1f}", f"{gpr_delta:+.1f} MoM")
    c2.metric("12M Z-Score", f"{z_val:.2f}", f"{(z_val - feat['Z_12'].mean()):.2f} vs avg")
    c3.metric("Percentile Rank", f"{pct_val:.0f}th", f"{latest['YoY']:+.1f}% YoY")
    c4.metric("War Risk Score", f"{risk_score:.0f} / 100",
              f"{'↑' if risk_score > 50 else '↓'} {'High' if risk_score > 66 else 'Moderate' if risk_score > 40 else 'Low'}")
    c5.metric("Risk Regime", regime, "Current classification")

    # Regime badge
    st.markdown(
        f"<p>Current Regime: "
        f"<span style='background:{regime_color};color:#fff;padding:3px 12px;"
        f"border-radius:12px;font-weight:700;'>{regime}</span></p>",
        unsafe_allow_html=True,
    )


def tab_historical(feat: pd.DataFrame, show_events: bool, show_bands: bool):
    st.subheader("Geopolitical Risk Index — Full Historical Timeline")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.06,
        subplot_titles=("GPR Level", "Year-on-Year Change (%)"),
    )

    prev_regime = None
    start_shade = None
    for dt, row in feat.iterrows():
        r = row["Regime"]
        if r != prev_regime:
            if prev_regime and prev_regime != "Normal":
                color = REGIME_COLORS.get(prev_regime, "rgba(0,0,0,0)")
                fig.add_vrect(
                    x0=start_shade, x1=dt,
                    fillcolor=color, opacity=0.12, layer="below", line_width=0,
                    row=1, col=1,
                )
            start_shade = dt
            prev_regime = r

    if show_bands:
        fig.add_trace(
            go.Scatter(
                x=feat.index, y=feat["BB_Upper"],
                mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=feat.index, y=feat["BB_Lower"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(74,158,255,0.1)",
                name="Bollinger Band", showlegend=True,
            ), row=1, col=1,
        )

    # MAs
    for ma, color, dash in [
        ("MA_12", "#ffa726", "dot"),
        ("MA_24", "#ef5350", "dash"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=feat.index, y=feat[ma],
                mode="lines", line=dict(color=color, width=1.5, dash=dash),
                name=ma.replace("_", "-") + "M MA",
            ), row=1, col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=feat.index, y=feat["GPR"],
            mode="lines", line=dict(color="#4a9eff", width=2),
            name="GPR",
        ), row=1, col=1,
    )

    if show_events:
        for date_str, label in EVENTS.items():
            dt = pd.Timestamp(date_str)
            if dt in feat.index:
                gpr_val = feat.loc[dt, "GPR"]
                fig.add_vline(x=dt, line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"), row=1, col=1)
                fig.add_annotation(
                    x=dt, y=gpr_val + 4, text=label[:12],
                    showarrow=False, font=dict(size=8, color="#cccccc"),
                    textangle=-55, row=1, col=1,
                )

    yoy = feat["YoY"].dropna()
    fig.add_trace(
        go.Bar(
            x=yoy.index, y=yoy,
            marker=dict(
                color=yoy.apply(lambda v: "#ff4b4b" if v > 0 else "#26c281"),
                opacity=0.75,
            ),
            name="YoY Change %",
        ), row=2, col=1,
    )
    fig.add_hline(y=0, line=dict(color="white", width=0.5), row=2, col=1)

    fig.update_layout(
        height=600, template="plotly_dark",
        legend=dict(orientation="h", y=1.02),
        margin=dict(t=60, b=20),
    )
    fig.update_yaxes(title_text="GPR Index", row=1, col=1)
    fig.update_yaxes(title_text="YoY %", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    if show_events:
        with st.expander("📅 Key Geopolitical Events in Dataset"):
            rows = []
            for date_str, label in EVENTS.items():
                dt = pd.Timestamp(date_str)
                if dt in feat.index:
                    rows.append({
                        "Date": dt.strftime("%b %Y"),
                        "Event": label,
                        "GPR": f"{feat.loc[dt,'GPR']:.1f}",
                        "Z-Score": f"{feat.loc[dt,'Z_12']:.2f}",
                        "Regime": feat.loc[dt, "Regime"],
                    })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

def tab_signals(feat: pd.DataFrame, z_thresh: float):
    st.subheader("War Risk Signal Detection")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.45, 0.30, 0.25],
            vertical_spacing=0.05,
            subplot_titles=("War Risk Score", "Z-Score (12M)", "Momentum (QoQ %)"),
        )

        fig.add_trace(
            go.Scatter(
                x=feat.index, y=feat["War_Risk_Score"],
                mode="lines", fill="tozeroy",
                line=dict(color="#ffa726", width=2),
                fillcolor="rgba(255,167,38,0.15)",
                name="War Risk Score",
            ), row=1, col=1,
        )
        fig.add_hline(y=66, line=dict(color="#ff4b4b", width=1, dash="dash"), row=1, col=1)
        fig.add_hline(y=40, line=dict(color="#26c281", width=1, dash="dash"), row=1, col=1)
        fig.add_annotation(x=feat.index[-1], y=70, text="High Risk", font=dict(color="#ff4b4b", size=9),
                           showarrow=False, xanchor="right", row=1, col=1)
        fig.add_annotation(x=feat.index[-1], y=35, text="Low Risk", font=dict(color="#26c281", size=9),
                           showarrow=False, xanchor="right", row=1, col=1)

        z = feat["Z_12"].dropna()
        fig.add_trace(
            go.Scatter(
                x=z.index, y=z, mode="lines",
                line=dict(color="#4a9eff", width=1.5), name="Z-Score",
            ), row=2, col=1,
        )
        fig.add_hline(y=z_thresh,  line=dict(color="#ffa726", width=1, dash="dot"), row=2, col=1)
        fig.add_hline(y=-z_thresh, line=dict(color="#26c281", width=1, dash="dot"), row=2, col=1)
        fig.add_hline(y=0,         line=dict(color="white", width=0.5), row=2, col=1)

        high_z = feat[feat["Z_12"] > z_thresh]
        fig.add_trace(
            go.Scatter(
                x=high_z.index, y=high_z["Z_12"],
                mode="markers", marker=dict(color="#ff4b4b", size=7, symbol="triangle-up"),
                name=f"Z > {z_thresh:.1f}",
            ), row=2, col=1,
        )

        qoq = feat["QoQ"].dropna()
        fig.add_trace(
            go.Bar(
                x=qoq.index, y=qoq,
                marker=dict(color=qoq.apply(lambda v: "#ff4b4b" if v > 0 else "#26c281"), opacity=0.75),
                name="QoQ %",
            ), row=3, col=1,
        )
        fig.add_hline(y=0, line=dict(color="white", width=0.5), row=3, col=1)

        fig.update_layout(height=580, template="plotly_dark", showlegend=False, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Risk Signal Summary")

        latest = feat.iloc[-1]
        signals = {
            "GPR Level":        (latest["GPR"], "normal"),
            "Z-Score (12M)":    (f"{latest['Z_12']:.2f}",
                                 "red" if latest["Z_12"] > z_thresh else "green"),
            "Z-Score (6M)":     (f"{latest['Z_6']:.2f}",
                                 "red" if latest["Z_6"] > z_thresh else "green"),
            "War Risk Score":   (f"{latest['War_Risk_Score']:.0f}/100",
                                 "red" if latest["War_Risk_Score"] > 66 else
                                 "orange" if latest["War_Risk_Score"] > 40 else "green"),
            "MoM Change":       (f"{latest['MoM']:+.1f}%",
                                 "red" if latest["MoM"] > 3 else "green"),
            "QoQ Change":       (f"{latest['QoQ']:+.1f}%",
                                 "red" if latest["QoQ"] > 5 else "green"),
            "YoY Change":       (f"{latest['YoY']:+.1f}%",
                                 "red" if latest["YoY"] > 10 else "green"),
            "Percentile":       (f"{latest['Percentile']:.0f}th",
                                 "red" if latest["Percentile"] > 75 else "green"),
            "Regime":           (latest["Regime"],
                                 "red" if latest["Regime"] in ("High Alert", "Extreme") else "blue"),
        }
        for label, (value, color_cls) in signals.items():
            dot = {"red": "🔴", "orange": "🟠", "green": "🟢", "blue": "🔵", "normal": "⚪"}[color_cls]
            st.markdown(f"**{label}** {dot}  \n`{value}`")
            st.divider()

        n_alerts = int((feat["Z_12"] > z_thresh).sum())
        pct_alerts = n_alerts / len(feat) * 100
        st.metric("Months in Alert (Z > threshold)", f"{n_alerts}", f"{pct_alerts:.1f}% of history")

        st.markdown("#### 5 Most Recent Alerts")
        recent_alerts = (
            feat[feat["Z_12"] > z_thresh][["GPR", "Z_12", "Regime"]]
            .sort_index(ascending=False)
            .head(5)
        )
        recent_alerts.index = recent_alerts.index.strftime("%b %Y")
        st.dataframe(recent_alerts.round(2), use_container_width=True)


def tab_stats(feat: pd.DataFrame):
    st.subheader("Statistical Deep-Dive")

    col1, col2 = st.columns(2)

    with col1:
        # Distribution
        gpr = feat["GPR"].dropna()
        mu, sigma = gpr.mean(), gpr.std()

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=gpr, nbinsx=40,
            name="GPR Distribution",
            marker_color="#4a9eff", opacity=0.75,
            histnorm="probability density",
        ))
        x_range = np.linspace(gpr.min(), gpr.max(), 200)
        y_norm  = stats.norm.pdf(x_range, mu, sigma)
        fig.add_trace(go.Scatter(
            x=x_range, y=y_norm,
            mode="lines", line=dict(color="#ffa726", width=2),
            name="Normal Fit",
        ))
        fig.add_vline(x=gpr.iloc[-1], line=dict(color="#ff4b4b", width=2, dash="dash"),
                      annotation_text=f"Current: {gpr.iloc[-1]:.1f}", annotation_font_color="#ff4b4b")
        fig.add_vline(x=mu, line=dict(color="#26c281", width=1.5, dash="dot"),
                      annotation_text=f"Mean: {mu:.1f}", annotation_font_color="#26c281")
        fig.update_layout(
            title="GPR Distribution (2000–present)",
            height=350, template="plotly_dark",
            xaxis_title="GPR", yaxis_title="Density",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Stats table
        skew = float(stats.skew(gpr.dropna()))
        kurt = float(stats.kurtosis(gpr.dropna()))
        stat_tbl = pd.DataFrame({
            "Statistic": ["Mean", "Median", "Std Dev", "Min", "Max",
                          "25th Pct", "75th Pct", "Skewness", "Kurtosis"],
            "Value":     [f"{mu:.2f}", f"{gpr.median():.2f}", f"{sigma:.2f}",
                          f"{gpr.min():.2f}", f"{gpr.max():.2f}",
                          f"{gpr.quantile(0.25):.2f}", f"{gpr.quantile(0.75):.2f}",
                          f"{skew:.3f}", f"{kurt:.3f}"],
        })
        st.dataframe(stat_tbl, hide_index=True, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=feat.index, y=feat["Percentile"],
            mode="lines", fill="tozeroy",
            line=dict(color="#a855f7", width=2),
            fillcolor="rgba(168,85,247,0.15)",
            name="Percentile Rank",
        ))
        fig2.add_hline(y=75, line=dict(color="#ff4b4b", width=1, dash="dot"),
                       annotation_text="75th", annotation_font_color="#ff4b4b")
        fig2.add_hline(y=25, line=dict(color="#26c281", width=1, dash="dot"),
                       annotation_text="25th", annotation_font_color="#26c281")
        fig2.update_layout(
            title="GPR Percentile Rank Over Time",
            height=350, template="plotly_dark",
            yaxis_title="Percentile",
        )
        st.plotly_chart(fig2, use_container_width=True)

        regime_counts = feat["Regime"].value_counts().reset_index()
        regime_counts.columns = ["Regime", "Months"]
        regime_counts["% of History"] = (regime_counts["Months"] / len(feat) * 100).round(1)
        regime_counts["Avg GPR"] = regime_counts["Regime"].apply(
            lambda r: feat.loc[feat["Regime"] == r, "GPR"].mean()
        ).round(1)
        st.markdown("#### Regime Frequency")
        st.dataframe(regime_counts, hide_index=True, use_container_width=True)

    st.markdown("#### Rolling Volatility & Trend")
    fig3 = make_subplots(rows=1, cols=2, subplot_titles=("12M Rolling Std Dev", "GPR vs Long-Run Mean"))

    fig3.add_trace(go.Scatter(
        x=feat.index, y=feat["Std_12"],
        mode="lines", line=dict(color="#ffa726", width=2), name="Std Dev (12M)",
    ), row=1, col=1)

    fig3.add_trace(go.Scatter(
        x=feat.index, y=feat["GPR"],
        mode="lines", line=dict(color="#4a9eff", width=1.5), name="GPR",
    ), row=1, col=2)
    fig3.add_trace(go.Scatter(
        x=feat.index, y=feat["Hist_Mean"],
        mode="lines", line=dict(color="#ef5350", width=1.5, dash="dash"), name="Expanding Mean",
    ), row=1, col=2)

    fig3.update_layout(height=320, template="plotly_dark", margin=dict(t=40, b=10))
    st.plotly_chart(fig3, use_container_width=True)

def tab_forecast(feat: pd.DataFrame):
    st.subheader("Trend Extrapolation & Predictive Signals")

    gpr = feat["GPR"].dropna().reset_index()
    gpr.columns = ["Date", "GPR"]
    gpr["t"] = np.arange(len(gpr))

    col_l, col_r = st.columns([1, 3])
    with col_l:
        lookback = st.slider("Trend Lookback (months)", 12, 120, 36,
                             help="Fit linear trend on the last N months")
        n_fwd    = st.slider("Forecast Horizon (months)", 3, 24, 12)
        conf     = st.slider("Confidence Level %", 80, 99, 90)

    tail = gpr.tail(lookback)
    coef = np.polyfit(tail["t"], tail["GPR"], 1) 
    poly = np.poly1d(coef)

    fitted   = poly(tail["t"])
    residuals = tail["GPR"].values - fitted
    se       = residuals.std()
    z_conf   = stats.norm.ppf(0.5 + conf / 200)

    last_t    = gpr["t"].iloc[-1]
    last_date = gpr["Date"].iloc[-1]
    future_t  = np.arange(last_t + 1, last_t + n_fwd + 1)
    future_dt = pd.date_range(last_date + pd.DateOffset(months=1), periods=n_fwd, freq="MS")
    fcast_val = poly(future_t)

    with col_r:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=gpr["Date"], y=gpr["GPR"],
            mode="lines", line=dict(color="#4a9eff", width=2), name="Historical GPR",
        ))

        fig.add_trace(go.Scatter(
            x=tail["Date"], y=fitted,
            mode="lines", line=dict(color="#ffa726", width=2, dash="dash"),
            name=f"Linear Trend ({lookback}M)",
        ))

        fig.add_trace(go.Scatter(
            x=np.concatenate([future_dt, future_dt[::-1]]),
            y=np.concatenate([fcast_val + z_conf * se, (fcast_val - z_conf * se)[::-1]]),
            fill="toself", fillcolor="rgba(255,167,38,0.15)",
            line=dict(width=0), showlegend=True,
            name=f"{conf}% Confidence Band",
        ))
        fig.add_trace(go.Scatter(
            x=future_dt, y=fcast_val,
            mode="lines+markers", line=dict(color="#ef5350", width=2, dash="dot"),
            marker=dict(size=6), name="Forecast",
        ))

        hist_mean = gpr["GPR"].mean()
        fig.add_hline(y=hist_mean, line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"),
                      annotation_text=f"Hist. Mean {hist_mean:.1f}", annotation_font_color="white")

        fig.update_layout(
            height=420, template="plotly_dark",
            title=f"GPR Trend Extrapolation — {n_fwd}M Forecast",
            xaxis_title="Date", yaxis_title="GPR",
        )
        st.plotly_chart(fig, use_container_width=True)

    fcast_df = pd.DataFrame({
        "Date":          future_dt.strftime("%b %Y"),
        "GPR Forecast":  fcast_val.round(1),
        f"Lower {conf}%": (fcast_val - z_conf * se).round(1),
        f"Upper {conf}%": (fcast_val + z_conf * se).round(1),
    })
    trend_dir = "↑ Rising" if coef[0] > 0 else "↓ Falling"
    st.info(f"Trend direction: **{trend_dir}** ({coef[0]:+.3f} pts/month over last {lookback} months)")
    st.dataframe(fcast_df, hide_index=True, use_container_width=True)

    st.markdown("#### Momentum-Based Risk Signals")
    momentum_cols = ["MoM", "QoQ", "YoY", "Delta_1", "Delta_3", "Delta_12"]
    avail = [c for c in momentum_cols if c in feat.columns]
    mo = feat[avail].tail(24)
    mo.index = mo.index.strftime("%b %Y")
    st.dataframe(mo.round(2), use_container_width=True)


def tab_regimes(feat: pd.DataFrame):
    st.subheader("Geopolitical Regime Analysis")

    spans = []
    curr_regime = None
    start_dt    = None
    for dt, row in feat.iterrows():
        r = row["Regime"]
        if r != curr_regime:
            if curr_regime:
                spans.append({
                    "Start":    start_dt,
                    "End":      dt,
                    "Regime":   curr_regime,
                    "Duration": int((dt - start_dt).days / 30),
                    "Avg GPR":  round(feat.loc[start_dt:dt, "GPR"].mean(), 1),
                    "Max GPR":  round(feat.loc[start_dt:dt, "GPR"].max(), 1),
                })
            curr_regime = r
            start_dt    = dt
    spans_df = pd.DataFrame(spans)
    if not spans_df.empty:
        spans_df["Start"] = spans_df["Start"].dt.strftime("%b %Y")
        spans_df["End"]   = spans_df["End"].dt.strftime("%b %Y")

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = go.Figure()
        for regime, color in REGIME_COLORS.items():
            mask = feat["Regime"] == regime
            sub  = feat[mask]
            fig.add_trace(go.Scatter(
                x=sub.index, y=sub["GPR"],
                mode="markers",
                marker=dict(color=color, size=6, opacity=0.8),
                name=regime,
            ))
        fig.add_trace(go.Scatter(
            x=feat.index, y=feat["MA_12"],
            mode="lines", line=dict(color="rgba(255,255,255,0.4)", width=1.5),
            name="12M MA",
        ))
        fig.update_layout(
            title="GPR Coloured by Risk Regime",
            height=380, template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rc = feat["Regime"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=rc.index,
            values=rc.values,
            marker_colors=[REGIME_COLORS.get(r, "#888") for r in rc.index],
            textinfo="label+percent",
            hole=0.4,
        ))
        fig2.update_layout(
            title="Regime Distribution",
            height=380, template="plotly_dark",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Regime Periods (all transitions)")
    high_spans = spans_df[spans_df["Regime"].isin(["High Alert", "Extreme"])] if not spans_df.empty else spans_df
    with st.expander("View High Alert / Extreme Periods", expanded=True):
        if not high_spans.empty:
            st.dataframe(high_spans, hide_index=True, use_container_width=True)
        else:
            st.info("No High Alert periods with current settings.")


def tab_data(feat: pd.DataFrame):
    st.subheader("Raw Data Explorer")

    display_cols = ["GPR", "MA_12", "Z_12", "War_Risk_Score", "Regime",
                    "Percentile", "MoM", "QoQ", "YoY"]
    avail = [c for c in display_cols if c in feat.columns]
    view  = feat[avail].copy()
    view.index = view.index.strftime("%Y-%m")
    view = view.sort_index(ascending=False)

    st.dataframe(view.round(2), use_container_width=True, height=500)

    csv_data = feat[avail].round(3).to_csv()
    st.download_button(
        "⬇ Download as CSV",
        data=csv_data,
        file_name="war_risk_analysis.csv",
        mime="text/csv",
    )

def main():
    raw  = load_gpr_data()
    feat = compute_features(raw)

    # Sidebar controls
    start, end, z_thresh, show_events, show_bands = sidebar(raw)

    # Filter data
    mask = (feat.index >= start) & (feat.index <= end)
    feat_filt = feat[mask].copy()

    if feat_filt.empty:
        st.error("No data for selected date range.")
        return

    kpi_row(raw, feat)   
    st.divider()

    tabs = st.tabs([
        "📈 Historical Analysis",
        "⚠️ War Risk Signals",
        "📊 Statistical Analysis",
        "🔮 Forecast & Trend",
        "🗂️ Regime Analysis",
        "🗄️ Data Explorer",
    ])

    with tabs[0]:
        tab_historical(feat_filt, show_events, show_bands)
    with tabs[1]:
        tab_signals(feat_filt, z_thresh)
    with tabs[2]:
        tab_stats(feat_filt)
    with tabs[3]:
        tab_forecast(feat_filt)
    with tabs[4]:
        tab_regimes(feat_filt)
    with tabs[5]:
        tab_data(feat_filt)


if __name__ == "__main__":
    main()
