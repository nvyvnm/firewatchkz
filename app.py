import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FireWatch KZ",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #080c10;
    color: #cdd9e5;
  }
  .main, .block-container { background-color: #080c10; }

  /* ── Hero ── */
  .hero {
    background: linear-gradient(160deg, #110500 0%, #2a0d00 40%, #110500 100%);
    border: 1px solid #ff6b2b33;
    border-radius: 20px;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: "🔥";
    position: absolute;
    right: 2rem; top: 1.5rem;
    font-size: 6rem;
    opacity: 0.08;
  }
  .hero-tag {
    display: inline-block;
    background: #ff6b2b22;
    border: 1px solid #ff6b2b55;
    color: #ff8c55;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.25rem 0.75rem;
    border-radius: 99px;
    margin-bottom: 1rem;
  }
  .hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: #ff6b2b;
    margin: 0 0 0.5rem 0;
    letter-spacing: -1.5px;
    line-height: 1;
  }
  .hero-sub {
    font-size: 1.05rem;
    color: #6e7f8d;
    margin: 0 0 1.2rem 0;
    max-width: 520px;
  }
  .hero-meta {
    font-size: 0.8rem;
    color: #4a5568;
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
  }
  .hero-meta span { color: #8b949e; }

  /* ── Metric cards ── */
  .metrics-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
  .metric-card {
    flex: 1; min-width: 130px;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
  }
  .metric-label {
    font-size: 0.72rem;
    color: #6e7f8d;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
  }
  .metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #ff6b2b;
    line-height: 1;
  }
  .metric-unit { font-size: 0.78rem; color: #4a5568; margin-top: 0.2rem; }

  /* ── Section titles ── */
  .section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    color: #6e7f8d;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-left: 3px solid #ff6b2b;
    padding-left: 0.75rem;
    margin: 2rem 0 1rem 0;
  }

  /* ── Risk result box ── */
  .risk-box {
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
    margin: 0.5rem 0 1rem 0;
    position: relative;
  }
  .risk-low  { background: #091a0f; border: 2px solid #238636; }
  .risk-med  { background: #1c1500; border: 2px solid #9e6a03; }
  .risk-high { background: #1c0500; border: 2px solid #da3633; }
  .risk-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
  .risk-level {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
  }
  .risk-pct {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.5rem;
    font-weight: 700;
    color: #cdd9e5;
    line-height: 1;
  }
  .risk-pct-label { font-size: 0.78rem; color: #6e7f8d; margin: 0.2rem 0 0.8rem; }
  .risk-advice {
    font-size: 0.88rem;
    color: #8b949e;
    background: #ffffff08;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-top: 0.5rem;
  }

  /* ── Sidebar ── */
  div[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #21262d;
  }
  .sidebar-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #cdd9e5;
    margin-bottom: 0.2rem;
  }
  .sidebar-sub { font-size: 0.8rem; color: #6e7f8d; margin-bottom: 1.2rem; }

  /* ── Button ── */
  .stButton > button {
    background: linear-gradient(135deg, #ff6b2b, #e05a1e);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.65rem 1.5rem;
    width: 100%;
    letter-spacing: 0.02em;
    transition: opacity 0.2s, transform 0.1s;
  }
  .stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
  .stButton > button:active { transform: translateY(0); }

  /* ── Slider accent ── */
  .stSlider > div > div > div { background: #ff6b2b !important; }

  /* ── Table ── */
  .stDataFrame { border-radius: 10px; overflow: hidden; }

  /* ── Divider ── */
  hr { border-color: #21262d !important; }

  /* ── Info box ── */
  .info-hint {
    background: #0d1117;
    border: 1px dashed #21262d;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    color: #4a5568;
    font-size: 0.9rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Data & Model ──────────────────────────────────────────────────────────────
@st.cache_data
def build_model():
    np.random.seed(42)
    n = 1200

    temp     = np.random.uniform(5,  50,  n)
    humidity = np.random.uniform(5,  95,  n)
    wind     = np.random.uniform(0,  30,  n)
    drought  = np.random.uniform(0,  800, n)
    ffmc     = np.random.uniform(18, 96,  n)
    dmc      = np.random.uniform(1,  291, n)
    rain     = np.random.uniform(0,  6.4, n)

    score = (
        0.30 * (temp / 50) +
        0.28 * (1 - humidity / 95) +
        0.14 * (wind / 30) +
        0.14 * (drought / 800) +
        0.09 * (ffmc / 96) +
        0.05 * (dmc / 291) -
        0.08 * (rain / 6.4)
    )
    score = np.clip(score + np.random.normal(0, 0.04, n), 0, 1)
    labels = np.where(score < 0.33, 0, np.where(score < 0.66, 1, 2))

    df = pd.DataFrame({
        "temp": temp, "humidity": humidity, "wind": wind,
        "drought": drought, "ffmc": ffmc, "dmc": dmc,
        "rain": rain, "label": labels
    })

    feats = ["temp", "humidity", "wind", "drought", "ffmc", "dmc", "rain"]
    X, y = df[feats], df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                  min_samples_split=4, random_state=42)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    return clf, acc, feats

model, accuracy, FEATURES = build_model()

RISK = {
    0: dict(label="Low Risk",    icon="🟢", css="risk-low",
            color="#238636", advice="Conditions are safe. Routine monitoring is sufficient."),
    1: dict(label="Moderate Risk", icon="🟡", css="risk-med",
            color="#9e6a03", advice="Exercise caution. Increased patrol activity recommended."),
    2: dict(label="High Risk",   icon="🔴", css="risk-high",
            color="#da3633", advice="DANGER — immediate response required. Alert fire services now."),
}

KZ_REGIONS = {
    "East Kazakhstan":    (49.5, 82.0, 2),
    "Almaty Region":      (43.5, 77.5, 2),
    "Pavlodar":           (52.5, 77.0, 1),
    "Karaganda":          (48.0, 73.0, 1),
    "Zhambyl":            (43.0, 71.5, 1),
    "Akmola":             (51.5, 71.0, 0),
    "Kostanay":           (53.0, 63.5, 0),
    "North Kazakhstan":   (54.0, 69.5, 0),
    "Turkestan":          (41.5, 68.5, 0),
    "Aktobe":             (50.3, 57.2, 0),
    "Kyzylorda":          (45.5, 65.0, 0),
    "West Kazakhstan":    (51.5, 51.5, 0),
    "Atyrau":             (47.5, 52.0, 0),
    "Mangystau":          (43.5, 52.0, 0),
}


# ── Hero ──────────────────────────────────────────────────────────────────────
now = datetime.now().strftime("%B %d, %Y · %H:%M")
st.markdown(f"""
<div class="hero">
  <div class="hero-tag">SmartScape Hackathon 2026 · Ecology Track</div>
  <h1>FireWatch KZ</h1>
  <p class="hero-sub">AI-powered early wildfire risk detection system for Kazakhstan — protecting forests before the first spark.</p>
  <div class="hero-meta">
    <div>Author <span>Nazym Yelubay</span></div>
    <div>Team <span>2N</span></div>
    <div>Updated <span>{now}</span></div>
    <div>Model <span>Random Forest · {accuracy*100:.1f}% accuracy</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Top metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    ("Model Accuracy", f"{accuracy*100:.1f}", "%"),
    ("Regions Monitored", "14", "oblasts"),
    ("Input Parameters", "7", "features"),
    ("Trees in Forest", "200", "estimators"),
    ("Training Samples", "960", "data points"),
]
for col, (label, val, unit) in zip([c1,c2,c3,c4,c5], cards):
    with col:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-unit">{unit}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header">🌡️ Weather Parameters</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Adjust the sliders to match current conditions and run the prediction.</div>', unsafe_allow_html=True)

    temp     = st.slider("Temperature (°C)",          5,   50,  30)
    humidity = st.slider("Relative Humidity (%)",      5,   95,  35)
    wind     = st.slider("Wind Speed (m/s)",            0,   30,  12)
    rain     = st.slider("Rainfall (mm/day)",           0.0, 6.4, 0.0, step=0.1)

    st.markdown("---")
    st.markdown("**Fire Weather Indices**")
    ffmc     = st.slider("FFMC — Fine Fuel Moisture",  18,  96,  78)
    dmc      = st.slider("DMC — Duff Moisture Code",    1,  291,  90)
    drought  = st.slider("DC — Drought Code",           0,  800, 320)

    st.markdown("---")
    run = st.button("🔍 Predict Fire Risk")

    st.markdown("""---
<p style='font-size:0.72rem;color:#4a5568;line-height:1.6'>
FFMC, DMC and DC are standard Canadian Forest Fire Weather Index components used globally for fire behavior prediction.
</p>""", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

# ── LEFT: prediction + importance ────────────────────────────────────────────
with left:
    st.markdown('<div class="section-title">Risk Prediction</div>', unsafe_allow_html=True)

    if run:
        inp  = pd.DataFrame([[temp, humidity, wind, drought, ffmc, dmc, rain]], columns=FEATURES)
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0]
        r    = RISK[pred]
        pct  = int(prob[pred] * 100)

        st.markdown(f"""
        <div class="risk-box {r['css']}">
          <div class="risk-icon">{r['icon']}</div>
          <div class="risk-level" style="color:{r['color']}">{r['label']}</div>
          <div class="risk-pct">{pct}%</div>
          <div class="risk-pct-label">model confidence</div>
          <div class="risk-advice">{r['advice']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        fig = go.Figure(go.Bar(
            x=["Low", "Moderate", "High"],
            y=[p * 100 for p in prob],
            marker_color=["#238636", "#9e6a03", "#da3633"],
            text=[f"{p*100:.1f}%" for p in prob],
            textposition="outside",
            width=0.5,
        ))
        fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#8b949e", family="Inter", size=12),
            yaxis=dict(range=[0, 115], showgrid=False,
                       title="Probability (%)", color="#6e7f8d"),
            xaxis=dict(showgrid=False),
            margin=dict(t=15, b=5, l=5, r=5),
            height=230,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Input summary
        with st.expander("View input summary"):
            summary = pd.DataFrame({
                "Parameter": ["Temperature", "Humidity", "Wind Speed", "Rainfall", "FFMC", "DMC", "DC"],
                "Value": [f"{temp} °C", f"{humidity} %", f"{wind} m/s",
                          f"{rain} mm", str(ffmc), str(dmc), str(drought)]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.markdown("""<div class="info-hint">
          👈 Set weather conditions in the sidebar<br>and click <strong>Predict Fire Risk</strong>
        </div>""", unsafe_allow_html=True)

    # Feature importance
    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
    fi = pd.DataFrame({
        "Feature": ["Temperature", "Humidity", "Drought Code", "FFMC", "Wind Speed", "DMC", "Rainfall"],
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig2 = go.Figure(go.Bar(
        x=fi["Importance"], y=fi["Feature"],
        orientation="h",
        marker=dict(
            color=fi["Importance"],
            colorscale=[[0, "#21262d"], [1, "#ff6b2b"]],
            showscale=False,
        ),
        text=[f"{v:.3f}" for v in fi["Importance"]],
        textposition="outside",
        width=0.6,
    ))
    fig2.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="Inter", size=11),
        xaxis=dict(showgrid=False, title="Contribution to prediction", color="#6e7f8d"),
        yaxis=dict(showgrid=False),
        margin=dict(t=5, b=5, l=5, r=55),
        height=270,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── RIGHT: map + table ────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-title">Kazakhstan Risk Map</div>', unsafe_allow_html=True)

    lats, lons, names, colors, sizes, hover = [], [], [], [], [], []
    cmap = {0: "#238636", 1: "#9e6a03", 2: "#da3633"}
    smap = {0: 12, 1: 16, 2: 22}
    rmap = {0: "Low", 1: "Moderate", 2: "High"}

    for region, (lat, lon, risk) in KZ_REGIONS.items():
        lats.append(lat); lons.append(lon); names.append(region)
        colors.append(cmap[risk]); sizes.append(smap[risk])
        hover.append(f"{region}<br>Risk: {rmap[risk]}")

    fig3 = go.Figure(go.Scattergeo(
        lat=lats, lon=lons,
        text=names,
        hovertext=hover,
        hoverinfo="text",
        mode="markers+text",
        textposition="top center",
        marker=dict(
            size=sizes, color=colors, opacity=0.9,
            line=dict(width=1.5, color="#080c10"),
        ),
        textfont=dict(color="#8b949e", size=8, family="Inter"),
    ))
    fig3.update_geos(
        scope="asia",
        center=dict(lat=48, lon=67),
        projection_scale=3.8,
        showland=True,    landcolor="#111820",
        showocean=True,   oceancolor="#080c10",
        showlakes=True,   lakecolor="#080c10",
        showrivers=True,  rivercolor="#0d1117",
        showcountries=True, countrycolor="#21262d",
        showsubunits=True,  subunitcolor="#1a2230",
        showframe=False,
        bgcolor="#080c10",
    )
    fig3.update_layout(
        paper_bgcolor="#080c10",
        margin=dict(t=0, b=0, l=0, r=0),
        height=340,
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Legend
    st.markdown("""
    <div style="display:flex;gap:1.8rem;margin:0.2rem 0 1rem;font-size:0.82rem">
      <span style="color:#da3633">● High risk</span>
      <span style="color:#9e6a03">● Moderate risk</span>
      <span style="color:#238636">● Low risk</span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Region Status</div>', unsafe_allow_html=True)
    risk_labels = {0: "🟢 Low", 1: "🟡 Moderate", 2: "🔴 High"}
    table_data = pd.DataFrame([
        {"Region": r, "Risk Level": risk_labels[d[2]], "Lat": f"{d[0]}°N", "Lon": f"{d[1]}°E"}
        for r, d in KZ_REGIONS.items()
    ])
    st.dataframe(table_data, use_container_width=True, hide_index=True, height=230)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem">
  <p style="color:#4a5568;font-size:0.78rem;margin:0">
    FireWatch KZ · Built by <strong style="color:#6e7f8d">Nazym Yelubay</strong> · Team <strong style="color:#6e7f8d">2N</strong>
  </p>
  <p style="color:#4a5568;font-size:0.78rem;margin:0">
    SmartScape Hackathon 2026 · STEMNOVA.kz · Ecology & Urban Environment Track
  </p>
</div>
""", unsafe_allow_html=True)
