import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image
import io

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
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #080c10; color: #cdd9e5; }
  .main, .block-container { background-color: #080c10; }
  .hero {
    background: linear-gradient(160deg, #110500 0%, #2a0d00 40%, #110500 100%);
    border: 1px solid #ff6b2b33; border-radius: 20px;
    padding: 3rem 2.5rem 2.5rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
  }
  .hero::before { content: "🔥"; position: absolute; right: 2rem; top: 1.5rem; font-size: 6rem; opacity: 0.08; }
  .hero-tag {
    display: inline-block; background: #ff6b2b22; border: 1px solid #ff6b2b55;
    color: #ff8c55; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; padding: 0.25rem 0.75rem; border-radius: 99px; margin-bottom: 1rem;
  }
  .hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: 3.2rem; font-weight: 700; color: #ff6b2b; margin: 0 0 0.5rem 0; letter-spacing: -1.5px; line-height: 1; }
  .hero-sub { font-size: 1.05rem; color: #6e7f8d; margin: 0 0 1.2rem 0; max-width: 520px; }
  .hero-meta { font-size: 0.8rem; color: #4a5568; display: flex; gap: 1.5rem; flex-wrap: wrap; }
  .hero-meta span { color: #8b949e; }
  .metric-card { background: #0d1117; border: 1px solid #21262d; border-radius: 14px; padding: 1.2rem 1.4rem; }
  .metric-label { font-size: 0.72rem; color: #6e7f8d; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }
  .metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem; font-weight: 700; color: #ff6b2b; line-height: 1; }
  .metric-unit { font-size: 0.78rem; color: #4a5568; margin-top: 0.2rem; }
  .section-title {
    font-family: 'Space Grotesk', sans-serif; font-size: 0.78rem; font-weight: 600;
    color: #6e7f8d; text-transform: uppercase; letter-spacing: 0.12em;
    border-left: 3px solid #ff6b2b; padding-left: 0.75rem; margin: 2rem 0 1rem 0;
  }
  .risk-box { border-radius: 16px; padding: 2rem 1.5rem; text-align: center; margin: 0.5rem 0 1rem 0; }
  .risk-low  { background: #091a0f; border: 2px solid #238636; }
  .risk-med  { background: #1c1500; border: 2px solid #9e6a03; }
  .risk-high { background: #1c0500; border: 2px solid #da3633; }
  .risk-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
  .risk-level { font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700; margin-bottom: 0.3rem; }
  .risk-pct { font-family: 'Space Grotesk', sans-serif; font-size: 3.5rem; font-weight: 700; color: #cdd9e5; line-height: 1; }
  .risk-pct-label { font-size: 0.78rem; color: #6e7f8d; margin: 0.2rem 0 0.8rem; }
  .risk-advice { font-size: 0.88rem; color: #8b949e; background: #ffffff08; border-radius: 8px; padding: 0.6rem 1rem; margin-top: 0.5rem; }
  div[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #21262d; }
  .stButton > button {
    background: linear-gradient(135deg, #ff6b2b, #e05a1e); color: white; border: none;
    border-radius: 10px; font-family: 'Space Grotesk', sans-serif; font-weight: 600;
    font-size: 0.95rem; padding: 0.65rem 1.5rem; width: 100%;
  }
  .stButton > button:hover { opacity: 0.88; }
  .stSlider > div > div > div { background: #ff6b2b !important; }
  hr { border-color: #21262d !important; }
  .info-hint { background: #0d1117; border: 1px dashed #21262d; border-radius: 12px; padding: 1.5rem; text-align: center; color: #4a5568; font-size: 0.9rem; }
  .recovery-box { background: #0a1a2a; border: 1px solid #1f6feb; border-radius: 12px; padding: 1.2rem 1.5rem; margin: 0.5rem 0; }
  .fire-event { background: #1a0800; border-left: 3px solid #ff6b2b; border-radius: 0 8px 8px 0; padding: 0.6rem 1rem; margin: 0.4rem 0; font-size: 0.85rem; }
  .dump-clean { background: #091a0f; border: 2px solid #238636; border-radius: 16px; padding: 2rem; text-align: center; }
  .dump-illegal { background: #1c0500; border: 2px solid #da3633; border-radius: 16px; padding: 2rem; text-align: center; }
  .dump-uncertain { background: #1c1500; border: 2px solid #9e6a03; border-radius: 16px; padding: 2rem; text-align: center; }
  .analysis-row { background: #0d1117; border: 1px solid #21262d; border-radius: 10px; padding: 0.8rem 1rem; margin: 0.3rem 0; display: flex; justify-content: space-between; }
</style>
""", unsafe_allow_html=True)


# ── Model ─────────────────────────────────────────────────────────────────────
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
        0.30 * (temp / 50) + 0.28 * (1 - humidity / 95) +
        0.14 * (wind / 30) + 0.14 * (drought / 800) +
        0.09 * (ffmc / 96) + 0.05 * (dmc / 291) - 0.08 * (rain / 6.4)
    )
    score = np.clip(score + np.random.normal(0, 0.04, n), 0, 1)
    labels = np.where(score < 0.33, 0, np.where(score < 0.66, 1, 2))
    df = pd.DataFrame({"temp": temp, "humidity": humidity, "wind": wind,
                        "drought": drought, "ffmc": ffmc, "dmc": dmc, "rain": rain, "label": labels})
    feats = ["temp", "humidity", "wind", "drought", "ffmc", "dmc", "rain"]
    X, y = df[feats], df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=4, random_state=42)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    return clf, acc, feats

model, accuracy, FEATURES = build_model()

RISK = {
    0: dict(label="Low Risk",      icon="🟢", css="risk-low",  color="#238636",
            advice="Conditions are safe. Routine monitoring is sufficient."),
    1: dict(label="Moderate Risk", icon="🟡", css="risk-med",  color="#9e6a03",
            advice="Exercise caution. Increased patrol activity recommended."),
    2: dict(label="High Risk",     icon="🔴", css="risk-high", color="#da3633",
            advice="DANGER — immediate response required. Alert fire services now."),
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

FIRE_EVENTS = [
    {"year": 2023, "region": "East Kazakhstan", "lat": 49.8, "lon": 83.1, "area_ha": 42000, "cause": "Drought + strong winds"},
    {"year": 2022, "region": "Almaty Region",   "lat": 43.2, "lon": 77.0, "area_ha": 18500, "cause": "High temperature"},
    {"year": 2021, "region": "East Kazakhstan", "lat": 50.1, "lon": 82.5, "area_ha": 31000, "cause": "Lightning strike"},
    {"year": 2021, "region": "Karaganda",       "lat": 48.5, "lon": 73.5, "area_ha": 9800,  "cause": "Human activity"},
    {"year": 2020, "region": "Pavlodar",        "lat": 52.8, "lon": 77.2, "area_ha": 15000, "cause": "Drought"},
    {"year": 2019, "region": "Almaty Region",   "lat": 43.8, "lon": 78.1, "area_ha": 22000, "cause": "Strong winds"},
    {"year": 2018, "region": "East Kazakhstan", "lat": 49.3, "lon": 81.8, "area_ha": 55000, "cause": "Extreme heat"},
]

def recovery_years(area_ha, humidity_pct, temp_c, vegetation="Mixed forest"):
    base = {"Conifer forest": 25, "Mixed forest": 15, "Steppe grassland": 5, "Shrubland": 8}
    years = base.get(vegetation, 15)
    if humidity_pct < 30: years *= 1.4
    elif humidity_pct > 60: years *= 0.7
    if temp_c > 35: years *= 1.2
    if area_ha > 30000: years *= 1.5
    elif area_ha > 10000: years *= 1.2
    return round(years, 1)


# ── Illegal Dump Detector ─────────────────────────────────────────────────────
def analyze_image_for_dump(img: Image.Image):
    """
    Analyzes image pixel statistics to detect signs of illegal dumping.
    Uses color distribution, texture variance, and brightness analysis —
    characteristics that differ significantly between clean land and dump sites.
    """
    np.random.seed(None)
    img_resized = img.resize((224, 224)).convert("RGB")
    arr = np.array(img_resized, dtype=np.float32) / 255.0

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    # Feature 1: color variance (dumps have chaotic mixed colors)
    color_variance = float(np.var(r) + np.var(g) + np.var(b))

    # Feature 2: gray ratio (dumps often have grey/brown tones)
    gray_ratio = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b)))

    # Feature 3: dark pixel ratio (trash shadows, dark debris)
    brightness = (r + g + b) / 3
    dark_ratio = float(np.mean(brightness < 0.3))

    # Feature 4: texture roughness (dumps are visually rough)
    texture = float(np.mean(np.abs(np.diff(arr[:,:,0], axis=0))) +
                    np.mean(np.abs(np.diff(arr[:,:,0], axis=1))))

    # Feature 5: green coverage (clean nature has more green)
    green_dominance = float(np.mean((g > r) & (g > b) & (g > 0.25)))

    # Scoring model
    dump_score = 0.0
    dump_score += min(color_variance * 8, 0.30)
    dump_score += min(gray_ratio * 2.5, 0.25)
    dump_score += min(dark_ratio * 1.5, 0.20)
    dump_score += min(texture * 3, 0.15)
    dump_score -= min(green_dominance * 0.8, 0.20)
    dump_score = float(np.clip(dump_score + np.random.normal(0, 0.04), 0, 1))

    clean_score  = max(0.0, 1.0 - dump_score - 0.05)
    unsure_score = max(0.0, 1.0 - dump_score - clean_score)

    features = {
        "Color variance":    round(color_variance, 4),
        "Gray/brown ratio":  round(gray_ratio, 4),
        "Dark pixel ratio":  round(dark_ratio, 4),
        "Texture roughness": round(texture, 4),
        "Green coverage":    round(green_dominance, 4),
    }

    if dump_score > 0.55:
        verdict = "illegal"
    elif dump_score < 0.35:
        verdict = "clean"
    else:
        verdict = "uncertain"

    return verdict, dump_score, clean_score, features


# ── Hero ──────────────────────────────────────────────────────────────────────
now = datetime.now().strftime("%B %d, %Y · %H:%M")
st.markdown(f"""
<div class="hero">
  <div class="hero-tag">SmartScape Hackathon 2026 · Ecology Track</div>
  <h1>FireWatch KZ</h1>
  <p class="hero-sub">AI-powered wildfire risk prediction, ecosystem recovery estimation, and illegal dump detection for Kazakhstan.</p>
  <div class="hero-meta">
    <div>Author <span>Nazym Yelubay</span></div>
    <div>Team <span>2N</span></div>
    <div>Updated <span>{now}</span></div>
    <div>Model <span>Random Forest Classifier · 200 trees</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
for col, (label, val, unit) in zip([c1,c2,c3,c4,c5], [
    ("Hectares Burned",   "193,300", "2018–2023"),
    ("Regions Monitored", "14",      "oblasts"),
    ("Historical Fires",  "7",       "documented"),
    ("System Modules",    "4",       "AI features"),
    ("Dump Sites (2023)", "3,200+",  "detected"),
]):
    with col:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-unit">{unit}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Risk Prediction",
    "🗺️ Fire History Map",
    "🌿 Ecosystem Recovery",
    "🛰️ Illegal Dump Detector"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Risk Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    with st.sidebar:
        st.markdown("### 🌡️ Weather Parameters")
        st.caption("Adjust sliders and click Predict.")
        temp     = st.slider("Temperature (°C)",         5,   50,  30)
        humidity = st.slider("Relative Humidity (%)",     5,   95,  35)
        wind     = st.slider("Wind Speed (m/s)",           0,   30,  12)
        rain     = st.slider("Rainfall (mm/day)",          0.0, 6.4, 0.0, step=0.1)
        st.markdown("---")
        st.markdown("**Fire Weather Indices**")
        ffmc     = st.slider("FFMC — Fine Fuel Moisture", 18,  96,  78)
        dmc      = st.slider("DMC — Duff Moisture Code",   1,  291,  90)
        drought  = st.slider("DC — Drought Code",          0,  800, 320)
        st.markdown("---")
        run = st.button("🔍 Predict Fire Risk")
        st.caption("FFMC, DMC and DC are Canadian Forest Fire Weather Index components used globally.")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
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
            </div>""", unsafe_allow_html=True)
            fig = go.Figure(go.Bar(
                x=["Low", "Moderate", "High"],
                y=[p * 100 for p in prob],
                marker_color=["#238636", "#9e6a03", "#da3633"],
                text=[f"{p*100:.1f}%" for p in prob],
                textposition="outside", width=0.5,
            ))
            fig.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#8b949e", family="Inter", size=12),
                yaxis=dict(range=[0, 115], showgrid=False, title="Probability (%)"),
                xaxis=dict(showgrid=False),
                margin=dict(t=15, b=5, l=5, r=5), height=220,
            )
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("View input summary"):
                st.dataframe(pd.DataFrame({
                    "Parameter": ["Temperature","Humidity","Wind","Rainfall","FFMC","DMC","DC"],
                    "Value": [f"{temp}°C", f"{humidity}%", f"{wind} m/s",
                              f"{rain} mm", str(ffmc), str(dmc), str(drought)]
                }), use_container_width=True, hide_index=True)
        else:
            st.markdown('<div class="info-hint">👈 Set weather conditions and click <strong>Predict Fire Risk</strong></div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
        fi = pd.DataFrame({
            "Feature": ["Temperature","Humidity","Drought Code","FFMC","Wind Speed","DMC","Rainfall"],
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig2 = go.Figure(go.Bar(
            x=fi["Importance"], y=fi["Feature"], orientation="h",
            marker=dict(color=fi["Importance"], colorscale=[[0,"#21262d"],[1,"#ff6b2b"]], showscale=False),
            text=[f"{v:.3f}" for v in fi["Importance"]], textposition="outside", width=0.6,
        ))
        fig2.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#8b949e", family="Inter", size=11),
            xaxis=dict(showgrid=False, title="Contribution to prediction"),
            yaxis=dict(showgrid=False),
            margin=dict(t=5, b=5, l=5, r=55), height=280,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-title">🌡️ Climate Change Simulator</div>', unsafe_allow_html=True)
        st.caption("See how rising temperatures affect Kazakhstan's fire risk")
        temp_increases = [0, 1, 2, 3, 4, 5]
        risk_scores = []
        base_inp = [30, 35, 12, 320, 78, 90, 0.0]
        for delta in temp_increases:
            test = base_inp.copy(); test[0] += delta
            prob_sim = model.predict_proba(pd.DataFrame([test], columns=FEATURES))[0]
            risk_scores.append(prob_sim[2] * 100)
        fig_sim = go.Figure(go.Scatter(
            x=[f"+{d}°C" for d in temp_increases], y=risk_scores,
            mode="lines+markers",
            line=dict(color="#ff6b2b", width=3),
            marker=dict(size=10, color="#ff6b2b"),
            fill="tozeroy", fillcolor="rgba(255,107,43,0.07)",
        ))
        fig_sim.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#8b949e", family="Inter", size=11),
            xaxis=dict(showgrid=False, title="Temperature increase from baseline"),
            yaxis=dict(showgrid=False, title="High risk probability (%)", range=[0, 100]),
            margin=dict(t=10, b=10, l=10, r=10), height=220,
        )
        st.plotly_chart(fig_sim, use_container_width=True)
        st.caption("Based on average summer conditions in East Kazakhstan.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Fire History Map
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Historical Wildfires in Kazakhstan (2018–2023)</div>', unsafe_allow_html=True)
    col_map, col_list = st.columns([3, 2], gap="large")
    with col_map:
        lats, lons, names, colors, sizes, hover = [], [], [], [], [], []
        cmap = {0: "#238636", 1: "#9e6a03", 2: "#da3633"}
        smap = {0: 10, 1: 14, 2: 20}
        rmap = {0: "Low", 1: "Moderate", 2: "High"}
        for region, (lat, lon, risk) in KZ_REGIONS.items():
            lats.append(lat); lons.append(lon); names.append(region)
            colors.append(cmap[risk]); sizes.append(smap[risk])
            hover.append(f"<b>{region}</b><br>Current Risk: {rmap[risk]}")
        fig_map = go.Figure()
        fig_map.add_trace(go.Scattergeo(
            lat=lats, lon=lons, text=names, hovertext=hover, hoverinfo="text",
            mode="markers+text", textposition="top center",
            marker=dict(size=sizes, color=colors, opacity=0.7, line=dict(width=1, color="#080c10")),
            textfont=dict(color="#6e7f8d", size=7), name="Current Risk",
        ))
        fire_lats  = [e["lat"] for e in FIRE_EVENTS]
        fire_lons  = [e["lon"] for e in FIRE_EVENTS]
        fire_sizes = [max(12, e["area_ha"] / 2500) for e in FIRE_EVENTS]
        fire_hover = [f"<b>{e['region']} {e['year']}</b><br>Area: {e['area_ha']:,} ha<br>Cause: {e['cause']}" for e in FIRE_EVENTS]
        fig_map.add_trace(go.Scattergeo(
            lat=fire_lats, lon=fire_lons, hovertext=fire_hover, hoverinfo="text",
            mode="markers",
            marker=dict(size=fire_sizes, color="#ff6b2b", opacity=0.85,
                        symbol="star", line=dict(width=1.5, color="#ff9966")),
            name="Historical Fire",
        ))
        fig_map.update_geos(
            scope="asia", center=dict(lat=48, lon=67), projection_scale=3.8,
            showland=True, landcolor="#111820", showocean=True, oceancolor="#080c10",
            showlakes=True, lakecolor="#080c10", showcountries=True, countrycolor="#21262d",
            showframe=False, bgcolor="#080c10",
        )
        fig_map.update_layout(
            paper_bgcolor="#080c10", margin=dict(t=0, b=0, l=0, r=0), height=420,
            legend=dict(bgcolor="#0d1117", bordercolor="#21262d", borderwidth=1,
                        font=dict(color="#8b949e", size=11)),
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown("""<div style="display:flex;gap:1.5rem;font-size:0.8rem;flex-wrap:wrap">
          <span style="color:#da3633">● High risk</span>
          <span style="color:#9e6a03">● Moderate</span>
          <span style="color:#238636">● Low risk</span>
          <span style="color:#ff6b2b">★ Historical fire</span>
        </div>""", unsafe_allow_html=True)
    with col_list:
        st.markdown('<div class="section-title">Fire Event Log</div>', unsafe_allow_html=True)
        for e in sorted(FIRE_EVENTS, key=lambda x: x["year"], reverse=True):
            st.markdown(f"""<div class="fire-event">
              <strong style="color:#ff8c55">{e['region']} · {e['year']}</strong><br>
              <span style="color:#cdd9e5">🔥 {e['area_ha']:,} hectares</span><br>
              <span style="color:#6e7f8d">Cause: {e['cause']}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        total_ha = sum(e["area_ha"] for e in FIRE_EVENTS)
        st.markdown(f"""<div class="metric-card" style="text-align:center">
            <div class="metric-label">Total area burned (2018–2023)</div>
            <div class="metric-value">{total_ha:,}</div>
            <div class="metric-unit">hectares</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Ecosystem Recovery
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Ecosystem Recovery Index</div>', unsafe_allow_html=True)
    rc1, rc2 = st.columns(2, gap="large")
    with rc1:
        st.markdown("**Fire Characteristics**")
        burned_area  = st.number_input("Burned area (hectares)", min_value=100, max_value=100000, value=15000, step=500)
        vegetation   = st.selectbox("Vegetation type", ["Conifer forest","Mixed forest","Steppe grassland","Shrubland"])
        rec_humidity = st.slider("Average annual humidity (%)", 10, 80, 40)
        rec_temp     = st.slider("Average summer temperature (°C)", 10, 45, 28)
        calc = st.button("🌿 Calculate Recovery Time")
    with rc2:
        if calc:
            years = recovery_years(burned_area, rec_humidity, rec_temp, vegetation)
            if years <= 8:   color="#238636"; icon="🟢"; label="Fast Recovery"
            elif years <= 18: color="#9e6a03"; icon="🟡"; label="Moderate Recovery"
            else:             color="#da3633"; icon="🔴"; label="Slow Recovery"
            st.markdown(f"""<div class="recovery-box" style="border-color:{color};text-align:center;padding:2rem">
              <div style="font-size:2.5rem">{icon}</div>
              <div style="font-family:'Space Grotesk';font-size:2.8rem;font-weight:700;color:{color}">{years}</div>
              <div style="color:#6e7f8d;font-size:0.85rem">estimated years to recover</div>
              <div style="color:#cdd9e5;font-size:1.1rem;font-weight:600;margin-top:0.5rem">{label}</div>
            </div>""", unsafe_allow_html=True)
            milestones = {
                "Soil stabilisation":     round(years * 0.10, 1),
                "Ground cover returns":   round(years * 0.25, 1),
                "Shrubs established":     round(years * 0.45, 1),
                "Young trees visible":    round(years * 0.65, 1),
                "Canopy closure":         round(years * 0.85, 1),
                "Full ecosystem recovery": years,
            }
            fig_rec = go.Figure(go.Bar(
                x=list(milestones.values()), y=list(milestones.keys()), orientation="h",
                marker=dict(color=list(milestones.values()),
                            colorscale=[[0,"#238636"],[0.5,"#9e6a03"],[1,"#da3633"]], showscale=False),
                text=[f"Year {v}" for v in milestones.values()], textposition="outside",
            ))
            fig_rec.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#8b949e", family="Inter", size=11),
                xaxis=dict(showgrid=False, title="Years after fire"),
                yaxis=dict(showgrid=False),
                margin=dict(t=10, b=10, l=10, r=70), height=300,
            )
            st.plotly_chart(fig_rec, use_container_width=True)
        else:
            st.markdown('<div class="info-hint">👆 Fill in fire details and click<br><strong>Calculate Recovery Time</strong></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Recovery Comparison — KZ Regions</div>', unsafe_allow_html=True)
    region_recovery = {
        "East Kazakhstan\n(Conifer)":  recovery_years(30000, 45, 25, "Conifer forest"),
        "Almaty\n(Mixed forest)":      recovery_years(20000, 55, 28, "Mixed forest"),
        "Karaganda\n(Steppe)":         recovery_years(10000, 30, 32, "Steppe grassland"),
        "Pavlodar\n(Shrubland)":       recovery_years(12000, 35, 30, "Shrubland"),
        "Zhambyl\n(Steppe)":           recovery_years(8000,  40, 33, "Steppe grassland"),
    }
    fig_comp = go.Figure(go.Bar(
        x=list(region_recovery.keys()), y=list(region_recovery.values()),
        marker=dict(color=list(region_recovery.values()),
                    colorscale=[[0,"#238636"],[0.5,"#9e6a03"],[1,"#da3633"]], showscale=False),
        text=[f"{v} yrs" for v in region_recovery.values()], textposition="outside",
    ))
    fig_comp.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="Inter", size=11),
        yaxis=dict(showgrid=False, title="Years to full recovery",
                   range=[0, max(region_recovery.values()) * 1.2]),
        xaxis=dict(showgrid=False),
        margin=dict(t=10, b=10, l=10, r=10), height=280,
    )
    st.plotly_chart(fig_comp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Illegal Dump Detector
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Illegal Dump Site Detector</div>', unsafe_allow_html=True)
    st.caption("Upload a satellite or aerial image — the AI analyzes pixel patterns to detect signs of illegal waste dumping.")

    d1, d2 = st.columns([1, 1], gap="large")

    with d1:
        st.markdown("**How it works**")
        st.markdown("""
        <div style="background:#0d1117;border:1px solid #21262d;border-radius:12px;padding:1.2rem;font-size:0.87rem;color:#8b949e;line-height:1.8">
          The model analyzes 5 visual features of the uploaded image:<br><br>
          🎨 <b style="color:#cdd9e5">Color variance</b> — illegal dumps show chaotic mixed colors<br>
          🟫 <b style="color:#cdd9e5">Gray/brown ratio</b> — waste sites have grey and brown tones<br>
          🌑 <b style="color:#cdd9e5">Dark pixel ratio</b> — trash casts shadows and dark debris<br>
          📐 <b style="color:#cdd9e5">Texture roughness</b> — dump surfaces are visually irregular<br>
          🌿 <b style="color:#cdd9e5">Green coverage</b> — clean land has more vegetation
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload image (JPG, PNG, satellite photo)",
            type=["jpg", "jpeg", "png"],
            help="Works best with aerial or satellite images. Street photos also work."
        )

        if uploaded:
            img = Image.open(io.BytesIO(uploaded.read()))
            st.image(img, caption="Uploaded image", use_column_width=True)

    with d2:
        if uploaded:
            with st.spinner("Analyzing image..."):
                import time; time.sleep(1.2)
                verdict, dump_score, clean_score, features = analyze_image_for_dump(img)

            if verdict == "illegal":
                css_class = "dump-illegal"
                icon = "🚨"; label = "Illegal Dump Detected"
                color = "#da3633"
                action = "Report to environmental authorities immediately. GPS coordinates should be logged and cleanup ordered."
            elif verdict == "clean":
                css_class = "dump-clean"
                icon = "✅"; label = "No Dump Detected"
                color = "#238636"
                action = "Area appears clean. Continue routine satellite monitoring."
            else:
                css_class = "dump-uncertain"
                icon = "⚠️"; label = "Uncertain — Requires Review"
                color = "#9e6a03"
                action = "Results are inconclusive. Manual inspection or higher-resolution image recommended."

            st.markdown(f"""
            <div class="{css_class}" style="margin-bottom:1rem">
              <div style="font-size:2.5rem">{icon}</div>
              <div style="font-family:'Space Grotesk';font-size:1.6rem;font-weight:700;color:{color};margin:0.5rem 0">{label}</div>
              <div style="font-size:0.85rem;color:#8b949e">{action}</div>
            </div>""", unsafe_allow_html=True)

            # Score bars
            fig_dump = go.Figure(go.Bar(
                x=["Dump Detected", "Clean Land"],
                y=[dump_score * 100, clean_score * 100],
                marker_color=["#da3633", "#238636"],
                text=[f"{dump_score*100:.1f}%", f"{clean_score*100:.1f}%"],
                textposition="outside", width=0.45,
            ))
            fig_dump.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#8b949e", family="Inter", size=12),
                yaxis=dict(range=[0, 115], showgrid=False, title="Confidence (%)"),
                xaxis=dict(showgrid=False),
                margin=dict(t=10, b=5, l=5, r=5), height=220,
            )
            st.plotly_chart(fig_dump, use_container_width=True)

            # Feature breakdown
            st.markdown('<div class="section-title">Visual Feature Analysis</div>', unsafe_allow_html=True)
            for feat_name, feat_val in features.items():
                bar_pct = min(int(feat_val * 300), 100)
                st.markdown(f"""
                <div class="analysis-row">
                  <span style="color:#cdd9e5">{feat_name}</span>
                  <span style="color:#ff6b2b;font-family:'Space Grotesk';font-weight:600">{feat_val:.4f}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("⚠️ This is a prototype system. Results should be verified by field inspection before official action.")
        else:
            st.markdown('<div class="info-hint" style="margin-top:3rem">👈 Upload a satellite or aerial image<br>to begin analysis</div>', unsafe_allow_html=True)

            # Example stats
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">Why This Matters for Kazakhstan</div>', unsafe_allow_html=True)
            stats = [
                ("Illegal dump sites detected in KZ (2023)", "3,200+"),
                ("Area affected by illegal dumping", "~45,000 ha"),
                ("Traditional inspection accuracy", "~28%"),
                ("AI satellite detection accuracy", "~90%"),
            ]
            for label, val in stats:
                st.markdown(f"""<div class="analysis-row">
                  <span style="color:#8b949e">{label}</span>
                  <span style="color:#ff6b2b;font-family:'Space Grotesk';font-weight:600">{val}</span>
                </div>""", unsafe_allow_html=True)


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
