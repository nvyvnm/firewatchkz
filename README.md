# 🔥 FireWatch KZ

**AI-powered wildfire risk prediction system for Kazakhstan**

> SmartScape Hackathon 2026 · Track: Ecology & Urban Environment · STEMNOVA.kz
> **Author:** Nazym Yelubay  |  **Team:** 2N

-----

## Overview

FireWatch KZ is a machine learning web application that predicts the risk of wildfire ignition based on real-time meteorological and fire weather index data. The system is designed specifically for the geographic and climatic conditions of Kazakhstan — one of Central Asia’s most fire-prone countries — and gives emergency services, forest rangers, and environmental agencies an instant, data-driven risk assessment before a fire starts.

The application takes 7 weather parameters as input, runs them through a trained Random Forest classifier, and outputs one of three risk levels (Low / Moderate / High) along with a confidence score. Results are visualized on an interactive map of Kazakhstan’s 14 regions, with clear action recommendations for each risk level.

-----

## Why This Matters

Kazakhstan has over 13 million hectares of forested land. Every year, wildfires destroy tens of thousands of hectares — particularly in the **East Kazakhstan** and **Almaty** regions, where dry continental summers, strong winds, and prolonged droughts create ideal fire conditions.

The core problem is not firefighting — it’s **detection lag**. By the time a fire is reported, it has often already spread beyond control. Studies on fire weather systems show that even a 30-minute improvement in early warning can reduce burned area by up to 40%.

FireWatch KZ addresses this by shifting the focus from reaction to **prevention**: instead of waiting for fire to appear, the system continuously assesses environmental risk based on conditions known to precede wildfire ignition.

**This matters beyond Kazakhstan too.** The Fire Weather Index (FWI) system used in this project is a globally recognized standard developed by the Canadian Forest Service and adopted by fire agencies in over 40 countries. The architecture of FireWatch KZ is portable — with local data, it can be adapted to any region.

-----

## Features

- **Real-time risk prediction** — input current weather conditions and get an immediate Low / Moderate / High risk classification with confidence percentage
- **Interactive Kazakhstan map** — 14 regions color-coded by risk level with hover tooltips showing region name and status
- **Feature importance chart** — horizontal bar chart showing which weather factors contribute most to the model’s decision
- **Probability distribution** — bar chart showing the model’s confidence across all three risk classes, not just the predicted one
- **Input summary panel** — expandable section showing every parameter used in the current prediction
- **Region status table** — sortable table with risk level, latitude, and longitude for all 14 oblasts
- **Live timestamp** — hero section updates with current date and time on each load

-----

## How the Model Works

### Algorithm: Random Forest Classifier

A Random Forest is an ensemble method that builds multiple decision trees during training and outputs the class that receives the most votes. It was chosen for this project because:

- It handles non-linear relationships between weather variables and fire risk well
- It is robust to outliers in weather data (e.g. sudden temperature spikes)
- It provides built-in feature importance scores without additional computation
- It achieves high accuracy with relatively small training sets

### Training Data

The model was trained on a synthetic dataset of 1,200 samples generated to reflect the real statistical distribution of fire weather conditions in Central Asia. Each sample includes 7 features and a risk label (0 = Low, 1 = Moderate, 2 = High) derived from a weighted risk score:

```
risk_score = 0.30 × (temp/50)
           + 0.28 × (1 − humidity/95)
           + 0.14 × (wind/30)
           + 0.14 × (drought/800)
           + 0.09 × (ffmc/96)
           + 0.05 × (dmc/291)
           − 0.08 × (rain/6.4)
```

Gaussian noise (σ = 0.04) was added to simulate real-world measurement variability. The dataset was split 80/20 for training and testing.

### Model Configuration

|Parameter        |Value                   |
|-----------------|------------------------|
|Algorithm        |Random Forest Classifier|
|n_estimators     |200                     |
|max_depth        |12                      |
|min_samples_split|4                       |
|random_state     |42                      |
|Test accuracy    |~92%+                   |
|Training samples |960                     |
|Test samples     |240                     |

-----

## Input Parameters

|Parameter        |Description                                                      |Unit |Range  |
|-----------------|-----------------------------------------------------------------|-----|-------|
|Temperature      |Air temperature at 2m height                                     |°C   |5 – 50 |
|Relative Humidity|Moisture content of the air                                      |%    |5 – 95 |
|Wind Speed       |Surface wind speed                                               |m/s  |0 – 30 |
|Rainfall         |Total precipitation in past 24h                                  |mm   |0 – 6.4|
|FFMC             |Fine Fuel Moisture Code — moisture of surface litter             |index|18 – 96|
|DMC              |Duff Moisture Code — moisture of loosely compacted organic layers|index|1 – 291|
|DC               |Drought Code — moisture of deep, compact organic layers          |index|0 – 800|

**About FWI Indices**

FFMC, DMC, and DC are the three fuel moisture components of the **Canadian Forest Fire Weather Index (FWI) System**, an internationally standardized framework used by fire management agencies in over 40 countries. Higher FFMC values indicate drier surface fuels that ignite more easily. Higher DC values reflect prolonged drought and indicate deep fire penetration potential. These indices are calculated daily from standard weather station observations.

-----

## Risk Levels

|Level     |Meaning                                                             |Recommended Action                                     |
|----------|--------------------------------------------------------------------|-------------------------------------------------------|
|🟢 Low     |Conditions are unlikely to support ignition or spread               |Routine monitoring                                     |
|🟡 Moderate|Elevated risk — fire could start and spread under certain conditions|Increased patrol, public advisories                    |
|🔴 High    |Dangerous conditions — rapid ignition and spread likely             |Alert fire services, restrict access to high-risk zones|

-----

## Tech Stack

|Layer        |Tool                |Version       |
|-------------|--------------------|--------------|
|Language     |Python              |3.10+         |
|ML framework |scikit-learn        |1.4.2         |
|Web interface|Streamlit           |1.35.0        |
|Data handling|pandas, NumPy       |2.2.2 / 1.26.4|
|Visualization|Plotly              |5.22.0        |
|Environment  |Google Colab / local|—             |

-----

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip package manager
- A modern web browser

### Option A — Run locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/firewatchkz.git
cd firewatchkz

# 2. (Optional but recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

### Option B — Run in Google Colab

```python
# Cell 1 — install dependencies
!pip install streamlit pyngrok -q

# Cell 2 — write app to file
%%writefile app.py
# paste the full contents of app.py here

# Cell 3 — launch with public URL
from pyngrok import ngrok
import subprocess, threading, time

threading.Thread(
    target=lambda: subprocess.run(["streamlit", "run", "app.py"])
).start()

time.sleep(4)
public_url = ngrok.connect(8501)
print("Open app at:", public_url)
```

-----

## Project Structure

```
firewatchkz/
├── app.py              # Main Streamlit application (model + UI)
├── requirements.txt    # Python dependencies with pinned versions
└── README.md           # Project documentation (this file)
```

-----

## Real-World Relevance

FireWatch KZ aligns directly with **UN Sustainable Development Goal 15 — Life on Land**, which calls for the protection, restoration, and sustainable management of terrestrial ecosystems.

Practically, a system like this could be integrated into:

- Kazakhstan’s **Committee of Forestry and Wildlife** monitoring dashboards
- Municipal emergency response platforms in high-risk oblasts
- Weather station networks already collecting FWI data daily
- School and community awareness tools in rural fire-prone areas

The lightweight architecture (no GPU required, runs in a browser, deployable on any server) makes it realistic to deploy even in regions with limited infrastructure.

-----

## Limitations & Future Work

|Limitation               |Planned Improvement                                              |
|-------------------------|-----------------------------------------------------------------|
|Synthetic training data  |Replace with real historical fire records from KazHydroMet       |
|Static region risk map   |Connect to live weather API for real-time regional updates       |
|No spatial resolution    |Add grid-level predictions using satellite vegetation data (NDVI)|
|No fire spread simulation|Integrate wind direction for spread prediction                   |
|Web only                 |Build mobile-friendly version for field rangers                  |

-----

## License

This project was built for educational and competition purposes at SmartScape Hackathon 2026. All code is open source and free to use with attribution.

-----

## Author

**Nazym Yelubay** · Team 2N
SmartScape Hackathon 2026 · STEMNOVA.kz
📧 [elubainazim@gmail.com](mailto:elubainazim@gmail.com)