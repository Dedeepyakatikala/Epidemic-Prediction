# 🦠 Hybrid Epidemic Prediction System

> An AI-driven framework for forecasting epidemic spread and risk levels across Indian states — combining deep learning, probabilistic modeling, and spatial analysis.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-Dashboard-61DAFB?logo=react&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

The **Hybrid Epidemic Prediction System** is an AI-powered framework that forecasts epidemic spread and risk levels across Indian states using a multi-model hybrid approach:

- 🔁 **Deep Learning (LSTM)** — Temporal trend forecasting
- 📊 **Markov Logistic Modeling** — Probabilistic risk transitions
- 🎲 **Bayesian Inference (Gibbs Sampling)** — Uncertainty quantification
- 🗺️ **Spatial Analysis** — State adjacency network modeling

Developed using COVID-19 data from India, the system provides **state-level predictions** of epidemic risk, active cases, and potential future outbreaks, while incorporating uncertainty estimation and neighboring state influence.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📈 **Temporal Forecasting** | LSTM-based prediction of future active cases from historical epidemic trends |
| 🗺️ **Spatial Modeling** | State adjacency matrix captures geographic spillover effects |
| 🎯 **Probabilistic Risk Prediction** | Markov model estimates transitions between low-risk and high-risk states |
| 📉 **Bayesian Uncertainty Estimation** | Gibbs Sampling produces credible intervals instead of single-point forecasts |
| 🖥️ **Interactive Dashboard** | State-wise monitoring, risk visualization, forecast plots, and neighbor analysis |

---

## 🏗️ Architecture

```
Raw COVID Data
       │
       ▼
Data Preprocessing
       │
       ▼
Merged Panel Dataset
       │
 ┌─────┴─────┐
 ▼           ▼
LSTM      Adjacency Matrix
 ▼           ▼
Trend     Neighbor Influence
    └─────┬─────┘
          ▼
 Markov Logistic Model
          ▼
    Gibbs Sampling
          ▼
 Risk Probabilities
 Case Forecasts
 Credible Intervals
          ▼
      Dashboard
```

---

## 📂 Repository Structure

```
Epidemic-Prediction/
│
├── data_raw/
│   ├── covid_19_india.csv               # Confirmed, recovered, death counts
│   ├── covid_vaccine_statewise.csv      # Daily & total vaccinations
│   ├── StatewiseTestingDetails.csv      # Daily tests, positivity rates
│   └── shapefiles/                      # Indian state boundary shapefiles
│
├── data_curated/
│   ├── panel_state_daily.csv
│   ├── panel_for_model.csv
│   ├── state_adjacency.csv
│   └── prophet_oneweek_forecast.csv
│
├── figs/
│   ├── active_distribution.png
│   ├── top_states_active.png
│   └── ...
│
├── models/
│   ├── global_lstm.h5
│   ├── global_lstm.keras
│   ├── markov_logistic.pkl
│   └── markov_scaler.pkl
│
├── notebooks/
│
├── results/
│   └── predictions_with_truth.csv
│
├── src/
│   ├── api.py
│   ├── predictors.py
│   └── dashboard/
│       └── dash_app.py
│
└── README.md
```

---

## 📊 Dataset Sources

### 🦠 COVID-19 Case Data
- **File:** `data_raw/covid_19_india.csv`
- **Contains:** Confirmed cases, recovered cases, death counts

### 💉 Vaccination Data
- **File:** `data_raw/covid_vaccine_statewise.csv`
- **Contains:** Daily vaccinations, total vaccinations, vaccination rate

### 🧪 Testing Data
- **File:** `data_raw/StatewiseTestingDetails.csv`
- **Contains:** Daily tests, total tests, positivity rates

### 🗺️ Spatial Data
- **Files:** `data_raw/shapefiles/`
- **Contains:** Indian state boundary shapefiles for adjacency modeling

---

## ⚙️ Data Preprocessing

The preprocessing pipeline handles:

- State name standardization
- Date formatting
- Missing value imputation
- Active case computation
- Rolling averages & growth rate calculation
- Risk indicator generation
- Neighbor risk computation
- Dataset merging

**Generated datasets in `data_curated/`:**

```
data_curated/
├── panel_state_daily.csv
├── panel_for_model.csv
├── state_adjacency.csv
└── prophet_oneweek_forecast.csv
```

---

## 🧠 Model Components

### 1. LSTM Network

> Learns epidemic trends over time and predicts future active cases.

**Input Features:**

| Feature | Description |
|---|---|
| `active` | Current active case count |
| `new_confirmed` | Daily new confirmed cases |
| `daily_tests` | Daily test count |
| `daily_vax` | Daily vaccinations administered |
| `vax_rate` | Vaccination rate |

**Configuration:**
- Lookback Window: **14 days**
- LSTM Units: **64**
- Region Embedding Size: **8**
- Dense layers with Dropout

---

### 2. Markov Logistic Risk Model

> Estimates future epidemic risk using a probabilistic state-transition approach.

**Inputs:**
- Previous risk state
- Neighbor risk (from adjacency matrix)
- LSTM trend output
- Testing rate
- Vaccination rate

**Output:** Probability of transitioning to a high-risk state

---

### 3. Gibbs Sampling (Bayesian Inference)

> Generates posterior distributions for risk uncertainty estimation.

**Produces:**
- Posterior risk probabilities
- Multi-state epidemic scenarios
- Credible intervals

---

## 📈 Results

The hybrid framework successfully:

- ✅ Captured temporal epidemic trends
- ✅ Modeled spatial spillover between neighboring states
- ✅ Generated uncertainty-aware forecasts
- ✅ Produced state-level risk probabilities
- ✅ Identified potential future outbreak regions

### Key Findings

> 🔗 **Neighboring states significantly affect epidemic propagation.**
>
> 💉 **Higher vaccination coverage reduces future risk.**
>
> 🤝 **Combining deep learning with probabilistic modeling improves both interpretability and forecasting reliability.**

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow, Keras |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn |
| **Spatial Analysis** | GeoPandas |
| **Visualization** | Matplotlib |
| **Backend API** | FastAPI |
| **Frontend** | React |
| **Probabilistic Modeling** | Bayesian Inference, Markov Models |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow pandas numpy scikit-learn geopandas matplotlib fastapi uvicorn
```

### Running the API

```bash
cd src
uvicorn api:app --reload
```

### Running the Dashboard

```bash
cd src/dashboard
python dash_app.py
```

---




<p align="center">Built with ❤️ for epidemic intelligence and public health forecasting</p>
