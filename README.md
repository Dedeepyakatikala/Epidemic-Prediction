Hybrid Epidemic Prediction System
Overview

The Hybrid Epidemic Prediction System is an AI-driven framework designed to forecast epidemic spread and risk levels across Indian states using a combination of:

Deep Learning (LSTM)
Probabilistic Markov Modeling
Bayesian Inference using Gibbs Sampling
Spatial Analysis using State Adjacency Networks

The project was developed using COVID-19 data from India and aims to provide state-level predictions of epidemic risk, active cases, and potential future outbreaks while incorporating uncertainty estimation and neighboring state influence.
Key Features
Temporal Forecasting
LSTM-based prediction of future active cases
Learns trends from historical epidemic data
Uses testing and vaccination information as additional signals
Spatial Modeling
State adjacency matrix captures geographic relationships
Neighboring states influence epidemic risk estimation
Probabilistic Risk Prediction
Markov Logistic Model estimates epidemic risk transitions
Predicts probability of states moving from low-risk to high-risk conditions
Bayesian Uncertainty Estimation
Gibbs Sampling generates posterior distributions
Produces credible intervals instead of single-point forecasts
Interactive Dashboard
State-wise epidemic monitoring
Risk visualization
Forecast plots
Neighbor influence analysis

Project Architecture
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

Dataset Sources
COVID-19 Case Data

Contains:

Confirmed cases
Recovered cases
Death counts

File:

data_raw/covid_19_india.csv
Vaccination Data

Contains:

Daily vaccinations
Total vaccinations
Vaccination rate

File:

data_raw/covid_vaccine_statewise.csv
Testing Data

Contains:

Daily tests
Total tests
Positivity rates

File:

data_raw/StatewiseTestingDetails.csv
Spatial Data

Indian state boundary shapefiles used to construct state adjacency relationships.

Files:

data_raw/shapefiles/

Data Preprocessing

The preprocessing pipeline includes:

State name standardization
Date formatting
Missing value handling
Active case computation
Rolling averages
Growth rate calculation
Risk indicator generation
Neighbor risk computation
Dataset merging

Generated datasets:

data_curated/
├── panel_state_daily.csv
├── panel_for_model.csv
├── state_adjacency.csv
└── prophet_oneweek_forecast.csv

Model Components
1. LSTM Network

Purpose:

Learn epidemic trends over time
Predict future active cases

Input Features:

active
new_confirmed
daily_tests
daily_vax
vax_rate

Configuration:

Lookback Window: 14 days
LSTM Units: 64
Region Embedding Size: 8
Dense Layers with Dropout

Output:

Predicted active case trend

2. Markov Logistic Risk Model

Purpose:

Estimate future epidemic risk using:

Previous risk state
Neighbor risk
LSTM trend
Testing rate
Vaccination rate

Output:

Probability of high-risk state

3. Gibbs Sampling

Purpose:

Bayesian posterior inference
Risk uncertainty estimation

Produces:

Posterior risk probabilities
Multi-state epidemic scenarios
Credible intervals

Repository Structure
Epidemic-Prediction/
│
├── data_raw/
│   ├── covid_19_india.csv
│   ├── covid_vaccine_statewise.csv
│   ├── StatewiseTestingDetails.csv
│   └── shapefiles/
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
│   └── other visualizations
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
│   ├── dashboard/
│   │   └── dash_app.py
│   └── model notebooks
│
└── README.md
Results

The hybrid framework successfully:

Captured temporal epidemic trends
Modeled spatial spillover between neighboring states
Generated uncertainty-aware forecasts
Produced state-level risk probabilities
Identified potential future outbreak regions

Key findings:

Neighboring states significantly affect epidemic propagation.
Higher vaccination coverage reduces future risk.
Combining deep learning with probabilistic modeling improves interpretability and forecasting reliability.

Technologies Used
Python
TensorFlow / Keras
Pandas
NumPy
Scikit-Learn
GeoPandas
Matplotlib
FastAPI
React
Bayesian Inference
Markov Models
