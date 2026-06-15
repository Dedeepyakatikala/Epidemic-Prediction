# src/predictors.py
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import nbinom, poisson

# -----------------------
# CONFIG / PATHS
import os
import tensorflow as tf
import joblib

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
MODEL_LSTM_PATH = os.path.join(BASE_DIR, "models", "global_lstm.keras")
MARKOV_PKL = os.path.join(BASE_DIR, "models", "markov_logistic.pkl")
MARKOV_SCALER = os.path.join(BASE_DIR, "models", "markov_scaler.pkl")

PANEL_PATH = os.path.join(BASE_DIR, "data_curated", "panel_state_daily.csv")
ADJ_PATH = os.path.join(BASE_DIR, "data_curated", "state_adjacency.csv")

# -----------------------
# Utilities: load data & models
# -----------------------
def load_panel():
    panel = pd.read_csv(PANEL_PATH, parse_dates=['Date'])
    panel = panel.sort_values(['state','Date']).reset_index(drop=True)
    return panel

def load_adj():
    adj = pd.read_csv(ADJ_PATH, index_col=0)
    return adj

def load_models():
    """
    Load all models and scalers with absolute paths.
    Returns:
        lstm (tf.keras.Model): LSTM model
        clf (sklearn estimator): Logistic Regression classifier
        scaler (sklearn scaler): Preprocessing scaler
    """
    # Load LSTM model
    if not os.path.exists(MODEL_LSTM_PATH):
        raise FileNotFoundError(f"LSTM model not found at {MODEL_LSTM_PATH}")
    lstm = tf.keras.models.load_model(MODEL_LSTM_PATH, compile=False)

    # Load logistic regression classifier
    if not os.path.exists(MARKOV_PKL):
        raise FileNotFoundError(f"Markov classifier not found at {MARKOV_PKL}")
    clf = joblib.load(MARKOV_PKL)

    # Load scaler
    if not os.path.exists(MARKOV_SCALER):
        raise FileNotFoundError(f"Scaler not found at {MARKOV_SCALER}")
    scaler = joblib.load(MARKOV_SCALER)

    return lstm, clf, scaler

# -----------------------
# LSTM multi-step predictor (iterative)
# -----------------------
def lstm_multi_step_forecast(lstm_model, X_seq, region_id, steps=7, seq_len=None):
    """
    Iteratively forecast steps ahead by feeding predictions back.
    - X_seq: np.array shape (seq_len, n_features)
    """
    if seq_len is None:
        seq_len = X_seq.shape[0]

    preds = []
    seq = X_seq.copy().astype(float)

    for h in range(steps):
        seq_in = seq.reshape((1, seq.shape[0], seq.shape[1]))
        reg_in = np.array([region_id])
        p = lstm_model([seq_in, reg_in], training=False).numpy()[0,0]
        preds.append(p)

        # Append predicted value and drop oldest row
        new_row = seq[-1].copy()
        new_row[0] = p
        seq = np.vstack([seq[1:], new_row])

    return np.array(preds)

# -----------------------
# Compute mean neighbor Z
# -----------------------
def compute_mean_neighbor_from_dict(state, Z_dict, adj):
    if state not in adj.index:
        return 0.0
    nbs = adj.loc[state][adj.loc[state]==1].index.tolist()
    if not nbs:
        return 0.0
    vals = [Z_dict.get(nb, 0) for nb in nbs]
    return float(np.mean(vals))

# -----------------------
# Conditional probability computation (logistic)
# -----------------------
def compute_conditional_prob(clf, scaler, Z_prev_self, mean_nb, trend_pred, daily_tests, daily_vax):
    x = np.array([Z_prev_self, mean_nb, trend_pred, daily_tests, daily_vax]).reshape(1,-1)
    x_s = scaler.transform(x)
    prob = clf.predict_proba(x_s)[0,1]
    return float(prob)

# -----------------------
# Gibbs sampler for multi-day rollout
# -----------------------
def gibbs_multi_day_rollout(adj, clf, scaler, Z_prev, lstm_preds_by_state, features_map, H=7, M=200, n_sweeps=1, random_seed=42):
    np.random.seed(random_seed)
    states = list(Z_prev.keys())
    trajectories = []

    for m in range(M):
        Z_t_minus_1 = Z_prev.copy()
        traj = {}
        for day in range(H):
            Z_current = Z_t_minus_1.copy()
            order = states.copy()
            np.random.shuffle(order)
            for s in order:
                nbs = adj.loc[s][adj.loc[s]==1].index.tolist() if s in adj.index else []
                mean_nb = np.mean([Z_current[nb] for nb in nbs]) if nbs else 0.0
                trend_pred = float(lstm_preds_by_state.get(s, np.zeros(H))[day])
                feats = features_map.get(s, {})
                daily_tests = float(feats.get('daily_tests', 0))
                daily_vax = float(feats.get('daily_vax', 0))
                prob = compute_conditional_prob(clf, scaler, Z_t_minus_1.get(s,0), mean_nb, trend_pred, daily_tests, daily_vax)
                Z_current[s] = 1 if np.random.rand() < prob else 0
            traj[day+1] = Z_current.copy()
            Z_t_minus_1 = Z_current.copy()
        trajectories.append(traj)

    return trajectories

# -----------------------
# Aggregate trajectories to forecasts
# -----------------------
def aggregate_trajectories_to_forecasts(trajectories, lstm_means_by_state, alpha=0.3, phi=20, cfr=0.01, death_lag=7):
    M = len(trajectories)
    H = len(trajectories[0].keys())
    states = list(lstm_means_by_state.keys())

    probs = {day: {s:0.0 for s in states} for day in range(1, H+1)}
    cases_samples = {day: {s: [] for s in states} for day in range(1, H+1)}
    deaths_samples = {day: {s: [] for s in states} for day in range(1, H+1)}

    for traj in trajectories:
        for day in range(1, H+1):
            Zs = traj[day]
            for s in states:
                z = Zs.get(s,0)
                probs[day][s] += z
                m = float(lstm_means_by_state.get(s, np.zeros(H))[day-1])
                lam = max(m * (1 + alpha * z), 0)
                if phi <= 0:
                    y = poisson.rvs(mu=lam)
                else:
                    r = phi
                    p = r / (r + lam) if (r + lam) > 0 else 1.0
                    y = nbinom.rvs(r, p)
                cases_samples[day][s].append(int(y))
                deaths_samples[day][s].append(int(round(y * cfr)))

    forecasts = {}
    for day in range(1, H+1):
        forecasts[day] = {}
        for s in states:
            p = probs[day][s] / float(M)
            cases_arr = np.array(cases_samples[day][s]) if cases_samples[day][s] else np.array([0])
            deaths_arr = np.array(deaths_samples[day][s]) if deaths_samples[day][s] else np.array([0])
            forecasts[day][s] = {
                'P_Z1': float(p),
                'cases_median': int(np.median(cases_arr)),
                'cases_q025': int(np.percentile(cases_arr, 2.5)),
                'cases_q975': int(np.percentile(cases_arr, 97.5)),
                'deaths_median': int(np.median(deaths_arr)),
                'deaths_q025': int(np.percentile(deaths_arr, 2.5)),
                'deaths_q975': int(np.percentile(deaths_arr, 97.5)),
            }
    return forecasts
