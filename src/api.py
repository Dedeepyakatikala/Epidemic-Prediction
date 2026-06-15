# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
from src.predictors import load_panel, load_adj, load_models, lstm_multi_step_forecast, gibbs_multi_day_rollout, aggregate_trajectories_to_forecasts
import numpy as np

app = FastAPI(title="Epidemic Forecast API")

# load resources once
panel = load_panel()
adj = load_adj()
lstm, clf, scaler = load_models()

class ForecastRequest(BaseModel):
    state: str
    horizon: int = 7
    ensemble: int = 200

@app.get("/")
def root():
    return {"message":"Epidemic Forecast API"}

@app.post("/predict")
def predict(req: ForecastRequest):
    state = req.state
    H = req.horizon
    M = req.ensemble

    # prepare Z_prev (last observed day)
    last_day = panel['Date'].max()
    last_rows = panel[panel['Date']==last_day]
    Z_prev = dict(zip(last_rows['state'], (last_rows['active']>0).astype(int)))

    # prepare feature maps and lstm inputs
    # build X_seq for each state from panel: latest seq_len rows (must match seq_len used in LSTM)
    seq_len = 14
    features = ['active','new_confirmed','daily_tests','daily_vax','vax_rate']
    lstm_preds_by_state = {}
    lstm_means_by_state = {}
    features_map = {}
    states = last_rows['state'].tolist()
    for st in states:
        sub = panel[panel['state']==st].sort_values('Date').reset_index(drop=True)
        recent = sub.tail(seq_len)
        if len(recent) < seq_len:
            # pad by repeating last
            pad = np.vstack([recent[features].values] * (seq_len - len(recent)))
            X_seq = np.vstack([pad, recent[features].values])
        else:
            X_seq = recent[features].values
        region_id = int(recent['state_id'].iloc[-1])
        preds = lstm_multi_step_forecast(lstm, X_seq, region_id, steps=H)
        lstm_preds_by_state[st] = preds
        lstm_means_by_state[st] = preds  # here treat model output as mean; if scaled, convert appropriately
        features_map[st] = {'daily_tests': int(recent['daily_tests'].iloc[-1]) if 'daily_tests' in recent.columns else 0,
                            'daily_vax': int(recent['daily_vax'].iloc[-1]) if 'daily_vax' in recent.columns else 0}

    # run Gibbs to create trajectories
    trajectories = gibbs_multi_day_rollout(adj, clf, scaler, Z_prev, lstm_preds_by_state, features_map, H=H, M=M)
    forecasts = aggregate_trajectories_to_forecasts(trajectories, lstm_means_by_state, alpha=0.3, phi=20, cfr=0.01)

    # prepare response for requested state
    res = {'state': state, 'horizon': H, 'forecast': []}
    for day in range(1, H+1):
        if state in forecasts[day]:
            res['forecast'].append({'day': day, **forecasts[day][state]})
        else:
            res['forecast'].append({'day': day, 'error': 'state not found'})
    return res

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
