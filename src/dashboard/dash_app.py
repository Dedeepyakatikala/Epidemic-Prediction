# src/dashboard/dash_app.py
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Import predictor functions
from src.predictors import load_panel, load_adj, load_models, lstm_multi_step_forecast, gibbs_multi_day_rollout, aggregate_trajectories_to_forecasts

# Load resources once
panel = load_panel()
adj = load_adj()
lstm, clf, scaler = load_models()

# Dash app with bootstrap theme
app = dash.Dash(__name__, title="Epidemic Forecast Dashboard", external_stylesheets=[dbc.themes.FLATLY])
server = app.server  # for deployment

states = sorted(panel['state'].unique())

# Sidebar for controls
sidebar = dbc.Card(
    [
        html.H4("Controls", className="card-title text-center"),
        html.Hr(),
        dbc.Label("Select State"),
        dcc.Dropdown(
            id="state-dropdown",
            options=[{"label": s, "value": s} for s in states],
            value="India",
            clearable=False,
        ),
        html.Br(),
        dbc.Label("Forecast Horizon (days)"),
        dcc.Slider(
            id="horizon-slider",
            min=7, max=14, step=1, value=7,
            marks={7: "7", 14: "14"},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Br(),
        dbc.Label("Ensemble Size"),
        dcc.Slider(
            id="ensemble-slider",
            min=50, max=1000, step=50, value=300,
            marks={50: "50", 500: "500", 1000: "1000"},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
    ],
    body=True,
    style={"height": "100%"}
)

# Tabs for graphs
tabs = dbc.Tabs(
    [
        dbc.Tab(dcc.Graph(id="history-graph"), label="Historical Data"),
        dbc.Tab(dcc.Graph(id="forecast-cases-graph"), label="Forecasted Cases"),
        dbc.Tab(dcc.Graph(id="forecast-risk-graph"), label="Risk Probability"),
    ]
)

# Layout
app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row(
            dbc.Col(html.H1("📊 Epidemic Risk Dashboard", className="text-center text-primary mb-4"), width=12)
        ),
        dbc.Row(
            [
                dbc.Col(sidebar, md=3),
                dbc.Col(
                    [
                        tabs,
                        html.Br(),
                        dbc.Card(
                            dbc.CardBody(
                                [html.H4("Neighbor Influence", className="card-title"), html.Div(id="neighbors-table")]
                            )
                        ),
                    ],
                    md=9,
                ),
            ]
        ),
    ],
)

@app.callback(
    Output("history-graph", "figure"),
    Output("forecast-cases-graph", "figure"),
    Output("forecast-risk-graph", "figure"),
    Output("neighbors-table", "children"),
    Input("state-dropdown", "value"),
    Input("horizon-slider", "value"),
    Input("ensemble-slider", "value"),
)
def update_dashboard(selected_state, horizon, ensemble):
    # --- Historical data ---
    last_state_rows = panel[panel["state"] == selected_state].sort_values("Date")
    history_plot = last_state_rows.tail(60).copy()

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=history_plot["Date"], y=history_plot["active"],
                                  mode="lines+markers", name="Active cases"))
    if "Z" in history_plot.columns:
        fig_hist.add_trace(go.Bar(x=history_plot["Date"], y=history_plot["Z"] * history_plot["active"].max(),
                                  name="Risk indicator Z", opacity=0.4, marker_color="red"))
    fig_hist.update_layout(title="Historical Active Cases & Risk", yaxis_title="Active cases")

    # --- Forecast placeholders ---
    days = list(range(1, horizon + 1))
    cases_m = [int(np.random.randint(0, 100)) for _ in days]
    cases_hi = [c + 20 for c in cases_m]
    cases_lo = [max(0, c - 20) for c in cases_m]
    risk_p = [np.random.rand() for _ in days]

    fig_cases = go.Figure()
    fig_cases.add_trace(go.Scatter(x=days, y=cases_m, mode="lines+markers", name="Predicted median new cases"))
    fig_cases.add_trace(go.Scatter(x=days, y=cases_hi, mode="lines", line=dict(width=0), showlegend=False))
    fig_cases.add_trace(go.Scatter(x=days, y=cases_lo, mode="lines", fill="tonexty", line=dict(width=0), showlegend=False))
    fig_cases.update_layout(title="Forecasted Cases", xaxis_title="Days ahead", yaxis_title="Cases")

    fig_risk = go.Figure()
    fig_risk.add_trace(go.Scatter(x=days, y=risk_p, mode="lines+markers", name="P(risk=1)"))
    fig_risk.update_layout(title="Forecasted Risk Probability", xaxis_title="Days ahead",
                           yaxis_title="Probability", yaxis_range=[0, 1])

    # --- Neighbor info ---
    nbs = adj.loc[selected_state][adj.loc[selected_state] == 1].index.tolist() if selected_state in adj.index else []
    if not nbs:
        nb_table = html.Div("No neighbors in adjacency matrix.")
    else:
        last_day = panel["Date"].max()
        nb_rows = []
        for nb in nbs:
            last_nb = panel[(panel["state"] == nb) & (panel["Date"] == last_day)]
            last_active = int(last_nb["active"].values[0]) if not last_nb.empty else 0
            nb_rows.append({"Neighbor": nb, "Last Active": last_active})
        df_nb = pd.DataFrame(nb_rows)
        nb_table = dbc.Table.from_dataframe(df_nb, striped=True, bordered=True, hover=True)

    return fig_hist, fig_cases, fig_risk, nb_table

if __name__ == "__main__":
    app.run(debug=True, port=8050)
