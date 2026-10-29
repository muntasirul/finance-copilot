# =============================================================================
# Finance Copilot — 30-Day Cash-Flow Forecasting (LightGBM quantile)
# Paste this ENTIRE file into one Kaggle notebook code cell → Save & Run All.
#
# Required input dataset: fincopilot-synthetic-transactions (transactions.parquet)
# GPU: not needed.
# Output: /kaggle/working/forecast/ (lgbm_q10/50/90.txt, metrics.json) → models/forecast/
#
# Daily net cash-flow per account → cumulative balance path with P10/P50/P90 bands.
# Baselines: seasonal-naive-ish + ETS. Rolling-origin backtest. Target sMAPE <= 12%.
# =============================================================================

import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'lightgbm', 'statsmodels', 'pandas', 'pyarrow', 'scikit-learn'], check=True)

import json
from pathlib import Path
import numpy as np, pandas as pd

CANDS = sorted(Path('/kaggle/input').rglob('*.parquet'))
if not CANDS:
    present = sorted(str(p) for p in Path('/kaggle/input').rglob('*'))
    raise FileNotFoundError(
        'No .parquet found under /kaggle/input. Found:\n  ' + '\n  '.join(present[:50]) +
        '\nAttach your dataset: right panel → Input → + Add Data → fincopilot-synthetic-transactions'
    )
DATA = CANDS[0]
OUT = Path('/kaggle/working/forecast'); OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(DATA); df['date'] = pd.to_datetime(df['date'])
df['signed'] = np.where(df.direction == 'debit', -df.amount, df.amount)
daily = df.groupby(['account_id','date'])['signed'].sum().rename('net').reset_index()
print(len(daily), 'account-days')

# ---------- build a regular daily series per account ----------
def regularise(d):
    idx = pd.date_range(d.date.min(), d.date.max(), freq='D')
    return d.set_index('date').reindex(idx).rename_axis('date').reset_index().fillna({'net': 0})

acct_frames = {a: regularise(g) for a, g in daily.groupby('account_id')}
ACCT = sorted(acct_frames)[0]  # demo on one account; loop all for full eval
s = acct_frames[ACCT]
s.head()

# ---------- features ----------
def make_features(s):
    d = s.copy()
    d['net_prev'] = d.net.shift(1)
    for lag in (1, 7, 14, 28):
        d[f'lag_{lag}'] = d.net_prev.shift(lag-1)
    for w in (7, 14, 28):
        d[f'roll{w}_mean'] = d.net_prev.rolling(w).mean()
        d[f'roll{w}_std'] = d.net_prev.rolling(w).std()
    d['dow'] = d.date.dt.weekday; d['dom'] = d.date.dt.day
    d['is_weekend'] = (d.dow >= 5).astype(int)
    d['is_month_end'] = d.date.dt.is_month_end.astype(int)
    d['is_month_start'] = d.date.dt.is_month_start.astype(int)
    return d

FEATS = ['lag_1','lag_7','lag_14','lag_28','roll7_mean','roll7_std','roll14_mean','roll14_std',
         'roll28_mean','roll28_std','dow','dom','is_weekend','is_month_end','is_month_start']
f = make_features(s).dropna()
f[FEATS + ['net']].tail()

# ---------- rolling-origin backtest: last 90 days in 3 folds of 30 ----------
import lightgbm as lgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
def smape(a, p):
    a, p = np.asarray(a), np.asarray(p)
    return float(np.mean(2*np.abs(p-a)/(np.abs(a)+np.abs(p)+1e-9))*100)

H, results = 30, []
fold_starts = [len(f) - H*3, len(f) - H*2, len(f) - H]
models = {}
for fs in fold_starts:
    tr, te = f.iloc[:fs], f.iloc[fs:fs+H]
    for q, name in ((0.1,'q10'), (0.5,'q50'), (0.9,'q90')):
        m = lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=400,
                              learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
        m.fit(tr[FEATS], tr.net)
        models.setdefault(name, m)
    p50 = models['q50'].predict(te[FEATS])
    naive = np.repeat(tr.net.rolling(7).mean().iloc[-1], H)  # seasonal-naive-ish
    try:
        ets = ExponentialSmoothing(tr.net, seasonal='add', seasonal_periods=7,
                                   trend='add', damped_trend=True).fit()
        p_ets = ets.forecast(H).values
    except Exception:
        p_ets = naive
    results.append({'fold_start': fs,
                    'lgbm_smape': smape(te.net, p50),
                    'naive_smape': smape(te.net, naive),
                    'ets_smape': smape(te.net, p_ets)})
    print(results[-1])

res = pd.DataFrame(results)
print(res.mean(numeric_only=True).round(2))

# ---------- 30-day forward forecast + intervals on the last fold ----------
fs = fold_starts[-1]; te = f.iloc[fs:fs+H]
p10 = models['q10'].predict(te[FEATS]); p50 = models['q50'].predict(te[FEATS])
p90 = models['q90'].predict(te[FEATS])
start_bal = float(s.loc[s.date < te.date.iloc[0], 'net'].sum()) + 1000
bal_true = start_bal + te.net.cumsum().values
bal_p10 = start_bal + np.cumsum(p10); bal_p50 = start_bal + np.cumsum(p50); bal_p90 = start_bal + np.cumsum(p90)
cov = float(np.mean((bal_true >= bal_p10) & (bal_true <= bal_p90)))
forecast_metrics = {
    'lgbm_net_smape': res.lgbm_smape.mean(), 'naive_net_smape': res.naive_smape.mean(),
    'ets_net_smape': res.ets_smape.mean(), 'balance_smape_p50': smape(bal_true, bal_p50),
    'interval_coverage_80': cov, 'horizon_days': H}
print(json.dumps(forecast_metrics, indent=2))

# ---------- refit on all data + save ----------
for q, name in ((0.1,'q10'), (0.5,'q50'), (0.9,'q90')):
    m = lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=400,
                          learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    m.fit(f[FEATS], f.net)
    m.booster_.save_model(str(OUT/f'lgbm_{name}.txt'))
(OUT/'metrics.json').write_text(json.dumps({'features': FEATS, **forecast_metrics}, indent=2))
print('saved to', OUT)
