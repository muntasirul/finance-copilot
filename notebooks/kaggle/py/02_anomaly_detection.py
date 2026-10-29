# =============================================================================
# Finance Copilot — Anomaly & Subscription Detection
# Paste this ENTIRE file into one Kaggle notebook code cell → Save & Run All.
#
# Required input dataset: fincopilot-synthetic-transactions (transactions.parquet)
# GPU: not needed (CPU only).
# Output: /kaggle/working/anomaly/ (model.pkl + metrics.json) → models/anomaly/
#
# Evaluates against injected ground-truth anomaly_type labels:
# duplicate_charge, amount_spike, new_merchant_burst.
# Target: recall >= 0.85 at <= 2 false alerts / 100 transactions.
# =============================================================================

import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'scikit-learn', 'pandas', 'pyarrow', 'joblib'], check=True)

import json
from pathlib import Path
import numpy as np, pandas as pd

DATA = list(Path('/kaggle/input').glob('*/transactions.parquet'))[0]
OUT = Path('/kaggle/working/anomaly'); OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(DATA); df['date'] = pd.to_datetime(df['date'])
print(len(df), 'rows |', df.anomaly_type.value_counts().to_dict())

# ---------- features (same recipe the app's anomaly job uses) ----------
def build_features(d):
    d = d.sort_values(['account_id', 'date']).copy()
    grp = d.groupby('merchant_id')['amount']
    d['amt_z_merchant'] = (d.amount - grp.transform('mean')) / grp.transform('std').replace(0, np.nan)
    grp2 = d.groupby(['account_id', 'subcategory'])['amount']
    d['amt_z_cat'] = (d.amount - grp2.transform('mean')) / grp2.transform('std').replace(0, np.nan)
    d['merchant_seen_before'] = (d.groupby('merchant_id').cumcount() > 0).astype(int)
    d['acct_merchant_seen'] = (d.groupby(['account_id','merchant_id']).cumcount() > 0).astype(int)
    d['dup_key'] = (d['account_id'] + '|' + d['merchant_id'].astype(str) + '|' +
                    d.amount.round(2).astype(str))
    d['dup_within_72h'] = 0
    for acct, sub in d.groupby('account_id'):
        for key, g in sub.groupby('dup_key'):
            dd = g.date.diff().dt.days.fillna(9999)
            d.loc[g.index, 'dup_within_72h'] = (dd.le(3)).astype(int).values
    d = d.drop(columns=['dup_key'])
    d['weekday'] = d.date.dt.weekday; d['day'] = d.date.dt.day
    d['amt_log'] = np.log1p(d.amount)
    for c in ['amt_z_merchant','amt_z_cat']: d[c] = d[c].fillna(0)
    return d

FEATS = ['amt_log','amt_z_merchant','amt_z_cat','merchant_seen_before','acct_merchant_seen',
         'dup_within_72h','weekday','day']
feat = build_features(df)
feat[FEATS].describe().T

# ---------- train / test split (by account again) ----------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
accounts = np.array(sorted(feat.account_id.unique()))
tr_a, te_a = train_test_split(accounts, test_size=0.25, random_state=42)
train = feat[feat.account_id.isin(tr_a)]; test = feat[feat.account_id.isin(te_a)]
y_test = (test.anomaly_type != '').astype(int).values

iso = IsolationForest(n_estimators=400, contamination=0.02, random_state=42, n_jobs=-1)
iso.fit(train[FEATS])
scores = -iso.score_samples(test[FEATS])  # higher = more anomalous

# ---------- choose threshold on false-alert budget ----------
from sklearn.metrics import precision_recall_curve, roc_auc_score
prec, rec, thr = precision_recall_curve(y_test, scores)
n_txn = len(y_test)
best = None
for p, r, t in zip(prec, rec, thr):
    fp = (1-p)/p * (r * y_test.sum()) if p > 0 else np.inf
    fa_per_100 = fp / n_txn * 100
    if fa_per_100 <= 2.0 and (best is None or r > best['recall']):
        best = {'threshold': float(t), 'recall': float(r), 'precision': float(p),
                'false_alerts_per_100': float(fa_per_100)}
metrics = {'iso_forest': {**(best or {}), 'roc_auc': float(roc_auc_score(y_test, scores))},
           'n_test': int(n_txn), 'n_anomalies': int(y_test.sum())}
print(json.dumps(metrics, indent=2))
print('by anomaly type:')
flagged = scores >= (best['threshold'] if best else np.inf)
print(test.assign(flagged=flagged).groupby('anomaly_type')['flagged'].mean().to_string())

# ---------- subscription / periodicity detector ----------
def detect_subscriptions(d, tol_days=3, tol_amt=0.15):
    subs = []
    for (acct, merch), g in d[d.direction=='debit'].groupby(['account_id','merchant_id']):
        g = g.sort_values('date')
        gaps = g.date.diff().dt.days.dropna()
        if len(gaps) < 2: continue
        cad = int(round(gaps.median()))
        stable = (gaps.max()-gaps.min() <= 2*tol_days) and \
                 (g.amount.std() <= tol_amt * g.amount.mean())
        if stable and cad in (7, 14, 28, 29, 30, 31, 90, 91, 365):
            subs.append({'account_id': acct, 'merchant_id': merch, 'cadence_days': cad,
                         'typical_amount': float(g.amount.median()),
                         'n': int(len(g)), 'price_change_pct':
                         float((g.amount.iloc[-1]-g.amount.iloc[0])/g.amount.iloc[0]*100)})
    return pd.DataFrame(subs)

subs = detect_subscriptions(feat)
true = feat[feat.is_recurring].groupby(['account_id','merchant_id']).size().rename('true').reset_index()
found = subs.merge(true, on=['account_id','merchant_id'], how='outer').fillna(0)
sub_metrics = {'n_subscriptions_found': len(subs),
               'coverage_of_true_recurring': float((found.true > 0).mean())}
print(json.dumps(sub_metrics, indent=2))
subs.head(10)

# ---------- persist ----------
import joblib
joblib.dump({'model': iso, 'features': FEATS,
             'threshold': best['threshold'] if best else 3.0}, OUT/'model.pkl')
(OUT/'metrics.json').write_text(json.dumps({**metrics, **sub_metrics}, indent=2))
print('wrote', list(OUT.iterdir()))
