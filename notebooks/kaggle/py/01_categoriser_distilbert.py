# =============================================================================
# Finance Copilot — Merchant-String Categoriser (DistilBERT → int8 ONNX)
# Paste this ENTIRE file into one Kaggle notebook code cell → Save & Run All.
#
# Required input dataset: fincopilot-synthetic-transactions (transactions.parquet)
# GPU: optional — enable T4 under Settings → Accelerator for ~10x speedup.
# Output: /kaggle/working/categoriser/ (model.onnx int8, tokenizer, meta.json,
#         metrics.json) → download into repo models/categoriser/
# =============================================================================

import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'transformers>=4.43', 'datasets>=2.20', 'onnx>=1.16', 'onnxruntime>=1.18',
                'optimum[onnxruntime]>=1.21', 'evaluate', 'pyarrow', 'scikit-learn'], check=True)

import os, json, math, re, unicodedata
from pathlib import Path
import numpy as np, pandas as pd

# ---------- locate input ----------
CANDS = sorted(Path('/kaggle/input').rglob('*.parquet'))
if not CANDS:
    present = sorted(str(p) for p in Path('/kaggle/input').rglob('*'))
    raise FileNotFoundError(
        'No .parquet file found under /kaggle/input.\n'
        'Found: ' + '\n  '.join(present[:50]) + '\n'
        'Fix: 1) Kaggle → Datasets → New Dataset → upload transactions.parquet\n'
        '     2) In THIS notebook: right panel → Input → + Add Data → pick the dataset\n'
        '     3) Re-run this cell.'
    )
DATA = CANDS[0]
# ignore pandas' FutureWarning noise from sklearn internals below
import warnings; warnings.filterwarnings('ignore', category=FutureWarning)
OUT = Path('/kaggle/working/categoriser'); OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(DATA)
print(DATA, len(df), 'rows,', df['persona'].nunique(), 'personas')
df.head(3)

# ---------- normalisation + features (MUST match repo: fincopilot/ingestion/normalise.py + ml/categoriser/features.py) ----------
_NOISE = re.compile(r'\b(POS|PURCHASE|PAYMENT|CARD|DEBIT|CREDIT|TXN|TRANSACTION|REF|AUTH|VISA|MASTERCARD|MC|ONLINE|ECOM|E-COM|INTL|TERMINAL|TRX|NEFT|IMPS|UPI|ACH|SEPA|BKASH|NAGAD)\b', re.I)
_LONG  = re.compile(r'\d{5,}'); _SHORT = re.compile(r'\b\d{1,4}\b')
_MASK  = re.compile(r'[xX*#]{3,}\d*'); _DATE = re.compile(r'\b\d{1,2}[/.-]\d{1,2}([/.-]\d{2,4})?\b')
_NON   = re.compile(r'[^a-z0-9<> ]+'); _SP = re.compile(r'\s+')
WEEKDAYS = ('mon','tue','wed','thu','fri','sat','sun')

def normalise(raw):
    s = unicodedata.normalize('NFKC', str(raw))
    s = _DATE.sub(' ', s); s = _MASK.sub(' <masked> ', s); s = _LONG.sub(' <num> ', s)
    s = _NOISE.sub(' ', s); s = _SHORT.sub(' ', s)
    s = _NON.sub(' ', s.lower())
    return _SP.sub(' ', s).strip()

def amount_bucket(a):
    return 'amt_0' if a <= 0 else f"amt_{min(int(math.log10(a)*2), 11)}"

def build_text(desc, amt, wd):
    return f"{normalise(desc)} [SEP] {amount_bucket(amt)} {WEEKDAYS[wd%7]}"

df['date'] = pd.to_datetime(df['date'])
df['text'] = [build_text(d, a, w) for d, a, w in zip(df.raw_descriptor, df.amount, df.date.dt.weekday)]
print(df[['raw_descriptor','text','subcategory']].sample(5).to_string())

# ---------- split by account (unseen users = honest generalisation) ----------
from sklearn.model_selection import train_test_split
accounts = np.array(sorted(df['account_id'].unique()))
tr_a, te_a = train_test_split(accounts, test_size=0.25, random_state=42)
train, test = df[df.account_id.isin(tr_a)].copy(), df[df.account_id.isin(te_a)].copy()
labels = sorted(df['subcategory'].unique())
lab2id = {l: i for i, l in enumerate(labels)}; id2lab = {i: l for l, i in lab2id.items()}
train['label'] = train.subcategory.map(lab2id); test['label'] = test.subcategory.map(lab2id)
print(f'train {len(train):,} | test {len(test):,} | {len(labels)} classes')

# ---------- baseline: TF-IDF char n-grams + Logistic Regression (the bar to beat) ----------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score
import time
base = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), min_df=2, sublinear_tf=True)),
    ('clf', LogisticRegression(max_iter=1000, C=4.0, class_weight='balanced', n_jobs=-1)),
])
t0=time.perf_counter(); base.fit(train.text, train.label); fit_s=time.perf_counter()-t0
t0=time.perf_counter(); probs = base.predict_proba(test.text); ms=(time.perf_counter()-t0)*1000/len(test)
full = np.zeros((len(test), len(labels))); full[:, base.classes_] = probs
bp = base.classes_[probs.argmax(1)]
baseline = {'macro_f1': float(f1_score(test.label, bp, average='macro')),
            'accuracy': float(accuracy_score(test.label, bp)),
            'top3': float(top_k_accuracy_score(test.label, full, k=3)),
            'ms_per_row': ms, 'fit_s': fit_s}
print('BASELINE', json.dumps(baseline, indent=2))

# ---------- DistilBERT fine-tune ----------
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from datasets import Dataset

MODEL = 'distilbert-base-uncased'
tok = AutoTokenizer.from_pretrained(MODEL)
MAX_LEN = 48
def tok_fn(b): return tok(b['text'], truncation=True, max_length=MAX_LEN)
ds_tr = Dataset.from_pandas(train[['text','label']], preserve_index=False).map(tok_fn, batched=True)
ds_te = Dataset.from_pandas(test[['text','label']], preserve_index=False).map(tok_fn, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(labels), id2label=id2lab, label2id=lab2id)

args = TrainingArguments(
    output_dir='/kaggle/working/ckpt', num_train_epochs=3, seed=42,
    per_device_train_batch_size=64, per_device_eval_batch_size=128,
    learning_rate=3e-5, weight_decay=0.01, warmup_ratio=0.06,
    eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
    metric_for_best_model='f1', report_to=[], fp16=torch.cuda.is_available(),
    logging_steps=50,
)
def compute(p):
    pred = p.predictions.argmax(1)
    return {'f1': f1_score(p.label_ids, pred, average='macro'),
            'acc': accuracy_score(p.label_ids, pred)}
trainer = Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_te,
                  processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                  compute_metrics=compute)
trainer.train()

# ---------- eval + calibration ----------
from sklearn.metrics import classification_report
out = trainer.predict(ds_te)
logits = out.predictions
e = np.exp(logits - logits.max(1, keepdims=True)); probs = e/e.sum(1, keepdims=True)
pred = probs.argmax(1); conf = probs.max(1)
ours = {'macro_f1': float(f1_score(test.label, pred, average='macro')),
        'accuracy': float(accuracy_score(test.label, pred)),
        'top3': float(top_k_accuracy_score(test.label, probs, k=3)),
        'ece': float(np.mean(np.abs(conf - (pred == test.label))))}
print('DISTILBERT', json.dumps(ours, indent=2))
print(classification_report(test.label, pred, target_names=[id2lab[i] for i in range(len(labels))], zero_division=0))

# ---------- export to int8 ONNX for the app ----------
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
hf_dir = Path('/kaggle/working/hf_export')
trainer.save_model(hf_dir); tok.save_pretrained(hf_dir)
ort = ORTModelForSequenceClassification.from_pretrained(hf_dir, export=True)
ort_dir = Path('/kaggle/working/ort_fp32')
ort.save_pretrained(ort_dir); tok.save_pretrained(ort_dir)
q = ORTQuantizer.from_pretrained(ort_dir)
qconf = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
q.quantize(save_dir=OUT, quantization_config=qconf)
tok.save_pretrained(OUT)
print('ONNX size MB:', round((OUT/'model.onnx').stat().st_size/1e6, 1))

# ---------- ONNX smoke test (must match repo infer.py behaviour) ----------
import onnxruntime as ort_rt
sess = ort_rt.InferenceSession(str(OUT/'model.onnx'), providers=['CPUExecutionProvider'])
names = {i.name for i in sess.get_inputs()}
sample = test.sample(256, random_state=0)
enc = tok(list(sample.text), padding=True, truncation=True, max_length=MAX_LEN, return_tensors='np')
feeds = {k: v.astype(np.int64) for k, v in enc.items() if k in names}
onnx_logits = sess.run(None, feeds)[0]
oe = np.exp(onnx_logits - onnx_logits.max(1, keepdims=True)); op = oe/oe.sum(1, keepdims=True)
onnx_f1 = f1_score(sample.label, op.argmax(1), average='macro')
print('onnx macro-F1 (256-sample):', round(float(onnx_f1), 4))

# ---------- meta + metrics ----------
version = 'distilbert-v1.0'
(OUT/'meta.json').write_text(json.dumps({
    'version': version, 'base_model': MODEL, 'max_length': MAX_LEN,
    'n_classes': len(labels), 'label2id': lab2id,
}, indent=2))
summary = {'baseline_tfidf_lr': baseline, 'distilbert_int8': ours, 'version': version}
(OUT/'metrics.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print('\nDone. Download /kaggle/working/categoriser/ → copy into repo models/categoriser/')
