# scripts/01_infer_headlines.py
# - Đọc data/news/headlines.csv (ticker,date,time,headline)
# - Load model/phobert_large_v2 để phân loại pos/neg
# - Tạo effective_date: tin sau 15:00 -> hiệu lực từ ngày làm việc kế tiếp
# - Lưu data/processed/headline_predictions.parquet

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, time, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from underthesea import word_tokenize

# -------- Paths --------
ROOT = Path(".")
MODEL_DIR = ROOT / "model" / "phobert_large_v2"
IN_CSV    = ROOT / "data" / "news" / "headlines.csv"
OUT_DIR   = ROOT / "data" / "processed"
OUT_FILE  = OUT_DIR / "headline_predictions.parquet"

MAX_LEN = 128
BATCH   = 64
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

def parse_date(s):
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    return pd.NaT

def parse_time(s):
    s = str(s).strip()
    # cho phép "HH:MM" hoặc "HH:MM:SS"
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).time()
        except Exception:
            pass
    return None

def next_business_day(d: datetime.date) -> datetime.date:
    # Mon-Fri; (sau sẽ lọc lại theo lịch giao dịch thực tế)
    cur = d + timedelta(days=1)
    while cur.weekday() >= 5:  # 5,6 = Sat,Sun
        cur += timedelta(days=1)
    return cur

# -------- Load data --------
df = pd.read_csv(IN_CSV)
df.columns = [c.lower() for c in df.columns]
assert set(["ticker", "date", "time", "headline"]).issubset(df.columns), \
    "headlines.csv phải có 4 cột: ticker,date,time,headline"

df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
df["date"]   = df["date"].apply(parse_date)
df["time"]   = df["time"].apply(parse_time)
df["headline"] = df["headline"].astype(str).str.strip()
df = df.dropna(subset=["ticker","date","headline"]).copy()
df = df[df["headline"] != ""].reset_index(drop=True)

# -------- Effective date với quy tắc 15:00 --------
cutoff = time(15, 0, 0)
eff_dates = []
for d, t in zip(df["date"], df["time"]):
    if t is not None and t >= cutoff:
        eff_dates.append(next_business_day(d))
    else:
        eff_dates.append(d)
df["effective_date"] = eff_dates

# -------- PhoBERT inference (dùng text đã tách từ) --------
tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

def infer_batch(texts):
    # PhoBERT thích văn bản đã word-seg
    seg = [word_tokenize(t, format="text") for t in texts]
    enc = tok(seg, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()  # [neg, pos]
    preds = probs.argmax(axis=1)
    return preds, probs

preds_all, pos_prob_all, neg_prob_all = [], [], []
for i in range(0, len(df), BATCH):
    texts = df["headline"].iloc[i:i+BATCH].tolist()
    preds, probs = infer_batch(texts)
    preds_all.extend(preds.tolist())
    pos_prob_all.extend(probs[:,1].tolist())
    neg_prob_all.extend(probs[:,0].tolist())

df["pred_id"] = preds_all                 # 0=neg, 1=pos
df["pred"]    = np.where(df["pred_id"]==1, "pos", "neg")
df["p_pos"]   = pos_prob_all
df["p_neg"]   = neg_prob_all

# -------- Save --------
OUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_FILE, index=False)
print(f"Saved: {OUT_FILE} | rows={len(df)}")
print(df[["ticker","date","time","effective_date","pred","p_pos"]].head(10))
