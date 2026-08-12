"""
Dataset loading and preprocessing for all experiments.
"""
import numpy as np
import pandas as pd
import glob
import os
import ssl
import urllib.request
import io


# ── 5G-NIDD ───────────────────────────────────────────────────────────────────

NIDD_NUMERIC = [
    'Dur', 'RunTime', 'Mean', 'Sum', 'Min', 'Max',
    'sTos', 'dTos', 'sTtl', 'dTtl',
    'TotPkts', 'SrcPkts', 'DstPkts',
    'TotBytes', 'SrcBytes', 'DstBytes',
    'sMeanPktSz', 'dMeanPktSz',
    'Load', 'SrcLoad', 'DstLoad',
    'Loss', 'SrcLoss', 'DstLoss', 'pLoss',
    'Rate', 'SrcRate', 'DstRate',
    'TcpRtt', 'SynAck', 'AckDat',
]

def load_5gnidd(csv_path, subsample_malicious=True, target_anomaly_rate=0.10,
                max_samples=20000, seed=0):
    """
    Load and preprocess 5G-NIDD dataset.

    Returns
    -------
    X       : (N, d) float32 features
    y       : (N,)   int labels (0=benign, 1=malicious)
    context : (N,)   int context (0=UDP, 1=TCP, 2=ICMP, 3=other)
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(csv_path, usecols=NIDD_NUMERIC + ['Label', 'Proto'])
    df = df.dropna(subset=NIDD_NUMERIC)

    for col in NIDD_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=NIDD_NUMERIC)

    df['label'] = (df['Label'].str.strip().str.lower() != 'benign').astype(int)

    # Context = protocol family
    proto = df['Proto'].str.strip().str.upper().fillna('OTHER')
    ctx_map = {'UDP': 0, 'TCP': 1, 'ICMP': 2}
    df['context'] = proto.map(ctx_map).fillna(3).astype(int)

    benign = df[df['label'] == 0]
    malicious = df[df['label'] == 1]

    if subsample_malicious:
        n_benign = min(len(benign), int(max_samples * (1 - target_anomaly_rate)))
        n_malicious = min(len(malicious), int(n_benign * target_anomaly_rate / (1 - target_anomaly_rate)))
        benign = benign.iloc[rng.choice(len(benign), n_benign, replace=False)]
        malicious = malicious.iloc[rng.choice(len(malicious), n_malicious, replace=False)]
    else:
        total = min(len(df), max_samples)
        df = df.iloc[rng.choice(len(df), total, replace=False)]
        benign = df[df['label'] == 0]
        malicious = df[df['label'] == 1]

    df_out = pd.concat([benign, malicious]).reset_index(drop=True)
    df_out = df_out.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Log-transform heavy-tailed features
    for col in ['TotBytes', 'SrcBytes', 'DstBytes', 'TotPkts', 'SrcPkts', 'DstPkts',
                'Load', 'SrcLoad', 'DstLoad', 'Rate', 'SrcRate', 'DstRate', 'Sum']:
        df_out[col] = np.log1p(df_out[col].clip(lower=0))

    X = df_out[NIDD_NUMERIC].values.astype(np.float32)
    # Standardize
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma[sigma < 1e-8] = 1.0
    X = (X - mu) / sigma
    X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)

    y = df_out['label'].values.astype(int)
    context = df_out['context'].values.astype(int)
    return X, y, context


# ── ColO-RAN ──────────────────────────────────────────────────────────────────

COLORAN_FEATURES = ['dl_brate', 'ul_brate', 'dl_snr', 'dl_mcs', 'dl_bler', 'rsrp', 'pl']

def load_coloran(data_dir, max_per_sched=5000, seed=0):
    """
    Load ColO-RAN UE metric data.

    Anomaly definition: dl_brate drops below the 10th percentile of the
    sched-specific training distribution (throughput degradation).
    Context: scheduling policy index (0, 1, 2).

    Returns
    -------
    X         : (N, 7) float32 features
    y         : (N,)   int labels (0=normal, 1=anomaly)
    context   : (N,)   int scheduling policy {0, 1, 2}
    thresholds: dict  sched → 10th-percentile threshold used
    """
    rng = np.random.default_rng(seed)
    frames = []

    for sched in [0, 1, 2]:
        pattern = os.path.join(data_dir, f'rome_static_medium/sched{sched}',
                               '*', '*', 'bs*', 'ue*.csv')
        files = glob.glob(pattern)
        if not files:
            continue
        rng.shuffle(files)
        rows_collected = []
        for fpath in files:
            try:
                df = pd.read_csv(fpath)
            except Exception:
                continue
            for col in COLORAN_FEATURES:
                if col not in df.columns:
                    df[col] = 0.0
            sub = df[COLORAN_FEATURES + ['is_attached']].copy()
            sub['sched'] = sched
            # Keep only rows after warm-up (non-zero dl_brate or dl_snr)
            sub = sub[sub['dl_brate'] > 0]
            rows_collected.append(sub)
            if sum(len(r) for r in rows_collected) >= max_per_sched * 2:
                break
        if not rows_collected:
            continue
        df_sched = pd.concat(rows_collected, ignore_index=True)
        if len(df_sched) > max_per_sched * 2:
            df_sched = df_sched.iloc[:max_per_sched * 2]
        frames.append(df_sched)

    df_all = pd.concat(frames, ignore_index=True)

    # Compute per-context anomaly threshold (10th percentile of dl_brate)
    thresholds = {}
    for sched in df_all['sched'].unique():
        rates = df_all[df_all['sched'] == sched]['dl_brate']
        thresholds[sched] = float(np.percentile(rates, 10))

    # Define anomaly label
    df_all['label'] = df_all.apply(
        lambda row: int(row['dl_brate'] < thresholds.get(row['sched'], 0)), axis=1
    )

    # Extract features
    X = df_all[COLORAN_FEATURES].values.astype(np.float32)
    # Log-transform brates
    X[:, 0] = np.log1p(X[:, 0])  # dl_brate
    X[:, 1] = np.log1p(X[:, 1])  # ul_brate

    # Standardize
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma[sigma < 1e-8] = 1.0
    X = (X - mu) / sigma
    X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)

    y = df_all['label'].values.astype(int)
    context = df_all['sched'].values.astype(int)
    return X, y, context, thresholds


# ── Thyroid ───────────────────────────────────────────────────────────────────

_THYROID_URL = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'thyroid-disease/thyroid0387.data')

_THYROID_COLS = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
    'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
    'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
    'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH_measured', 'TSH', 'T3_measured', 'T3',
    'TT4_measured', 'TT4', 'T4U_measured', 'T4U',
    'FTI_measured', 'FTI', 'TBG_measured', 'TBG',
    'referral_source', 'id',
]


def load_thyroid(cache_path=None, target_n=7200, target_anomaly_rate=0.07, seed=0):
    """
    Load UCI thyroid0387 dataset.

    Returns X (N,d), y (N,), context (N,) where context = age_bin (0=age<=50, 1=age>50).
    """
    rng = np.random.default_rng(seed)

    # Download or load from cache
    if cache_path and os.path.exists(cache_path):
        raw = open(cache_path).read()
    else:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(_THYROID_URL, context=ctx) as r:
            raw = r.read().decode('utf-8', errors='ignore')
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                f.write(raw)

    df = pd.read_csv(io.StringIO(raw), header=None, na_values=['?'],
                     names=_THYROID_COLS)

    # Label: '-' prefix = normal (0), anything else = anomaly (1)
    df['label'] = df['id'].astype(str).str[0].apply(lambda c: 0 if c == '-' else 1)

    # Age: clip to [1, 100]
    df['age'] = pd.to_numeric(df['age'], errors='coerce').clip(1, 100)
    df['age'] = df['age'].fillna(df['age'].median())
    df['age_bin'] = (df['age'] > 50).astype(int)

    # Sex: F=0, M=1
    df['sex_num'] = df['sex'].map({'F': 0, 'M': 1}).fillna(0)

    # Binary features: t=1, f=0
    binary_cols = [
        'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
        'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment',
        'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre',
        'tumor', 'hypopituitary', 'psych',
        'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured',
        'FTI_measured', 'TBG_measured',
    ]
    for col in binary_cols:
        df[col] = df[col].map({'t': 1, 'f': 0, 1: 1, 0: 0}).fillna(0).astype(float)

    # Continuous features: impute with median
    cont_cols = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
    for col in cont_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    feature_cols = ['sex_num', 'age'] + binary_cols + cont_cols

    # Subsample to target_n with target_anomaly_rate
    normals = df[df['label'] == 0].copy()
    anomalies = df[df['label'] == 1].copy()
    n_anom = min(len(anomalies), int(target_n * target_anomaly_rate))
    n_norm = min(len(normals), target_n - n_anom)
    normals = normals.iloc[rng.choice(len(normals), n_norm, replace=False)]
    anomalies = anomalies.iloc[rng.choice(len(anomalies), n_anom, replace=False)]
    df_out = pd.concat([normals, anomalies]).sample(frac=1, random_state=seed).reset_index(drop=True)

    X = df_out[feature_cols].values.astype(np.float32)
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma[sigma < 1e-8] = 1.0
    X = (X - mu) / sigma
    X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)

    y = df_out['label'].values.astype(int)
    context = df_out['age_bin'].values.astype(int)
    return X, y, context


# ── Synthetic O-RAN ───────────────────────────────────────────────────────────

def generate_oran(n_samples=10000, anomaly_rate=0.1, seed=0):
    """
    Synthetic O-RAN KPI monitoring dataset.

    20 continuous features: 10 KPI measurements (throughput, latency, SNR, etc.),
    5 xApp control signals, 5 channel quality indicators.

    Normal samples have two regimes:
      - Mode 1 "stable operation" (70%): KPIs ~ N(0, 1)
      - Mode 2 "high-load normal" (30%): KPIs ~ N(3.5, 0.5), representing
        elevated-but-valid traffic bursts not fully captured by the digital twin.
    Anomalies represent xApp-induced KPI degradation: KPIs ~ N(7, 0.5).

    Context C in {0,1,2,3}: quartile of the channel quality index (feature 15),
    which is independent of the stable/high-load split so both modes appear in
    every context.  This ensures the within-context DT mismatch (mode-2 samples
    are absent from the GMM training after score-based exclusion) causes C-PO-COAD
    to violate, while C-PP-COAD detects the mismatch via gamma and falls back to
    real calibration data.

    Returns X (N, 20), y (N,), context (N,).
    """
    rng = np.random.default_rng(seed)
    n_anom = int(n_samples * anomaly_rate)
    n_norm = n_samples - n_anom
    n_mode1 = int(n_norm * 0.70)
    n_mode2 = n_norm - n_mode1

    # Mode 1: stable operation
    X_m1 = np.zeros((n_mode1, 20), dtype=np.float32)
    X_m1[:, :10]  = rng.normal(0.0, 1.0, (n_mode1, 10))   # KPI measurements
    X_m1[:, 10:15] = rng.normal(0.0, 1.0, (n_mode1, 5))   # xApp control signals
    X_m1[:, 15:20] = rng.normal(0.0, 1.0, (n_mode1, 5))   # channel quality (context)

    # Mode 2: high-load normal (KPIs elevated, partially overlapping with anomaly region)
    X_m2 = np.zeros((n_mode2, 20), dtype=np.float32)
    X_m2[:, :10]  = rng.normal(4.0, 1.5, (n_mode2, 10))
    X_m2[:, 10:15] = rng.normal(0.0, 1.0, (n_mode2, 5))
    X_m2[:, 15:20] = rng.normal(0.0, 1.0, (n_mode2, 5))   # same dist → independent

    # Anomalies: xApp-induced KPI degradation (partially overlapping mode-2 region)
    X_anom = np.zeros((n_anom, 20), dtype=np.float32)
    X_anom[:, :10]  = rng.normal(6.5, 1.5, (n_anom, 10))
    X_anom[:, 10:15] = rng.normal(0.0, 1.0, (n_anom, 5))
    X_anom[:, 15:20] = rng.normal(0.0, 1.0, (n_anom, 5))

    X_norm = np.vstack([X_m1, X_m2])
    X = np.vstack([X_norm, X_anom])
    y = np.concatenate([np.zeros(n_norm, dtype=int), np.ones(n_anom, dtype=int)])

    # Context: quartile of channel quality index (feature 15), independent of KPI modes
    f15_norm = X_norm[:, 15]
    quartiles = np.percentile(f15_norm, [25, 50, 75])
    context = np.digitize(X[:, 15], quartiles).astype(int)  # 0, 1, 2, 3

    # Shuffle
    idx = rng.permutation(n_samples)
    return X[idx], y[idx], context[idx]
