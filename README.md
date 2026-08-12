# C-PP-COAD: Context-Aware Prediction-Powered Conformal Online Anomaly Detection

Code for the paper:

> **Online Conformal Anomaly Detection with Prediction-Powered Data Acquisition**  
> Amirmohammad Farzaneh and Osvaldo Simeone  
> Institute for Intelligent Networked Systems (INSI), Northeastern University London

## Overview

C-PP-COAD is a framework for online anomaly detection that:
- Wraps any pre-trained anomaly score function
- Leverages digital twin (DT) synthetic data to reduce reliance on real calibration data
- Adaptively acquires real data via active p-values, gated by context
- Guarantees control of the smoothed decaying-memory FDR (sFDR) at every time step

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ recommended.

## Data Setup

Create a `data/` directory and populate each subdirectory as follows.

### Thyroid (auto-downloaded)

The Thyroid disease dataset is fetched automatically from the UCI ML Repository on first run. No manual download needed.

### 5G-NIDD

Download the 5G-NIDD dataset from:
> S. Samarakoon et al., "5G-NIDD: A Comprehensive Network Intrusion Detection Dataset Generated over 5G Wireless Network," 2022.  
> Available at: https://ieee-dataport.org/documents/5g-nidd-comprehensive-network-intrusion-detection-dataset-generated-over-5g-wireless

Place the extracted `Combined.csv` at:
```
data/5g_nidd/Combined.csv
```

### ColO-RAN

Download the ColO-RAN dataset from:
> M. Polese et al., "ColO-RAN: Developing Machine Learning-based xApps for Open RAN Closed-loop Control on Programmable Experimental Platforms," IEEE Trans. Mobile Comput., 2022.  
> Available at: https://github.com/wineslab/colosseum-oran-coloran-dataset

Place the extracted dataset directory at:
```
data/coloran/
```
The loader expects the subdirectory structure `data/coloran/rome_static_medium/sched{0,1,2}/...`.

### Synthetic O-RAN

Generated programmatically — no download required.

## Running Experiments

Each experiment script runs 100 independent trials and saves results to `figures/`.

```bash
# Thyroid disease detection (Figs. 3 & 4)
python exp_thyroid.py

# Synthetic O-RAN conflict detection (Fig. 5)
python exp_oran.py

# 5G-NIDD network intrusion detection (Fig. 6)
python exp_5gnidd.py

# ColO-RAN UE throughput degradation detection (Fig. 7)
python exp_coloran.py
```

Each script saves a `.npy` results file and the corresponding PDF figure(s) to `figures/`.

## Repository Structure

```
├── exp_thyroid.py        # Thyroid experiments (classifier comparison + benchmark)
├── exp_oran.py           # Synthetic O-RAN experiment
├── exp_5gnidd.py         # 5G-NIDD experiment
├── exp_coloran.py        # ColO-RAN experiment
├── src/
│   ├── core.py           # LORD procedure, sFDR/power/CDAR metrics
│   ├── methods.py        # COAD, C-COAD, PP-COAD, C-PP-COAD, PO-COAD, Stat-AD, FC-COAD
│   ├── datasets.py       # Dataset loaders and synthetic data generator
│   ├── runner.py         # Experiment runner (multi-run, data splits)
│   └── plotting.py       # Figure generation (3-panel and 2-panel layouts)
└── figures/              # Output directory for PDFs and cached .npy results
```

## Methods

| Method | Context | Synthetic data | Real data | sFDR control |
|--------|---------|---------------|-----------|--------------|
| COAD | No | No | Always | Yes |
| C-COAD | Yes | No | Always | Yes |
| PO-COAD | No | Always | Never | No |
| C-PO-COAD | Yes | Always | Never | No |
| PP-COAD | No | Gating | Adaptive | Yes |
| **C-PP-COAD (proposed)** | Yes | Gating | Adaptive | Yes |
| Stat-AD | — | No | No | No |
| FC-COAD | — | No | Always | No |

## Citation

```bibtex
@article{farzaneh2026cppCOAD,
  title   = {Online Conformal Anomaly Detection with Prediction-Powered Data Acquisition},
  author  = {Farzaneh, Amirmohammad and Simeone, Osvaldo},
  journal = {IEEE Transactions on Machine Learning in Communications and Networking},
  year    = {2026}
}
```
