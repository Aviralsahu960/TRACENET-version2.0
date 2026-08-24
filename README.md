# TraceNet v2 — Anti-Money Laundering Detection Using Graph Neural Networks

> Detecting financial crime by mapping transactions as a graph and using GraphSAGE to identify suspicious patterns across accounts — built on real, verified criminal data.

---

## What is TraceNet?

TraceNet is an Anti-Money Laundering (AML) detection system. Traditional bank monitoring watches each channel — mobile, ATM, UPI, wire — in isolation. Criminals exploit this by splitting money across channels in patterns that look normal individually but are obviously suspicious when seen together.

TraceNet maps every transaction as a graph (accounts = nodes, transactions = edges) and uses a Graph Neural Network to score each account's "suspicion level" based on its entire transaction neighbourhood — not just its own activity.

**In one line:** *TraceNet sees the full picture that siloed banking systems miss.*

---

## TraceNet v2 vs TraceNet v1 — What Changed and Why

| | TraceNet v1 | TraceNet v2 |
|---|---|---|
| **Dataset** | Synthetic (Faker-generated fake data) | Real — Elliptic Bitcoin Dataset (verified by professional forensics analysts) |
| **Evaluation** | Evaluated on training data (data leakage) | Proper 80/20 train/test split — model never sees test data during training |
| **Accuracy claim** | 99.96% (fake — model memorised training data) | 97.4% (real — evaluated on unseen test nodes only) |
| **Recall** | 100% (fake) | 92.8% (real — caught 844 of 909 actual criminals) |
| **Class imbalance** | Not handled | Weighted loss — 9.2x penalty for missing illicit transactions |
| **Threshold system** | Binary block/approve | Three-zone: auto-approve / human review / auto-block |
| **Data scale** | ~5,000 synthetic nodes | 46,564 real verified nodes, 36,624 edges |
| **Reproducibility** | Random seed not fixed | Fixed seed (42) — same results every run |
| **Build approach** | Solo, existing code | Team of 6, built from scratch with full understanding |

### The most important change

v1 had a fundamental flaw: the model was evaluated on the same data it trained on. A model that sees the exam answers during study will score 100% — that proves nothing. v2 fixes this with a proper train/test split. The 97.4% accuracy in v2 is honest. The 99.96% in v1 was not.

---

## Dataset

**Elliptic Bitcoin Dataset** — published by Elliptic Analytics, available on [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

| Property | Value |
|---|---|
| Total nodes (transactions) | 203,769 |
| Total edges (money flows) | 234,355 |
| Features per node | 166 |
| Labeled illicit | 4,545 (2.2%) |
| Labeled licit | 42,019 (20.6%) |
| Unlabeled (unknown) | 157,205 (77.1%) |
| Time steps | 49 |

Labels were verified by professional blockchain forensics analysts — not generated or guessed. Only transactions confirmed through real criminal investigations are labeled illicit.

---

## Model Architecture

**3-Layer GraphSAGE (Graph SAmple and aggreGatE)**

```
Input: 166 node features
       ↓
SAGEConv Layer 1: 166 → 128  +  BatchNorm  +  ReLU  +  Dropout(0.3)
       ↓
SAGEConv Layer 2: 128 → 64   +  BatchNorm  +  ReLU  +  Dropout(0.3)
       ↓
SAGEConv Layer 3: 64 → 2
       ↓
Output: log-softmax → risk score (licit / illicit)
```

**Total parameters:** 59,714  
**Training time:** ~28 seconds on RTX 4050 GPU

### Why GraphSAGE?
- Scales to large graphs by sampling a fixed number of neighbours per node
- Captures neighbourhood patterns — essential for detecting mule rings where individual accounts look innocent but the network pattern is obviously criminal
- More scalable than GCN (Hamilton et al., 2017) — the original paper this is based on

---

## Results

Evaluated on **9,313 test nodes the model never saw during training.**

| Metric | Value |
|---|---|
| Accuracy | 97.4% |
| Precision (Illicit) | 83.1% |
| Recall (Illicit) | 92.8% |
| F1 Score (Illicit) | 87.7% |
| Criminals caught (TP) | 844 / 909 |
| Criminals missed (FN) | 65 |
| False alarms (FP) | 172 |
| Legitimate cleared (TN) | 8,232 |

### Three-Zone Confidence System

| Risk Score | Action | What happens |
|---|---|---|
| 0 — 40% | ✅ Auto Approve | Transaction clears immediately |
| 40 — 75% | ⚠️ Human Review | Held for analyst review within 24 hours |
| 75 — 100% | 🚨 Auto Block | Blocked + SAR report generated |

This prevents the binary block/approve flaw of v1 — borderline cases go to human review, not auto-block, minimising wrongful blocks on legitimate customers.

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Model | PyTorch + PyTorch Geometric | GraphSAGE training and inference |
| Data | pandas, numpy | Loading and processing Elliptic CSVs |
| Evaluation | scikit-learn | Train/test split, metrics, confusion matrix |
| Backend | FastAPI + uvicorn | REST API for real-time transaction scoring |
| Frontend | Streamlit | Interactive dashboard |
| Language | Python 3.12 | Everything — one language, no integration overhead |

---

## Project Structure

```
tracenet-v2/
├── data/
│   ├── elliptic_txns_features.csv
│   ├── elliptic_txns_classes.csv
│   └── elliptic_txns_edgelist.csv
├── models/
│   ├── gnn_model.pth          ← trained model weights
│   └── model_config.json      ← metrics and architecture config
├── backend/
│   └── api.py                 ← FastAPI server
├── frontend/
│   └── app.py                 ← Streamlit dashboard
├── evaluate/
│   └── compare.py             ← XGBoost vs GNN baseline comparison
├── trainmodel.py              ← full training pipeline
├── requirements.txt
└── README.md
```

---

## Setup & Running

### Requirements
- Python 3.12 (not 3.13 or 3.14 — PyTorch does not support those yet)
- Git

### Step 1 — Clone the repo
```bash
git clone https://github.com/Aviralsahu960/tracenet-v2.git
cd tracenet-v2
```

### Step 2 — Create virtual environment
```bash
py -3.12 -m venv tracenet_env
tracenet_env\Scripts\activate
```

### Step 3 — Install dependencies
CPU (works on any laptop):
```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install -r requirements.txt
```

GPU (NVIDIA only — RTX series recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install torch-geometric
pip install -r requirements.txt
```

### Step 4 — Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and place all 3 CSV files in the `data/` folder.

### Step 5 — Train the model
```bash
python trainmodel.py
```
Model saves to `models/gnn_model.pth` automatically.

### Step 6 — Run the app (two terminals)
Terminal 1:
```bash
python -m uvicorn backend.api:app --reload
```
Terminal 2:
```bash
python -m streamlit run frontend/app.py
```
Open **http://localhost:8501**

---

## Related Work

| Paper | What they did | How v2 differs |
|---|---|---|
| Weber et al. (2019) — *Anti-Money Laundering in Bitcoin* | First to apply GCN to Elliptic dataset, ~70% recall | We use GraphSAGE (more scalable), achieved 92.8% recall |
| Hamilton et al. (2017) — *GraphSAGE* | Introduced the sample-and-aggregate GNN approach | This is the core algorithm we implement |
| Liu et al. (2021) — *GNN for Fraud Detection* | Applied GNN to e-commerce fraud | We apply same principle to financial AML |

---

## Limitations

- Trained on Bitcoin transactions — real banking AML would require retraining on bank-specific data
- Only 23% of the Elliptic dataset has verified labels — the unlabeled 77% cannot be used for evaluation
- Static model — requires periodic retraining as new criminal cases are confirmed
- False positives exist (172 in test set) — production deployment would require human review workflow for borderline cases


---

## Acknowledgements

- [Elliptic](https://www.elliptic.co/) for the dataset
- Hamilton et al. (2017) for the GraphSAGE algorithm
- Weber et al. (2019) for the baseline approach on this dataset

---

*TraceNet v2 • August 2026*
