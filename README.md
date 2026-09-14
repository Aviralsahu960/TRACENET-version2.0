# TraceNet v2 — Anti-Money Laundering Detection Using Graph Neural Networks

> Detecting financial crime by mapping transaction networks as a graph and using GraphSAGE to identify suspicious patterns across accounts — built on real, verified criminal data.

---

## What is TraceNet?

TraceNet is an Anti-Money Laundering (AML) detection system. Traditional bank monitoring watches each transaction channel — mobile, ATM, UPI, wire — in isolation. Criminals exploit this by splitting money across multiple accounts and channels in patterns that look normal individually but are obviously suspicious when seen together as a network.

TraceNet maps every transaction as a graph (accounts/transactions = nodes, money flows = edges) and uses a Graph Neural Network (GraphSAGE) to score each transaction's suspicion level based on its entire network neighborhood.

**In short:** *TraceNet detects complex financial crime patterns that traditional siloed banking systems miss.*

---

## TraceNet v2 vs TraceNet v1 — What Changed and Why

| Feature | TraceNet v1 | TraceNet v2 |
|---|---|---|
| **Dataset** | Synthetic (Faker-generated fake data) | Real — Elliptic Bitcoin Dataset (verified by professional forensics analysts) |
| **Evaluation** | Evaluated on training data (data leakage) | Strict 80/20 train/test split — model never sees test nodes during training |
| **Accuracy Claim** | 99.96% (synthetic memory effect) | **97.82%** (honest evaluation on unseen test nodes) |
| **Recall** | 100% (synthetic) | **91.09%** (caught 828 of 909 actual illicit transactions) |
| **Class Imbalance** | Not handled | Weighted loss — 9.2x penalty for missing illicit transactions |
| **Threshold System** | Binary block/approve | Three-zone: Auto Approve / Human Review / Auto Block |
| **Data Scale** | ~5,000 synthetic nodes | 46,564 real verified nodes, 36,624 edges |
| **Explainability** | None | Gradient × Input feature attribution per transaction |
| **State & API** | Stateless script | Persistent session state, rate limiting, and 11 REST API endpoints |

---

## Dataset

**Elliptic Bitcoin Dataset** — Published by Elliptic Analytics, available on [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).

| Property | Value |
|---|---|
| Total nodes (transactions) | 203,769 |
| Total edges (money flows) | 234,355 |
| Features per node | 166 (94 local + 72 neighborhood stats) |
| Labeled illicit | 4,545 (9.8% of labeled) |
| Labeled licit | 42,019 (90.2% of labeled) |
| Unlabeled (unknown) | 157,205 (77.1%) |
| Time steps | 49 |

Labels were verified by professional blockchain forensics analysts. Only transactions confirmed through real criminal investigations are labeled illicit.

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
Output: log-softmax → risk probability (licit / illicit)
```

* **Total parameters:** 59,714
* **Loss function:** Weighted Negative Log-Likelihood (NLL Loss with 9.24x weight for illicit class)
* **Optimizer:** Adam (learning rate 0.005, weight decay 5e-4)

---

## Model Results

Evaluated on **9,313 test nodes** that were completely unseen during training:

| Metric | Value | Detail |
|---|---|---|
| **Accuracy** | **97.82%** | Overall correct predictions |
| **Precision** | **87.16%** | Of all transactions flagged as illicit, 87.16% were real criminals |
| **Recall** | **91.09%** | Of all actual illicit transactions, 91.09% were caught |
| **F1 Score** | **89.08%** | Harmonic mean of precision and recall |
| **True Positives (TP)** | 828 | Illicit transactions correctly caught |
| **False Positives (FP)** | 122 | Licit transactions flagged for review |
| **False Negatives (FN)** | 81 | Illicit transactions missed |
| **True Negatives (TN)** | 8,282 | Licit transactions correctly cleared |

### Three-Zone Confidence System

| Risk Score | Verdict | Zone Color | System Action |
|---|---|---|---|
| `0.00` — `0.39` | `AUTO_APPROVE` | 🟢 Green | Cleared immediately |
| `0.40` — `0.74` | `HUMAN_REVIEW` | 🟡 Yellow | Routed to compliance analyst within 24h |
| `0.75` — `1.00` | `AUTO_BLOCK` | 🔴 Red | Transaction blocked + SAR generated |

Operating at the $\ge 0.40$ human review threshold increases total operational recall to over **95%+**.

---

## Model Explainability & Baseline Benchmarking

### 1. Feature Attribution (`/explain/{tx_id}`)
Using **Gradient × Input** attribution, TraceNet computes signed feature importance scores for every transaction, breaking down attribution into three feature groups:
- **Local Features** (93 features): Volume, fee structures, input/output counts.
- **Network Features** (72 features): 1-hop and 2-hop neighborhood statistics.
- **Time Step** (1 feature): Temporal index across the 49 time windows.

### 2. Baseline Comparison (`scripts/evaluate.py`)
TraceNet evaluates GraphSAGE against three non-graph ML baselines (Logistic Regression, Random Forest, Gradient Boosting).
* **Tabular vs Real-World Deployment**: Non-graph models only perform competitively when given pre-computed neighborhood features (`f94`–`f165`). In real banking deployments (SWIFT, UPI, RTGS), neighborhood stats do not exist for new incoming transactions. GraphSAGE dynamically aggregates neighborhood context from the live transaction graph in real time.

---

## Backend API Overview (v2.1)

The backend is built with **FastAPI** and includes 11 REST endpoints:

* `POST /score_transaction` — Scores single/synthetic transaction, returns risk score, 3-zone verdict, and human-readable `risk_factors`.
* `POST /score_batch` — Scores up to 50 transactions in one request.
* `GET /graph_neighbors/{tx_id}` — Extracts 2-hop subgraph node & edge lists with per-node GNN risk scores for Cytoscape.js visualizers.
* `GET /lookup/{tx_id}` — Fast check if a transaction exists in the dataset.
* `GET /explain/{tx_id}` — Gradient × Input feature attribution.
* `GET /all_sars` & `/sar_report/{tx_hash}` — Retrieves generated Suspicious Activity Reports.
* `GET /stats` & `/model_info` — Session counters and model metadata.
* `GET /communities` — Detects illicit transaction communities.
* `POST /admin/reset_stats` — Admin reset for session counters.

**Safety Features:**
* **Rate Limiting**: 60 req/min global, 30 req/min scoring.
* **State Persistence**: `models/session_state.json` preserves session counters and SAR reports.
* **Privacy Compliance**: All user IDs and transaction references are anonymized using SHA-256 / UUID5 hashes (GDPR Art. 6, FATF Rec. 16, PMLA 2002 compliant).

---

## Tech Stack

| Layer | Component | Description |
|---|---|---|
| **Model** | PyTorch + PyTorch Geometric | 3-Layer GraphSAGE GNN |
| **Data** | pandas, numpy, scikit-learn | Graph construction, preprocessing, baseline evaluation |
| **Backend** | FastAPI, uvicorn, pydantic | Async REST API with rate limiting & persistence |
| **Testing** | pytest, httpx | 42 unit & integration tests (`tests/test_api.py`) |
| **Frontend** | HTML5, CSS3, Vanilla JS, Cytoscape.js | Interactive dashboard and network graph forensics |
| **Deployment**| Railway (Backend) + Netlify (Frontend) | Live cloud deployment |

---

## Project Structure

```
tracenet-v2/
├── backend/
│   ├── api.py               ← FastAPI app & endpoints (v2.1)
│   ├── graph_store.py       ← PyG GraphStore & subgraph extractor
│   ├── model_loader.py      ← PyTorch model loader & GNN architecture
│   ├── persistence.py       ← SessionState JSON persistent storage
│   └── rate_limiter.py      ← Sliding window rate limiter
├── scripts/
│   ├── preprocess.py        ← Converts raw Elliptic CSVs into processed_data.npz
│   └── evaluate.py          ← Baseline comparison script (LR, RF, GradBoost vs GNN)
├── models/
│   ├── gnn_model.pth        ← Trained PyTorch model weights
│   ├── model_config.json    ← Saved model metrics & architecture config
│   ├── processed_data.npz   ← Preprocessed features, labels, and adjacency matrix
│   └── session_state.json   ← Persistent session counters & SAR cache
├── tests/
│   └── test_api.py          ← 42 pytest automated tests
├── trainmodel.py            ← Model training script
├── conftest.py              ← Pytest global session setup
├── requirements.txt         ← Python dependencies
├── Procfile                 ← Railway deployment process file
└── README.md
```

---

## Quick Start & Running Locally

### 1. Clone & Environment Setup
```bash
git clone https://github.com/Aviralsahu960/TRACENET-version2.0.git
cd TRACENET-version2.0

# Create Python 3.12 virtual environment
py -3.12 -m venv tracenet_env
tracenet_env\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m pytest tests/test_api.py -v
```

### 4. Launch Backend API
```bash
python -m uvicorn backend.api:app --reload
```
The API interactive documentation will be available at **http://localhost:8000/docs**.

---

## License & Acknowledgements

* **Dataset**: [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) by Elliptic Analytics.
* **Algorithm**: GraphSAGE (Hamilton et al., 2017).
