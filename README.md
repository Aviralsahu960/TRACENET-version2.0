# TraceNet v2 — Anti-Money Laundering Detection Using Graph Neural Networks

> Detecting financial crime by mapping transaction networks as a graph and using GraphSAGE to identify suspicious patterns across accounts — built on real, verified criminal data.

[![Backend](https://img.shields.io/badge/Backend-Railway%20Live-success?logo=railway)](https://tracenet-version20-production.up.railway.app)
[![Frontend](https://img.shields.io/badge/Frontend-Netlify%20Live-00C7B7?logo=netlify)](https://tracenet-v2.netlify.app)
[![Model](https://img.shields.io/badge/Model-GraphSAGE%203--Layer-blue?logo=pytorch)](https://pytorch-geometric.readthedocs.io)
[![Accuracy](https://img.shields.io/badge/Accuracy-97.62%25-brightgreen)](https://tracenet-version20-production.up.railway.app/model_info)
[![License](https://img.shields.io/badge/Dataset-Elliptic%20Bitcoin-orange)](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

---

## 🔴 Live Deployment

| Service | URL | Status |
|---|---|---|
| **Backend API** (Railway) | https://tracenet-version20-production.up.railway.app | ✅ Live |
| **Frontend** (Netlify) | https://tracenet-v2.netlify.app | ✅ Live |
| **API Docs** | https://tracenet-version20-production.up.railway.app/docs | ✅ Interactive |
| **Health Check** | https://tracenet-version20-production.up.railway.app/health | ✅ 200 OK |

The Netlify frontend **auto-connects to the Railway backend** — no local Python server needed to use the deployed site. If the Railway backend is temporarily cold-starting, the frontend falls back to a built-in client-side GNN emulator so the demo never breaks.

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
| **Accuracy Claim** | 99.96% (synthetic memory effect) | **97.62%** (honest evaluation on unseen test nodes) |
| **Recall** | 100% (synthetic) | **92.52%** (caught 841 of 909 actual illicit transactions) |
| **Class Imbalance** | Not handled | Weighted loss — 9.2x penalty for missing illicit transactions |
| **Threshold System** | Binary block/approve | Three-zone: Auto Approve / Human Review / Auto Block |
| **Data Scale** | ~5,000 synthetic nodes | 46,564 real verified nodes, 36,624 edges |
| **Explainability** | None | Gradient × Input feature attribution per transaction |
| **State & API** | Stateless script | Persistent session state, rate limiting, and 11 REST API endpoints |
| **Frontend** | None | Full interactive forensic dashboard (desktop + mobile responsive) |
| **Deployment** | Local only | Railway (Backend) + Netlify (Frontend) — always-on cloud |

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
| **Accuracy** | **97.62%** | Overall correct predictions |
| **Precision (Illicit)** | **84.52%** | Of all transactions flagged as illicit, 84.52% were real criminals |
| **Recall (Illicit)** | **92.52%** | Of all actual illicit transactions, 92.52% were caught |
| **F1 Score (Illicit)** | **88.34%** | Harmonic mean of precision and recall |
| **True Positives (TP)** | 841 | Illicit transactions correctly caught |
| **False Positives (FP)** | 154 | Licit transactions flagged for review |
| **False Negatives (FN)** | 68 | Illicit transactions missed |
| **True Negatives (TN)** | 8,250 | Licit transactions correctly cleared |

### Three-Zone Confidence System

| Risk Score | Verdict | Zone | System Action |
|---|---|---|---|
| `0.00` — `0.39` | `AUTO_APPROVE` | 🟢 Green | Cleared immediately |
| `0.40` — `0.74` | `HUMAN_REVIEW` | 🟡 Yellow | Routed to compliance analyst within 24h |
| `0.75` — `1.00` | `AUTO_BLOCK` | 🔴 Red | Transaction blocked + SAR auto-generated |

Operating at the ≥ 0.40 human review threshold increases total operational recall to over **95%+**.

---

## Frontend — Interactive Forensic Dashboard

The frontend is a fully interactive, zero-dependency vanilla JS/CSS dashboard deployed on Netlify. It works on both **desktop and mobile phones**.

### Pages & Features

| Tab | Features |
|---|---|
| **Dashboard** | Live KPI cards (nodes, illicit %, accuracy), session throughput charts, backend status, community cluster alert |
| **Transaction Analysis** | GNN risk scoring with 1-click random presets (illicit/review/licit), risk gauge, verdict badge, risk factor breakdown, SAR generation |
| **Graph Explorer** | Interactive 20–36 node SVG graph, zoom/pan/drag, node click inspection, filter by illicit/review/licit/all, random preset button |
| **Alerts & SARs** | List of all generated Suspicious Activity Reports, click to open full SAR dossier in modal, download as text |
| **Model Insights** | Full architecture, live metrics from Railway API, confusion matrix, precision/recall breakdown |
| **Reports & Sharing** | Privacy-safe interbank sharing (SHA-256 anonymised), batch scoring export |

### Smart API Routing

```javascript
// On Netlify → directly calls Railway cloud backend
// On localhost → calls local server, auto-failover to Railway if local is down
const API_BASE = isLocalhost ? 'http://127.0.0.1:5000' : 'https://tracenet-version20-production.up.railway.app';
```

### Mobile Responsive

| Feature | Desktop | Mobile (≤ 640px) |
|---|---|---|
| Navigation | Left sidebar | Fixed bottom tab bar |
| Brand header | Sidebar | Fixed top bar with backend status |
| Grids | 4-column KPIs | Single column |
| Graph canvas | 650px tall | 340px tall, touch-friendly |
| Modals | Centered dialog | Bottom sheet |
| Tables | Full columns | Horizontal scroll |

---

## Backend API Overview (v2.1)

Base URL (Production): `https://tracenet-version20-production.up.railway.app`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — model, graph & session status |
| `GET` | `/model_info` | Full model metrics, architecture, zone thresholds |
| `GET` | `/stats` | Session counters (scored, approved, blocked, SARs) |
| `POST` | `/score_transaction` | GNN risk scoring — returns score, verdict, risk factors |
| `POST` | `/score_batch` | Batch scoring up to 50 transactions |
| `GET` | `/graph_neighbors/{tx_id}` | 2-hop subgraph with per-node risk scores |
| `GET` | `/explain/{tx_id}` | Gradient × Input feature attribution |
| `GET` | `/sar_report/{tx_hash}` | Full SAR dossier for a flagged transaction |
| `GET` | `/all_sars` | List all SARs generated this session |
| `POST` | `/interbank_share` | Privacy-safe SHA-256 transaction sharing |
| `GET` | `/communities` | Suspicious cluster detection across the graph |
| `GET` | `/lookup/{tx_id}` | Check if a transaction exists in the dataset |
| `POST` | `/admin/reset_stats` | Reset session counters (API key protected) |

### Example: Score a Transaction

```bash
curl -X POST https://tracenet-version20-production.up.railway.app/score_transaction \
  -H "Content-Type: application/json" \
  -d '{"tx_id": "355110272", "amount": 250000, "channel": "crypto"}'
```

Response:
```json
{
  "tx_hash": "94db1db0...",
  "risk_score": 0.9997,
  "risk_percent": 100,
  "verdict": "AUTO_BLOCK",
  "zone": "red",
  "used_dataset_node": true,
  "neighbor_count": 56,
  "sar_available": true
}
```

**Safety Features:**
* **Rate Limiting**: 60 req/min global, 30 req/min scoring (per-IP sliding window)
* **State Persistence**: `models/session_state.json` preserves session counters and SAR reports
* **Privacy Compliance**: All user IDs and transaction references are anonymized using SHA-256 / UUID5 hashes (GDPR Art. 6, FATF Rec. 16, PMLA 2002 compliant)
* **CORS**: Configured for cross-origin requests from Netlify (`allow_origins=["*"]`)

---

## Tech Stack

| Layer | Component | Description |
|---|---|---|
| **Model** | PyTorch 2.6 + PyTorch Geometric | 3-Layer GraphSAGE GNN |
| **Data** | pandas, numpy, scikit-learn | Graph construction, preprocessing, baseline evaluation |
| **Backend** | FastAPI, uvicorn, pydantic v2 | Async REST API with rate limiting & persistence |
| **Testing** | pytest, httpx | Automated unit & integration tests |
| **Frontend** | HTML5, CSS3, Vanilla JS | Interactive dashboard, no frameworks, no build step |
| **Graph Viz** | SVG (custom renderer) | 20–36 node interactive forensic graph explorer |
| **Backend Cloud** | Railway | Always-on PyTorch GNN inference server |
| **Frontend Cloud** | Netlify | CDN-hosted static frontend with auto-deploy from GitHub |

---

## Project Structure

```
tracenet-v2/
├── backend/
│   ├── api.py               ← FastAPI app & 13 endpoints (v2.1)
│   ├── graph_store.py       ← PyG GraphStore & subgraph extractor
│   ├── model_loader.py      ← PyTorch model loader, GNN architecture & explainability
│   ├── persistence.py       ← SessionState JSON persistent storage
│   └── rate_limiter.py      ← Sliding window rate limiter
├── TraceNet_Frontend/
│   ├── index.html           ← App shell — sidebar nav, mobile bottom bar, modals
│   ├── style.css            ← Dark forensic UI, full mobile responsive (640px breakpoint)
│   └── app.js               ← Full SPA — all 6 tabs, graph renderer, Railway/local API routing
├── scripts/
│   ├── preprocess.py        ← Converts raw Elliptic CSVs into processed_data.npz
│   └── evaluate.py          ← Baseline comparison (LR, RF, GradBoost vs GNN)
├── models/
│   ├── gnn_model.pth        ← Trained PyTorch model weights (241 KB)
│   ├── model_config.json    ← Saved model metrics & architecture config
│   ├── processed_data.npz   ← Preprocessed features, labels & adjacency matrix
│   └── session_state.json   ← Persistent session counters & SAR cache (runtime)
├── tests/
│   └── test_api.py          ← pytest automated tests
├── trainmodel.py            ← Model training script
├── conftest.py              ← Pytest global session setup
├── requirements.txt         ← Python dependencies
├── Procfile                 ← Railway deployment process file
├── railway.toml             ← Railway build & deploy config
├── netlify.toml             ← Netlify publish dir, redirects & headers
├── .env.example             ← Environment variable template
└── README.md
```

---

## Quick Start — Running Locally

### 1. Clone & Environment Setup

```bash
git clone https://github.com/Aviralsahu960/TRACENET-version2.0.git
cd TRACENET-version2.0

# Create Python 3.12 virtual environment
py -3.12 -m venv tracenet_env
tracenet_env\Scripts\activate     # Windows
# source tracenet_env/bin/activate  # macOS/Linux
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Backend API

```bash
python -m uvicorn backend.api:app --host 127.0.0.1 --port 5000 --reload
```

Interactive API docs: **http://127.0.0.1:5000/docs**

### 4. Serve Frontend Locally

```bash
python -m http.server 3000 --directory TraceNet_Frontend
```

Open: **http://localhost:3000**

The frontend will auto-connect to your local backend at `127.0.0.1:5000`. If local backend is stopped, it automatically fails over to the live Railway cloud backend.

### 5. Run Tests (Optional)

```bash
python -m pytest tests/test_api.py -v
```

---

## Deployment

### Backend → Railway

The backend deploys automatically from GitHub via `railway.toml`.

```toml
[deploy]
startCommand = "uvicorn backend.api:app --host 0.0.0.0 --port $PORT"
```

Required files committed to git:
- `models/gnn_model.pth` — trained weights
- `models/model_config.json` — model config
- `models/processed_data.npz` — preprocessed graph

### Frontend → Netlify

The frontend deploys automatically from GitHub via `netlify.toml`.

```toml
[build]
  publish = "TraceNet_Frontend"
```

The `RAILWAY_URL` is hardcoded in `app.js` — no environment variable needed for the frontend.

---

## Environment Variables

Copy `.env.example` to `.env` for local development:

```bash
# Model file paths (defaults work from project root)
MODEL_PATH=models/gnn_model.pth
CONFIG_PATH=models/model_config.json
DATA_PATH=models/processed_data.npz

# Optional: enable API key auth (leave blank to disable)
API_KEY=

# Optional: restrict CORS origins (default: * = allow all)
ALLOWED_ORIGINS=*
```

On Railway, `PORT` is set automatically — do not set it manually.

---

## Model Explainability & Baseline Benchmarking

### Feature Attribution (`/explain/{tx_id}`)

Using **Gradient × Input** attribution, TraceNet computes signed feature importance scores for every transaction, breaking down attribution into three feature groups:

- **Local Features** (93 features): Volume, fee structures, input/output counts
- **Network Features** (72 features): 1-hop and 2-hop neighborhood statistics
- **Time Step** (1 feature): Temporal index across the 49 time windows

### Baseline Comparison (`scripts/evaluate.py`)

TraceNet evaluates GraphSAGE against non-graph ML baselines (Logistic Regression, Random Forest, Gradient Boosting).

> **Why GNN wins in production:** Tabular models only perform competitively when given pre-computed neighborhood features (`f94`–`f165`). In real banking deployments (SWIFT, UPI, RTGS), neighborhood stats do not exist for *new incoming transactions*. GraphSAGE dynamically aggregates neighborhood context from the live transaction graph in real time — no pre-computation needed.

---

## License & Acknowledgements

* **Dataset**: [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) by Elliptic Analytics
* **Algorithm**: Hamilton, W., Ying, Z., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs.* (GraphSAGE)
* **Compliance framework**: GDPR Art. 6(1)(f), FATF Recommendation 16, PMLA 2002
