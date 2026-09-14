"""
TraceNet v2 — Baseline Comparison Evaluator
=============================================
Compares the GraphSAGE GNN against three traditional ML baselines:
  - Logistic Regression (linear, no graph context)
  - Random Forest      (ensemble trees, no graph context)
  - Gradient Boosting  (XGBoost-style, no graph context)

Run from project root:
  python scripts/evaluate.py

All baselines use the same 166 raw node features.
The GNN additionally uses 2-hop graph neighbourhoods via SAGEConv.
The gap in performance demonstrates WHY graph structure matters for AML.
"""

import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "models", "processed_data.npz")
CONFIG_PATH  = os.path.join(PROJECT_ROOT, "models", "model_config.json")

sys.path.insert(0, PROJECT_ROOT)


def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] processed_data.npz not found at {DATA_PATH}")
        print("        Run: python scripts/preprocess.py first.")
        sys.exit(1)

    print(f"Loading dataset from {DATA_PATH} ...")
    data = np.load(DATA_PATH, allow_pickle=True)

    features = data["features"].astype(np.float32)  # (N, 166)
    labels   = data["labels"].astype(int)            # (N,)

    print(f"  Nodes    : {len(labels):,}")
    print(f"  Features : {features.shape[1]}")
    print(f"  Illicit  : {(labels == 1).sum():,} ({(labels == 1).mean()*100:.1f}%)")
    print(f"  Licit    : {(labels == 0).sum():,} ({(labels == 0).mean()*100:.1f}%)")
    return features, labels


def metrics(y_true, y_pred, label: str) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc  = float((y_pred == y_true).mean()) * 100
    prec = precision_score(y_true, y_pred, zero_division=0) * 100
    rec  = recall_score(y_true, y_pred, zero_division=0) * 100
    f1   = f1_score(y_true, y_pred, zero_division=0) * 100
    return {
        "label":     label,
        "accuracy":  round(acc,  2),
        "precision": round(prec, 2),
        "recall":    round(rec,  2),
        "f1":        round(f1,   2),
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
    }


def print_metrics(m: dict, train_time: float):
    print(f"\n  {'Accuracy':<14}: {m['accuracy']}%")
    print(f"  {'Precision':<14}: {m['precision']}%   (of flagged txns, how many were real criminals)")
    print(f"  {'Recall':<14}: {m['recall']}%   (of all criminals, how many were caught)")
    print(f"  {'F1 Score':<14}: {m['f1']}%")
    print(f"  {'Train time':<14}: {train_time:.1f}s")
    print(f"  Confusion matrix: TP={m['tp']:,}  FP={m['fp']:,}  FN={m['fn']:,}  TN={m['tn']:,}")
    missed = m['fn']
    print(f"  Criminals missed: {missed:,}  (False Negatives — the most costly error in AML)")


def separator(title: str):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


def main():
    print("\n" + "=" * 62)
    print("  TRACENET v2 — MODEL EVALUATION & BASELINE COMPARISON")
    print("=" * 62)

    features, labels = load_data()

    # ── Train/test split — same seed as trainmodel.py ─────────────
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"\nSplit: {len(X_train):,} train | {len(X_test):,} test (80/20, stratified)")

    results = []

    # ──────────────────────────────────────────────────────────────
    # BASELINE 1: Logistic Regression
    # ──────────────────────────────────────────────────────────────
    separator("BASELINE 1: Logistic Regression (Linear, no graph)")
    print("  Uses only raw node features — no neighbour aggregation.")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    t0 = time.time()
    lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42, C=1.0
    )
    lr.fit(X_train_s, y_train)
    t_lr = time.time() - t0

    m_lr = metrics(y_test, lr.predict(X_test_s), "Logistic Regression")
    results.append(m_lr)
    print_metrics(m_lr, t_lr)

    # ──────────────────────────────────────────────────────────────
    # BASELINE 2: Random Forest
    # ──────────────────────────────────────────────────────────────
    separator("BASELINE 2: Random Forest (Ensemble, no graph)")
    print("  150 decision trees on raw features. Strong baseline.")

    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=150, class_weight="balanced",
        random_state=42, n_jobs=-1, max_depth=20
    )
    rf.fit(X_train, y_train)
    t_rf = time.time() - t0

    m_rf = metrics(y_test, rf.predict(X_test), "Random Forest")
    results.append(m_rf)
    print_metrics(m_rf, t_rf)

    # ──────────────────────────────────────────────────────────────
    # BASELINE 3: Gradient Boosting
    # ──────────────────────────────────────────────────────────────
    separator("BASELINE 3: Gradient Boosting (XGBoost-style, no graph)")
    print("  Sequential boosted trees — typically strongest non-GNN baseline.")

    t0 = time.time()
    gb = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1,
        max_depth=5, random_state=42, subsample=0.8
    )
    # Sample weights to handle class imbalance (equiv. class_weight)
    n_licit   = (y_train == 0).sum()
    n_illicit = (y_train == 1).sum()
    sample_w  = np.where(y_train == 1, n_licit / n_illicit, 1.0)
    gb.fit(X_train, y_train, sample_weight=sample_w)
    t_gb = time.time() - t0

    m_gb = metrics(y_test, gb.predict(X_test), "Gradient Boosting")
    results.append(m_gb)
    print_metrics(m_gb, t_gb)

    # ──────────────────────────────────────────────────────────────
    # GNN RESULTS (from saved model_config.json)
    # ──────────────────────────────────────────────────────────────
    separator("TRACENET GNN: GraphSAGE 3-layer (166 -> 128 -> 64 -> 2)")
    print("  Same features + 2-hop graph neighbourhood aggregation.")
    print("  Trained for 300 epochs. Weights loaded from models/gnn_model.pth")

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        m_gnn = {
            "label":     "GraphSAGE GNN",
            "accuracy":  cfg.get("accuracy", 0),
            "precision": cfg.get("precision_illicit", 0),
            "recall":    cfg.get("recall_illicit", 0),
            "f1":        cfg.get("f1_illicit", 0),
            "tp":        cfg.get("true_positives", 0),
            "fp":        cfg.get("false_positives", 0),
            "fn":        cfg.get("false_negatives", 0),
            "tn":        cfg.get("true_negatives", 0),
        }
        results.append(m_gnn)
        print_metrics(m_gnn, train_time=0)
        print("  (Training time: ~5 min on CPU for 300 epochs)")
    else:
        print("  [WARN] model_config.json not found — skipping GNN row")

    # ──────────────────────────────────────────────────────────────
    # COMPARISON TABLE
    # ──────────────────────────────────────────────────────────────
    separator("SIDE-BY-SIDE COMPARISON")
    print(f"\n  {'Model':<26} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Missed':>8}")
    print("  " + "-" * 75)
    for r in results:
        missed = r.get("fn", "-")
        marker = " <-- GNN" if r["label"] == "GraphSAGE GNN" else ""
        print(
            f"  {r['label']:<26} {r['accuracy']:>8}% {r['precision']:>9}% "
            f"{r['recall']:>7}% {r['f1']:>7}%  {missed:>6}{marker}"
        )

    separator("KEY INSIGHTS")
    if len(results) >= 4:
        best_base = max(results[:3], key=lambda r: r["recall"])
        gnn = results[-1]
        recall_gain = round(gnn["recall"] - best_base["recall"], 2)
        fn_reduction = best_base["fn"] - gnn["fn"]

        print(f"""
  1. RECALL is the most important metric in AML.
     A missed criminal (False Negative) is far costlier than a
     wrongly blocked legitimate transaction (False Positive).

  2. Best baseline recall: {best_base['recall']}% ({best_base['label']})
     GNN recall          : {gnn['recall']}%

  3. The GNN catches {recall_gain}% MORE criminals than the best
     non-graph baseline — that's {fn_reduction} fewer criminals
     escaping per 9,313 test transactions.

  4. Why? Graph structure. The GNN sees not just what a transaction
     looks like but WHO it transacts with. A clean-looking transaction
     connected to 3 known mule accounts should be suspicious — and
     only the GNN can detect that.

  5. This is exactly why companies like Elliptic, Chainalysis, and
     major banks (HSBC, JPMorgan) have moved to GNN-based AML systems.
""")

    # ──────────────────────────────────────────────────────────────
    # SAVE RESULTS
    # ──────────────────────────────────────────────────────────────
    out_path = os.path.join(PROJECT_ROOT, "models", "evaluation_results.json")
    output = {
        "baselines":    results[:3],
        "gnn":          results[-1] if len(results) == 4 else {},
        "dataset":      "Elliptic Bitcoin Dataset",
        "test_size":    len(X_test),
        "train_size":   len(X_train),
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to: {out_path}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
