"""
TraceNet v2 — Preprocessing Script
===================================
Run this ONCE locally before deploying to Railway.

What it does:
  - Loads the 3 raw Elliptic CSVs
  - Filters to labeled nodes only (46,564 nodes)
  - Saves everything as a compact .npz file (~8-15 MB compressed)
  - Saves feature means for synthetic transaction scoring

Output: models/processed_data.npz

Usage:
  python scripts/preprocess.py
"""

import pandas as pd
import numpy as np
import os

# ── Paths ────────────────────────────────────────────────────────
FEATURES_PATH = "elliptic_bitcoin_dataset/elliptic_txs_features.csv"
CLASSES_PATH  = "elliptic_bitcoin_dataset/elliptic_txs_classes.csv"
EDGES_PATH    = "elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"
OUTPUT_PATH   = "models/processed_data.npz"

def main():
    os.makedirs("models", exist_ok=True)

    print("Loading features CSV (this takes ~30s for 689 MB)...")
    features = pd.read_csv(FEATURES_PATH, header=None)
    features.columns = ["txId"] + [f"f{i}" for i in range(1, 167)]
    print(f"  Features shape: {features.shape}")

    print("Loading classes CSV...")
    classes = pd.read_csv(CLASSES_PATH)
    print(f"  Classes shape: {classes.shape}")
    print(f"  Class counts:\n{classes['class'].value_counts()}")

    print("Loading edgelist CSV...")
    edges = pd.read_csv(EDGES_PATH)
    print(f"  Edges shape: {edges.shape}")

    # ── Filter to labeled nodes only ─────────────────────────────
    print("\nFiltering to labeled nodes...")
    df = features.merge(classes, on="txId", how="left")
    df = df[df["class"] != "unknown"].copy()
    df["label"] = (df["class"] == "1").astype(int)
    print(f"  Labeled nodes: {len(df)}")
    print(f"  Illicit: {df['label'].sum()} | Licit: {(df['label']==0).sum()}")

    # ── Build node index ─────────────────────────────────────────
    node_to_idx = {int(tx_id): idx for idx, tx_id in enumerate(df["txId"].values)}

    # ── Build edge list (only labeled↔labeled edges) ─────────────
    print("\nBuilding labeled edge list...")
    src_list, dst_list = [], []
    for _, row in edges.iterrows():
        s, d = int(row["txId1"]), int(row["txId2"])
        if s in node_to_idx and d in node_to_idx:
            src_list.append(node_to_idx[s])
            dst_list.append(node_to_idx[d])
    print(f"  Labeled edges: {len(src_list)}")

    # ── Feature matrix ───────────────────────────────────────────
    feature_cols = [f"f{i}" for i in range(1, 167)]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    tx_ids = df["txId"].values.astype(np.int64)

    # ── Compute per-class means for synthetic scoring ─────────────
    licit_mask   = (y == 0)
    illicit_mask = (y == 1)
    mean_licit   = X[licit_mask].mean(axis=0).astype(np.float32)
    mean_illicit = X[illicit_mask].mean(axis=0).astype(np.float32)
    std_licit    = X[licit_mask].std(axis=0).astype(np.float32)

    # ── Save ─────────────────────────────────────────────────────
    print(f"\nSaving to {OUTPUT_PATH}...")
    np.savez_compressed(
        OUTPUT_PATH,
        features     = X,
        labels       = y,
        tx_ids       = tx_ids,
        edge_src     = np.array(src_list, dtype=np.int64),
        edge_dst     = np.array(dst_list, dtype=np.int64),
        mean_licit   = mean_licit,
        mean_illicit = mean_illicit,
        std_licit    = std_licit,
    )

    size_mb = os.path.getsize(OUTPUT_PATH) / 1_048_576
    print(f"\n[DONE] Saved {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print(f"   Nodes:        {len(tx_ids):,}")
    print(f"   Edges:        {len(src_list):,}")
    print(f"   Feature dim:  166")
    print(f"\nNext: git add models/processed_data.npz && git commit -m 'add preprocessed data'")

if __name__ == "__main__":
    main()
