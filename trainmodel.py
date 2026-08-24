import pandas as pd

# Load all 3 files
features = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_features.csv", header=None)
classes = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_classes.csv")
edges = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv")

# See what they look like
print("=== FEATURES ===")
print(f"Shape: {features.shape}")  # should be (203769, 167)
print(features.head(3))

print("\n=== CLASSES ===")
print(f"Shape: {classes.shape}")
print(classes['class'].value_counts())  # shows illicit/licit/unknown counts

print("\n=== EDGES ===")
print(f"Shape: {edges.shape}")
print(edges.head(3))

# ── STEP 2: Clean and prepare the data ──────────────────────

# The features file has no header — column 0 is the txId, rest are features
features.columns = ['txId'] + [f'f{i}' for i in range(1, 167)]

# Merge features with labels
df = features.merge(classes, on='txId', how='left')

# Keep only labeled nodes (illicit=1, licit=2) — drop unknowns
df = df[df['class'] != 'unknown'].copy()

# Convert labels: illicit(1) → 1, licit(2) → 0
df['label'] = (df['class'] == '1').astype(int)

print(f"Labeled nodes: {len(df)}")
print(f"Illicit: {df['label'].sum()} | Licit: {(df['label']==0).sum()}")
print(f"Class ratio: {df['label'].sum()/len(df)*100:.1f}% illicit")

# ── STEP 3: Build the graph ──────────────────────────────────
import numpy as np

# Create a mapping: transaction ID (big number) → integer index (0, 1, 2, 3...)
# PyTorch needs simple integers, not big ID numbers like 230425980
all_node_ids = df['txId'].values
node_to_idx = {node_id: idx for idx, node_id in enumerate(all_node_ids)}

print(f"Total nodes in our graph: {len(node_to_idx)}")

# Build edge list — but only keep edges where BOTH nodes are in our labeled set
src_nodes = []
dst_nodes = []

for _, row in edges.iterrows():
    src = row['txId1']
    dst = row['txId2']
    if src in node_to_idx and dst in node_to_idx:
        src_nodes.append(node_to_idx[src])
        dst_nodes.append(node_to_idx[dst])

print(f"Edges connecting labeled nodes: {len(src_nodes)}")

# Convert to numpy arrays (we'll convert to PyTorch tensors in next step)
src_nodes = np.array(src_nodes)
dst_nodes = np.array(dst_nodes)

# ── STEP 4: Convert to PyTorch tensors ───────────────────────
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# Feature matrix — columns 1 to 166 (skip column 0 which is txId)
feature_cols = [f'f{i}' for i in range(1, 167)]
x = torch.tensor(df[feature_cols].values, dtype=torch.float)

# Labels — illicit=1, licit=0
y = torch.tensor(df['label'].values, dtype=torch.long)

# Edge index — PyTorch Geometric needs shape [2, num_edges]
edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

# Train/test split — 80% train, 20% test, stratified so illicit ratio is preserved
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=df['label'].values
)

# Create masks — True means this node is used for training/testing
train_mask = torch.zeros(len(df), dtype=torch.bool)
test_mask = torch.zeros(len(df), dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

# Pack everything into one PyTorch Geometric Data object
data = Data(x=x, edge_index=edge_index, y=y,
            train_mask=train_mask, test_mask=test_mask)

print(f"Feature matrix shape: {data.x.shape}")
print(f"Labels shape: {data.y.shape}")
print(f"Edge index shape: {data.edge_index.shape}")
print(f"Training nodes: {train_mask.sum().item()}")
print(f"Test nodes: {test_mask.sum().item()}")
print(f"Illicit in train: {y[train_mask].sum().item()}")
print(f"Illicit in test: {y[test_mask].sum().item()}")

# ── STEP 5: Define the GNN Model ─────────────────────────────
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

class TraceNetGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 3 GraphSAGE layers: 166 → 128 → 64 → 2
        self.conv1 = SAGEConv(166, 128)
        self.conv2 = SAGEConv(128, 64)
        self.conv3 = SAGEConv(64, 2)
        self.dropout = torch.nn.Dropout(0.3)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 3 — output
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

# Set up device — uses GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Move data and model to device
model = TraceNetGNN().to(device)
data = data.to(device)

# Weighted loss to handle class imbalance
# Illicit nodes are rare so we give them higher weight during training
num_licit = (y == 0).sum().item()
num_illicit = (y == 1).sum().item()
weight = torch.tensor([1.0, num_licit / num_illicit], dtype=torch.float).to(device)

print(f"Loss weight for illicit class: {num_licit / num_illicit:.1f}x")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("\nModel architecture:")
print(model)

# ── STEP 6: Train the Model ───────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

print("\nTraining started...")
print("-" * 50)

model.train()
for epoch in range(300):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=weight)
    loss.backward()
    optimizer.step()

    if epoch % 30 == 0:
        # Quick accuracy check on train set during training
        pred = out.argmax(dim=1)
        train_correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
        train_acc = int(train_correct) / data.train_mask.sum().item()
        print(f"Epoch {epoch:>3d} | Loss: {loss.item():.4f} | Train Acc: {train_acc*100:.1f}%")

print("-" * 50)
print("Training complete.")

# ── STEP 7: Evaluate on Test Set ─────────────────────────────
from sklearn.metrics import classification_report, confusion_matrix

model.eval()
with torch.no_grad():
    out = model(data)
    probs = torch.exp(out)          # convert to actual probabilities
    illicit_prob = probs[:, 1]      # probability of being illicit for each node
    pred = out.argmax(dim=1)        # binary prediction (0 or 1)

# Only look at test nodes
test_pred = pred[data.test_mask].cpu().numpy()
test_true = data.y[data.test_mask].cpu().numpy()
test_probs = illicit_prob[data.test_mask].cpu().numpy()

# Standard metrics
print("\n" + "=" * 50)
print("  TRACENET GNN — TEST SET RESULTS")
print("=" * 50)
print(classification_report(test_true, test_pred,
      target_names=['Licit', 'Illicit'], digits=4))

# Confusion matrix
cm = confusion_matrix(test_true, test_pred)
print("Confusion Matrix:")
print(f"                 Predicted Licit  Predicted Illicit")
print(f"Actual Licit         {cm[0][0]:>6}              {cm[0][1]:>6}")
print(f"Actual Illicit       {cm[1][0]:>6}              {cm[1][1]:>6}")
print("=" * 50)
tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives  (Licit correctly cleared):  {tn}")
print(f"False Positives (Licit wrongly blocked):     {fp}")
print(f"False Negatives (Illicit missed):            {fn}")
print(f"True Positives  (Illicit correctly caught):  {tp}")

# ── THREE ZONE BREAKDOWN ──────────────────────────────────────
print("\n" + "=" * 50)
print("  THREE-ZONE CONFIDENCE BREAKDOWN")
print("=" * 50)

import numpy as np

low_mask    = test_probs < 0.40
medium_mask = (test_probs >= 0.40) & (test_probs < 0.75)
high_mask   = test_probs >= 0.75

# For each zone — how many actual criminals vs legitimate users
zones = {
    "AUTO APPROVE  (0-40%)":   low_mask,
    "HUMAN REVIEW  (40-75%)":  medium_mask,
    "AUTO BLOCK    (75-100%)": high_mask,
}

for zone_name, mask in zones.items():
    total     = mask.sum()
    criminals = ((test_true == 1) & mask).sum()
    legit     = ((test_true == 0) & mask).sum()
    if total > 0:
        print(f"\n{zone_name}")
        print(f"  Total transactions : {total}")
        print(f"  Actual criminals   : {criminals} ({criminals/total*100:.1f}%)")
        print(f"  Actual legitimate  : {legit} ({legit/total*100:.1f}%)")

print("\n" + "=" * 50)
print("  THRESHOLD ANALYSIS")
print("=" * 50)
print(f"  Transactions auto-approved : {low_mask.sum()} (model very confident = clean)")
print(f"  Sent for human review      : {medium_mask.sum()} (borderline cases)")
print(f"  Auto-blocked               : {high_mask.sum()} (model very confident = criminal)")
print(f"\n  Of auto-blocked transactions:")
print(f"  Actual criminals caught    : {((test_true==1) & high_mask).sum()}")
print(f"  Legitimate wrongly blocked : {((test_true==0) & high_mask).sum()}")

# ── STEP 8: Save the Model ────────────────────────────────────
import os
import json

os.makedirs('models', exist_ok=True)

# Save model weights
torch.save(model.state_dict(), 'models/gnn_model.pth')

# Save metrics and config
config = {
    "num_features": 166,
    "hidden_1": 128,
    "hidden_2": 64,
    "num_nodes": len(df),
    "num_edges": len(src_nodes),
    "num_illicit": int(num_illicit),
    "num_licit": int(num_licit),
    "accuracy": round(float((test_pred == test_true).mean()) * 100, 2),
    "precision_illicit": round(float(cm[1][1] / (cm[1][1] + cm[0][1])) * 100, 2),
    "recall_illicit": round(float(cm[1][1] / (cm[1][1] + cm[1][0])) * 100, 2),
    "f1_illicit": round(float(2 * cm[1][1] / (2 * cm[1][1] + cm[0][1] + cm[1][0])) * 100, 2),
    "true_positives": int(tp),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_negatives": int(tn),
    "dataset": "Elliptic Bitcoin Dataset",
    "model": "GraphSAGE 3-layer"
}

with open('models/model_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\n✅ Model saved to models/gnn_model.pth")
print("✅ Config saved to models/model_config.json")
print("\nFinal metrics saved:")
print(f"  Accuracy:  {config['accuracy']}%")
print(f"  Precision: {config['precision_illicit']}%")
print(f"  Recall:    {config['recall_illicit']}%")
print(f"  F1 Score:  {config['f1_illicit']}%")