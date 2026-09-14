"""
TraceNet v2 — Model Loader  v2.1
==================================
Loads the trained GraphSAGE model once at startup.
Exposes inference and explainability methods.

Architecture mirrors trainmodel.py exactly:
  SAGEConv 166->128 + BatchNorm + ReLU + Dropout(0.3)
  SAGEConv 128->64  + BatchNorm + ReLU + Dropout(0.3)
  SAGEConv 64->2    + log_softmax

New in v2.1:
  - explain_prediction(): gradient x input attribution per feature group
  - Feature group mapping (Elliptic dataset):
      f0        = time step
      f1-f93    = local transaction features  (93 features)
      f94-f165  = neighbourhood aggregates    (72 features)
"""

import json
import logging
import os

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

logger = logging.getLogger(__name__)

# ── Feature group definitions (Elliptic dataset) ─────────────────
# The 166 features break into 3 groups based on the original paper.
# Group names match what is shown to users in risk_factors and /explain.
FEATURE_GROUPS = {
    "time_step":         (0,  1),    # 1 feature  — which of 49 time steps
    "local_features":    (1,  94),   # 93 features — tx-level (volume, fees, etc.)
    "network_features":  (94, 166),  # 72 features — 1-hop and 2-hop neighbourhood stats
}


# ── GNN Architecture (must match trainmodel.py exactly) ──────────
class TraceNetGNN(torch.nn.Module):
    """3-layer GraphSAGE — 166 -> 128 -> 64 -> 2."""

    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(166, 128)
        self.conv2 = SAGEConv(128, 64)
        self.conv3 = SAGEConv(64, 2)
        self.dropout = torch.nn.Dropout(0.3)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)

    def forward(self, x, edge_index):
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
        # Output layer
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


# ── Model Loader ─────────────────────────────────────────────────
class ModelLoader:
    """
    Singleton-style loader. Instantiate once at app startup via lifespan.
    Thread-safe for read-only inference.
    """

    def __init__(
        self,
        model_path:  str = "models/gnn_model.pth",
        config_path: str = "models/model_config.json",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # ── Config ───────────────────────────────────────────────
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path) as f:
            self.config = json.load(f)
        logger.info(f"Model config loaded: accuracy={self.config.get('accuracy')}%")

        # ── Weights ───────────────────────────────────────────────
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        self.model = TraceNetGNN().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")

    # ── Core inference ────────────────────────────────────────────

    def predict_full_graph(
        self, features: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Run inference on the full graph or subgraph.
        Returns: (N, 2) probability tensor — [:, 0] = licit, [:, 1] = illicit.
        """
        features   = features.to(self.device)
        edge_index = edge_index.to(self.device)
        with torch.no_grad():
            log_probs = self.model(features, edge_index)
            probs = torch.exp(log_probs)
        return probs.cpu()

    def predict_single(
        self, node_features: torch.Tensor, edge_index: torch.Tensor | None = None
    ) -> tuple[float, float]:
        """
        Score a single node (possibly with mini-subgraph context).
        Returns (licit_prob, illicit_prob) as plain floats.

        If edge_index is None, uses a self-loop so BatchNorm works on batch-size-1.
        """
        if edge_index is None:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        features   = node_features.to(self.device)
        edge_index = edge_index.to(self.device)

        with torch.no_grad():
            log_probs = self.model(features, edge_index)
            probs = torch.exp(log_probs)

        licit_prob   = float(probs[0, 0])
        illicit_prob = float(probs[0, 1])
        return licit_prob, illicit_prob

    # ── Explainability ────────────────────────────────────────────

    def explain_prediction(
        self,
        features:   torch.Tensor,
        edge_index: torch.Tensor,
        node_idx:   int = 0,
    ) -> dict:
        """
        Gradient x Input attribution for a target node in a subgraph.

        Method: Gradient x Input (Simonyan et al.)
          importance_i = gradient(output_illicit / input_i) * input_i

        Positive importance → feature pushes toward ILLICIT classification.
        Negative importance → feature pushes toward LICIT classification.

        Returns a dict with:
          - feature_group_importance: how much each group drove the prediction
          - dominant_group: which group was most influential
          - illicit_drive_ratio: 0-1, how much of attribution pushed toward illicit
          - top_illicit_features: top 5 feature indices pushing toward illicit
          - top_licit_features:   top 5 feature indices pushing toward licit
        """
        # Need gradients — temporarily put model in eval with grad enabled
        self.model.eval()

        feat = features.clone().float().to(self.device)
        feat.requires_grad_(True)
        edge_idx = edge_index.to(self.device)

        # Forward pass (no torch.no_grad() here — we need the graph)
        log_probs = self.model(feat, edge_idx)

        # Backprop on the illicit log-probability of the target node
        log_probs[node_idx, 1].backward()

        with torch.no_grad():
            grads     = feat.grad          # (N, 166)
            node_grad = grads[node_idx]    # (166,)
            node_feat = feat[node_idx].detach()

            # Gradient x Input — signed attribution per feature
            attr = (node_grad * node_feat).cpu().numpy()

            # Split into illicit-push (+) and licit-push (-) totals
            illicit_push = float(attr.clip(min=0).sum())
            licit_push   = float((-attr).clip(min=0).sum())
            total_abs    = illicit_push + licit_push + 1e-8

            # Feature group breakdown
            group_scores: dict[str, float] = {}
            for group_name, (start, end) in FEATURE_GROUPS.items():
                group_attr = abs(attr[start:end]).sum()
                group_scores[group_name] = float(group_attr)

            group_total = sum(group_scores.values()) + 1e-8
            group_pct   = {
                k: round(v / group_total, 3)
                for k, v in group_scores.items()
            }

            dominant_group = max(group_pct, key=group_pct.get)

            # Top contributing feature indices (1-indexed to match column names)
            sorted_illicit = sorted(
                range(166), key=lambda i: attr[i], reverse=True
            )
            sorted_licit = sorted(
                range(166), key=lambda i: attr[i]
            )

            top_illicit = [
                {"feature": f"f{i+1}", "score": round(float(attr[i]), 4)}
                for i in sorted_illicit[:5]
                if attr[i] > 0
            ]
            top_licit = [
                {"feature": f"f{i+1}", "score": round(float(-attr[i]), 4)}
                for i in sorted_licit[:5]
                if attr[i] < 0
            ]

        return {
            "method":                "gradient_x_input",
            "node_idx":              node_idx,
            "illicit_drive_ratio":   round(illicit_push / total_abs, 3),
            "licit_drive_ratio":     round(licit_push   / total_abs, 3),
            "dominant_group":        dominant_group,
            "feature_group_importance": group_pct,
            "feature_group_descriptions": {
                "time_step":         "Which of 49 time steps this transaction belongs to",
                "local_features":    "Transaction-level features: volume, fee, addresses, input/output counts (93 features)",
                "network_features":  "Aggregated stats from 1-hop and 2-hop neighbourhood (72 features) — the GNN's unique advantage",
            },
            "top_illicit_features":  top_illicit,
            "top_licit_features":    top_licit,
        }
