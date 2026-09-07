"""
TraceNet v2 — Model Loader
===========================
Loads the trained GraphSAGE model once at startup.
Exposes predict_full_graph() and predict_single() for the API.

Architecture mirrors trainmodel.py exactly:
  SAGEConv 166→128 + BatchNorm + ReLU + Dropout(0.3)
  SAGEConv 128→64  + BatchNorm + ReLU + Dropout(0.3)
  SAGEConv 64→2    + log_softmax
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import json
import os
import logging

logger = logging.getLogger(__name__)


# ── GNN Architecture (must match trainmodel.py exactly) ──────────
class TraceNetGNN(torch.nn.Module):
    """3-layer GraphSAGE — 166 → 128 → 64 → 2."""

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
    """

    def __init__(
        self,
        model_path: str = "models/gnn_model.pth",
        config_path: str = "models/model_config.json",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path) as f:
            self.config = json.load(f)
        logger.info(f"Model config loaded: accuracy={self.config.get('accuracy')}%")

        # Build model and load weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        self.model = TraceNetGNN().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")

    # ── Inference methods ─────────────────────────────────────────

    def predict_full_graph(
        self, features: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Run inference on the full graph.
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

        If edge_index is None, uses a self-loop so the node can still
        pass through BatchNorm without crashing on batch-size-1.
        """
        if edge_index is None:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        features   = node_features.to(self.device)
        edge_index = edge_index.to(self.device)

        # BatchNorm needs > 1 sample in training mode — eval mode is fine
        with torch.no_grad():
            log_probs = self.model(features, edge_index)
            probs = torch.exp(log_probs)

        # If multiple nodes, return probs for node 0 (the target)
        licit_prob   = float(probs[0, 0])
        illicit_prob = float(probs[0, 1])
        return licit_prob, illicit_prob
