"""
TraceNet v2 — Graph Store
==========================
Loads the preprocessed graph data (models/processed_data.npz) once at startup.

Provides:
  - Full feature matrix + edge index for whole-graph inference
  - Per-node lookups: tx_id → features, label, neighbors
  - Synthetic feature generation for new transactions
  - Community detection on illicit subgraph
"""

import numpy as np
import torch
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class GraphStore:
    """In-memory graph and feature store, loaded once at startup."""

    def __init__(self, data_path: str = "models/processed_data.npz"):
        logger.info(f"Loading graph data from {data_path}...")
        data = np.load(data_path, allow_pickle=True)

        # Core data
        self.features     = torch.tensor(data["features"],     dtype=torch.float)
        self.labels       = data["labels"].astype(np.int32)
        self.tx_ids       = data["tx_ids"].astype(np.int64)
        self.edge_src     = data["edge_src"].astype(np.int64)
        self.edge_dst     = data["edge_dst"].astype(np.int64)

        # Feature distribution stats (for synthetic scoring)
        self.mean_licit   = data["mean_licit"].astype(np.float32)
        self.mean_illicit = data["mean_illicit"].astype(np.float32)
        self.std_licit    = data["std_licit"].astype(np.float32)

        # Build lookups
        self.node_to_idx: dict[int, int] = {
            int(tx): idx for idx, tx in enumerate(self.tx_ids)
        }

        # Adjacency list — bidirectional
        self.adj: dict[int, list[int]] = defaultdict(list)
        for s, d in zip(self.edge_src, self.edge_dst):
            self.adj[int(s)].append(int(d))
            self.adj[int(d)].append(int(s))

        # Pre-build full graph edge_index tensor (np.stack avoids slow list-of-arrays path)
        self._full_edge_index = torch.from_numpy(
            np.stack([self.edge_src, self.edge_dst], axis=0).astype(np.int64)
        )

        logger.info(
            f"Graph loaded: {len(self.tx_ids):,} nodes, "
            f"{len(self.edge_src):,} edges"
        )

    # ── Full graph ────────────────────────────────────────────────

    def get_full_graph(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (features, edge_index) for whole-graph inference."""
        return self.features, self._full_edge_index

    # ── Node lookups ──────────────────────────────────────────────

    def get_node_idx(self, tx_id: int | str) -> int | None:
        return self.node_to_idx.get(int(tx_id), None)

    def get_node_features(self, node_idx: int) -> torch.Tensor:
        """Return (1, 166) feature tensor for a single node."""
        return self.features[node_idx].unsqueeze(0)

    def get_node_label(self, node_idx: int) -> str:
        label = int(self.labels[node_idx])
        return {0: "licit", 1: "illicit"}.get(label, "unknown")

    # ── Neighbourhood ─────────────────────────────────────────────

    def get_k_hop_neighbors(self, node_idx: int, hops: int = 2) -> list[int]:
        """BFS up to `hops` hops. Returns list of node indices including center."""
        visited = {node_idx}
        frontier = {node_idx}
        for _ in range(hops):
            nxt = set()
            for n in frontier:
                nxt.update(self.adj.get(n, []))
            frontier = nxt - visited
            visited.update(frontier)
        return list(visited)

    def get_subgraph(
        self, node_idx: int, hops: int = 2
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """
        Return (features, edge_index, original_indices) for the k-hop subgraph.
        edge_index is re-indexed to 0..len(subgraph_nodes)-1.
        """
        sub_nodes = self.get_k_hop_neighbors(node_idx, hops)
        sub_set   = set(sub_nodes)
        local_map = {orig: local for local, orig in enumerate(sub_nodes)}

        sub_feat = self.features[sub_nodes]

        src_l, dst_l = [], []
        for s, d in zip(self.edge_src, self.edge_dst):
            s, d = int(s), int(d)
            if s in sub_set and d in sub_set:
                src_l.append(local_map[s])
                dst_l.append(local_map[d])

        if not src_l:
            # Isolated node — use self-loop
            src_l, dst_l = [0], [0]

        sub_edge_index = torch.tensor([src_l, dst_l], dtype=torch.long)
        return sub_feat, sub_edge_index, sub_nodes

    def get_neighbor_info(self, node_idx: int, hops: int = 2) -> dict:
        """
        Return JSON-serialisable neighbor data for the Graph Forensics endpoint.
        """
        sub_nodes = self.get_k_hop_neighbors(node_idx, hops)
        nodes_out = []
        for idx in sub_nodes:
            nodes_out.append({
                "id":    str(self.tx_ids[idx]),
                "label": self.get_node_label(idx),
                "risk":  None,  # filled in by API after GNN inference
            })

        sub_set   = set(sub_nodes)
        local_map = {orig: local for local, orig in enumerate(sub_nodes)}
        edges_out = []
        for s, d in zip(self.edge_src, self.edge_dst):
            s, d = int(s), int(d)
            if s in sub_set and d in sub_set:
                edges_out.append({
                    "source": str(self.tx_ids[s]),
                    "target": str(self.tx_ids[d]),
                })

        return {
            "center":      str(self.tx_ids[node_idx]),
            "nodes":       nodes_out,
            "edges":       edges_out,
            "node_count":  len(nodes_out),
            "edge_count":  len(edges_out),
        }

    # ── Synthetic feature generation ──────────────────────────────

    def make_synthetic_features(self, base_risk: float) -> torch.Tensor:
        """
        Build a (1, 166) feature tensor for a new (unseen) transaction.

        base_risk in [0, 1]:
          0.0 → features near the licit mean
          1.0 → features near the illicit mean
        Linearly interpolates + adds small random noise for realism.
        """
        base_risk  = float(np.clip(base_risk, 0.0, 1.0))
        vec = (
            (1.0 - base_risk) * self.mean_licit
            + base_risk       * self.mean_illicit
            + np.random.normal(0, 0.02 * self.std_licit, size=self.mean_licit.shape)
        ).astype(np.float32)
        return torch.tensor(vec, dtype=torch.float).unsqueeze(0)

    # ── Community detection ───────────────────────────────────────

    def detect_illicit_communities(
        self, min_size: int = 3, max_communities: int = 10
    ) -> list[dict]:
        """
        Connected components in the illicit-heavy subgraph.
        Returns list of community dicts sorted by illicit_ratio desc.
        """
        illicit_set = set(
            int(i) for i, lbl in enumerate(self.labels) if lbl == 1
        )

        # BFS on adjacency restricted to nodes with illicit neighbors
        visited = set()
        communities = []

        for start in illicit_set:
            if start in visited:
                continue
            component = set()
            queue = [start]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                for nb in self.adj.get(node, []):
                    if nb not in visited:
                        queue.append(nb)

            if len(component) < min_size:
                continue

            illicit_count = sum(1 for n in component if n in illicit_set)
            illicit_ratio = illicit_count / len(component)

            communities.append({
                "id":           len(communities) + 1,
                "size":         len(component),
                "illicit_count":illicit_count,
                "illicit_ratio":round(illicit_ratio, 3),
                "members":      [str(self.tx_ids[n]) for n in list(component)[:10]],
            })

        communities.sort(key=lambda c: c["illicit_ratio"], reverse=True)
        return communities[:max_communities]
