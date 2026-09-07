"""
conftest.py — pytest configuration for TraceNet v2
====================================================
Runs before all tests. Ensures the model and graph store are loaded
with correct absolute paths regardless of where pytest is invoked from.
"""

import os
import sys
import pytest

# ── Make sure project root is on sys.path ────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Absolute paths to model files ────────────────────────────────
MODEL_PATH  = os.path.join(PROJECT_ROOT, "models", "gnn_model.pth")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "model_config.json")
DATA_PATH   = os.path.join(PROJECT_ROOT, "models", "processed_data.npz")


@pytest.fixture(scope="session", autouse=True)
def load_backend_globals():
    """
    Session-scoped fixture: loads the GNN model and graph store once
    for the entire test session, then injects them into backend.api's
    module-level globals (_loader, _store) before any test runs.

    This replicates what the FastAPI lifespan does at server startup,
    but with explicit absolute paths so pytest works from any directory.
    """
    import backend.api as api_module
    from backend.model_loader import ModelLoader
    from backend.graph_store import GraphStore

    # Verify files exist before trying to load
    for path, label in [(MODEL_PATH, "Model weights"), (CONFIG_PATH, "Model config"), (DATA_PATH, "Processed data")]:
        if not os.path.exists(path):
            pytest.fail(
                f"{label} not found at: {path}\n"
                f"Run: python scripts/preprocess.py  — then try again."
            )

    # Load and inject
    api_module._loader = ModelLoader(model_path=MODEL_PATH, config_path=CONFIG_PATH)
    api_module._store  = GraphStore(data_path=DATA_PATH)

    yield  # tests run here

    # Cleanup (optional — session ends anyway)
    api_module._loader = None
    api_module._store  = None
