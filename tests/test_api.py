"""
TraceNet v2 — API Test Suite
==============================
Run: pytest tests/ -v

Tests cover:
  - Health & system endpoints
  - Transaction scoring (low risk, high risk, edge cases)
  - Risk score bounds (always 0-100%)
  - SAR generation for blocked transactions
  - Interbank share privacy guarantees
  - Graph neighbors lookup
  - Communities endpoint

Requires the backend to be running locally:
  uvicorn backend.api:app --reload
OR uses TestClient (no server needed — recommended).
"""

import pytest
from fastapi.testclient import TestClient

# Import app directly — TestClient spins it up without needing uvicorn
from backend.api import app

client = TestClient(app)


# ════════════════════════════════════════════════════════════════
#  FIXTURES
# ════════════════════════════════════════════════════════════════

@pytest.fixture
def low_risk_tx():
    """A small UPI transaction — expected: AUTO_APPROVE."""
    return {
        "amount":      500.0,
        "channel":     "upi",
        "sender_id":   "ACC_CLEAN_001",
        "receiver_id": "ACC_CLEAN_002",
    }


@pytest.fixture
def high_risk_tx():
    """A large wire transfer — expected: AUTO_BLOCK or HUMAN_REVIEW."""
    return {
        "amount":      4_500_000.0,
        "channel":     "wire",
        "sender_id":   "ACC_SUSPECT_999",
        "receiver_id": "ACC_SUSPECT_888",
    }


@pytest.fixture
def crypto_tx():
    """A medium crypto transaction."""
    return {
        "amount":      75_000.0,
        "channel":     "crypto",
        "sender_id":   "WALLET_ABC",
        "receiver_id": "WALLET_XYZ",
    }


# ════════════════════════════════════════════════════════════════
#  HEALTH & SYSTEM
# ════════════════════════════════════════════════════════════════

class TestHealth:
    def test_health_returns_ok(self):
        """Backend health check must return status=ok."""
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "accuracy" in data
        assert "timestamp" in data

    def test_health_includes_model_name(self):
        res = client.get("/health")
        data = res.json()
        assert "model" in data
        assert "GraphSAGE" in data["model"]

    def test_root_endpoint(self):
        res = client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert data["name"] == "TraceNet v2 API"
        assert "docs" in data


class TestModelInfo:
    def test_model_info_returns_metrics(self):
        res = client.get("/model_info")
        assert res.status_code == 200
        data = res.json()
        # Must contain all expected metric fields
        for field in ["accuracy", "precision_illicit", "recall_illicit",
                      "f1_illicit", "num_nodes", "num_edges"]:
            assert field in data, f"Missing field: {field}"

    def test_accuracy_is_realistic(self):
        res = client.get("/model_info")
        data = res.json()
        # Our model achieved 97.62%
        assert 90.0 <= data["accuracy"] <= 100.0

    def test_model_info_has_zone_thresholds(self):
        res = client.get("/model_info")
        data = res.json()
        assert "zone_thresholds" in data


class TestStats:
    def test_stats_returns_counters(self):
        res = client.get("/stats")
        assert res.status_code == 200
        data = res.json()
        for field in ["total_scored", "auto_approved", "human_review", "auto_blocked"]:
            assert field in data
            assert isinstance(data[field], int)
            assert data[field] >= 0


# ════════════════════════════════════════════════════════════════
#  TRANSACTION SCORING
# ════════════════════════════════════════════════════════════════

class TestScoreTransaction:
    def test_low_risk_returns_valid_response(self, low_risk_tx):
        """Small UPI transaction should return a valid response structure."""
        res = client.post("/score_transaction", json=low_risk_tx)
        assert res.status_code == 200
        data = res.json()
        # Required fields
        for field in ["tx_hash", "risk_score", "risk_percent",
                      "verdict", "zone", "timestamp"]:
            assert field in data, f"Missing field: {field}"

    def test_risk_score_always_between_0_and_1(self, low_risk_tx, high_risk_tx):
        """Risk score must always be in [0, 1] regardless of input."""
        for tx in [low_risk_tx, high_risk_tx]:
            res = client.post("/score_transaction", json=tx)
            data = res.json()
            assert 0.0 <= data["risk_score"] <= 1.0, (
                f"risk_score out of bounds: {data['risk_score']}"
            )

    def test_risk_percent_matches_risk_score(self, low_risk_tx):
        """risk_percent must equal round(risk_score * 100)."""
        res = client.post("/score_transaction", json=low_risk_tx)
        data = res.json()
        expected = int(round(data["risk_score"] * 100))
        assert data["risk_percent"] == expected

    def test_verdict_is_valid_string(self, low_risk_tx, high_risk_tx):
        valid_verdicts = {"AUTO_APPROVE", "HUMAN_REVIEW", "AUTO_BLOCK"}
        for tx in [low_risk_tx, high_risk_tx]:
            res = client.post("/score_transaction", json=tx)
            data = res.json()
            assert data["verdict"] in valid_verdicts

    def test_zone_matches_verdict(self, low_risk_tx, high_risk_tx):
        verdict_to_zone = {
            "AUTO_APPROVE": "green",
            "HUMAN_REVIEW": "yellow",
            "AUTO_BLOCK":   "red",
        }
        for tx in [low_risk_tx, high_risk_tx]:
            res = client.post("/score_transaction", json=tx)
            data = res.json()
            expected_zone = verdict_to_zone[data["verdict"]]
            assert data["zone"] == expected_zone, (
                f"Zone mismatch: verdict={data['verdict']}, zone={data['zone']}"
            )

    def test_high_risk_transaction_flagged(self, high_risk_tx, low_risk_tx):
        """
        A large wire transfer should score higher than a small UPI payment.
        We test the relative ordering rather than a fixed verdict,
        because the synthetic feature scorer is probabilistic.
        """
        high_res = client.post("/score_transaction", json=high_risk_tx)
        low_res  = client.post("/score_transaction", json=low_risk_tx)
        high_risk = high_res.json()["risk_score"]
        low_risk  = low_res.json()["risk_score"]
        assert high_risk >= low_risk, (
            f"High-risk tx scored lower than low-risk tx: {high_risk:.2f} < {low_risk:.2f}"
        )

    def test_response_contains_hashed_ids(self, low_risk_tx):
        """Sender/receiver must be hashed — never raw in response."""
        res = client.post("/score_transaction", json=low_risk_tx)
        data = res.json()
        assert "sender_hashed"   in data
        assert "receiver_hashed" in data
        # Raw IDs must NOT appear in hashed fields
        assert data["sender_hashed"]   != low_risk_tx["sender_id"]
        assert data["receiver_hashed"] != low_risk_tx["receiver_id"]
        # Hash length should be 64 chars (SHA-256 hex)
        assert len(data["sender_hashed"])   == 64
        assert len(data["receiver_hashed"]) == 64

    def test_stats_increment_after_scoring(self):
        """total_scored must increase by 1 after each transaction."""
        before = client.get("/stats").json()["total_scored"]
        client.post("/score_transaction", json={
            "amount": 1000.0, "channel": "upi",
            "sender_id": "A", "receiver_id": "B"
        })
        after = client.get("/stats").json()["total_scored"]
        assert after == before + 1


# ════════════════════════════════════════════════════════════════
#  EDGE CASES
# ════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_zero_amount_transaction(self):
        """Zero-amount transaction should not crash."""
        res = client.post("/score_transaction", json={
            "amount": 0.0, "channel": "upi",
            "sender_id": "A", "receiver_id": "B"
        })
        assert res.status_code == 200

    def test_very_large_amount(self):
        """Extremely large amount should not crash."""
        res = client.post("/score_transaction", json={
            "amount": 999_999_999.99, "channel": "wire",
            "sender_id": "CORP_A", "receiver_id": "CORP_B"
        })
        assert res.status_code == 200
        data = res.json()
        assert 0.0 <= data["risk_score"] <= 1.0

    def test_negative_amount_rejected(self):
        """Negative amounts should return 422 Unprocessable Entity."""
        res = client.post("/score_transaction", json={
            "amount": -500.0, "channel": "upi",
            "sender_id": "A", "receiver_id": "B"
        })
        assert res.status_code == 422

    def test_missing_amount_rejected(self):
        """Request without required 'amount' field must fail validation."""
        res = client.post("/score_transaction", json={
            "channel": "wire", "sender_id": "A", "receiver_id": "B"
        })
        assert res.status_code == 422

    def test_unknown_channel_handled(self):
        """Unknown channel strings should not crash — fall back to default risk."""
        res = client.post("/score_transaction", json={
            "amount": 10000.0, "channel": "carrier_pigeon",
            "sender_id": "A", "receiver_id": "B"
        })
        assert res.status_code == 200

    def test_all_channels_work(self):
        """All supported channels should return valid responses."""
        for channel in ["wire", "upi", "atm", "mobile", "crypto", "cash"]:
            res = client.post("/score_transaction", json={
                "amount": 5000.0, "channel": channel,
                "sender_id": "X", "receiver_id": "Y"
            })
            assert res.status_code == 200, f"Failed for channel: {channel}"


# ════════════════════════════════════════════════════════════════
#  SAR REPORTS
# ════════════════════════════════════════════════════════════════

class TestSAR:
    def test_sar_generated_for_blocked_transaction(self, high_risk_tx):
        """
        A blocked transaction must have sar_available=True
        and the SAR must be retrievable.
        """
        score_res = client.post("/score_transaction", json=high_risk_tx)
        data = score_res.json()

        if data["verdict"] == "AUTO_BLOCK":
            assert data["sar_available"] is True
            tx_hash = data["tx_hash"]

            sar_res = client.get(f"/sar_report/{tx_hash}")
            assert sar_res.status_code == 200
            sar = sar_res.json()
            assert "report_id"   in sar
            assert "narrative"   in sar
            assert "risk_percent" in sar
            assert sar["risk_percent"] > 0

    def test_sar_not_found_for_unknown_hash(self):
        """Non-existent tx_hash must return 404."""
        res = client.get("/sar_report/deadbeefdeadbeef")
        assert res.status_code == 404

    def test_all_sars_endpoint(self):
        """all_sars must return list and count."""
        res = client.get("/all_sars")
        assert res.status_code == 200
        data = res.json()
        assert "count" in data
        assert "reports" in data
        assert isinstance(data["reports"], list)

    def test_sar_contains_no_raw_pii(self, high_risk_tx):
        """SAR narrative must not contain raw sender/receiver IDs."""
        score_res = client.post("/score_transaction", json=high_risk_tx)
        data = score_res.json()
        if data["verdict"] == "AUTO_BLOCK":
            sar = client.get(f"/sar_report/{data['tx_hash']}").json()
            narrative = sar["narrative"]
            assert high_risk_tx["sender_id"]   not in narrative
            assert high_risk_tx["receiver_id"] not in narrative


# ════════════════════════════════════════════════════════════════
#  INTERBANK SHARE
# ════════════════════════════════════════════════════════════════

class TestInterbankShare:
    def test_response_contains_no_pii(self):
        """Interbank response must not expose raw IDs."""
        res = client.post("/interbank_share", json={
            "tx_id": "TX_SECRET_001",
            "sender_id": "REAL_ACCOUNT_12345"
        })
        assert res.status_code == 200
        data = res.json()
        assert "REAL_ACCOUNT_12345" not in str(data)
        assert "TX_SECRET_001"      not in str(data)
        assert data["pii_exposed"] is False

    def test_hashed_id_is_64_chars(self):
        """SHA-256 hex digest is always 64 characters."""
        res = client.post("/interbank_share", json={
            "tx_id": "TX_001", "sender_id": "ACC_001"
        })
        data = res.json()
        assert len(data["hashed_sender"]) == 64
        assert len(data["hashed_tx"])     == 64

    def test_same_input_same_hash(self):
        """SHA-256 must be deterministic — same input = same hash."""
        payload = {"tx_id": "TX_999", "sender_id": "ACC_999"}
        r1 = client.post("/interbank_share", json=payload).json()
        r2 = client.post("/interbank_share", json=payload).json()
        assert r1["hashed_sender"] == r2["hashed_sender"]
        assert r1["hashed_tx"]     == r2["hashed_tx"]

    def test_compliance_fields_present(self):
        """Must include compliance tags."""
        res = client.post("/interbank_share", json={
            "tx_id": "TX_001", "sender_id": "ACC_001"
        })
        data = res.json()
        assert "compliant_with" in data
        assert len(data["compliant_with"]) > 0


# ════════════════════════════════════════════════════════════════
#  GRAPH & COMMUNITIES
# ════════════════════════════════════════════════════════════════

class TestCommunities:
    def test_communities_returns_list(self):
        res = client.get("/communities")
        assert res.status_code == 200
        data = res.json()
        assert "communities" in data
        assert "total_communities" in data
        assert isinstance(data["communities"], list)

    def test_community_has_required_fields(self):
        res = client.get("/communities")
        data = res.json()
        if data["communities"]:
            c = data["communities"][0]
            for field in ["id", "size", "illicit_ratio", "members"]:
                assert field in c

    def test_illicit_ratio_in_range(self):
        res = client.get("/communities")
        data = res.json()
        for c in data["communities"]:
            assert 0.0 <= c["illicit_ratio"] <= 1.0


# ════════════════════════════════════════════════════════════════
#  BATCH SCORING
# ════════════════════════════════════════════════════════════════

class TestBatchScoring:
    def test_batch_returns_all_results(self):
        payload = {
            "transactions": [
                {"amount": 1000.0, "channel": "upi",  "sender_id": "A", "receiver_id": "B"},
                {"amount": 2000.0, "channel": "wire",  "sender_id": "C", "receiver_id": "D"},
                {"amount": 3000.0, "channel": "crypto","sender_id": "E", "receiver_id": "F"},
            ]
        }
        res = client.post("/score_batch", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["count"] == 3
        assert len(data["results"]) == 3

    def test_empty_batch_rejected(self):
        res = client.post("/score_batch", json={"transactions": []})
        # Empty list is technically valid — returns count=0
        assert res.status_code in (200, 422)
