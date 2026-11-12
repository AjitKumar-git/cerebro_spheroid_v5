import math
import os
import tempfile
import torch
import pytest

from scripts.track_loss_regression import (
    LossRecord,
    RegressionStatus,
    LossTrajectoryAnalyzer,
    render_ascii_sparkline,
    evaluate_checkpoint_loss,
)
from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5


def test_loss_record_serialization():
    rec = LossRecord(
        step=12300,
        loss=3.25,
        eval_loss=4.82,
        ema_loss=3.28,
        delta=-0.03,
        status=RegressionStatus.SMOOTH_DESCENT,
        timestamp="2026-11-12T01:00:00Z",
        source="colab",
        md5="testmd5",
    )
    d = rec.to_dict()
    assert d["step"] == 12300
    assert d["status"] == "SMOOTH_DESCENT"
    rec2 = LossRecord.from_dict(d)
    assert rec2.step == rec.step
    assert rec2.status == RegressionStatus.SMOOTH_DESCENT
    assert rec2.loss == pytest.approx(3.25)


def test_trajectory_analyzer_smooth_descent():
    records = [
        LossRecord(step=1000 * i, loss=5.0 - 0.1 * i, eval_loss=6.0 - 0.12 * i)
        for i in range(1, 6)
    ]
    analyzer = LossTrajectoryAnalyzer(regression_threshold=0.15)
    analysis = analyzer.analyze(records)
    assert analysis.status == RegressionStatus.SMOOTH_DESCENT
    assert analysis.slope < 0.0
    assert analysis.is_regressing is False


def test_trajectory_analyzer_sudden_regression_spike():
    records = [
        LossRecord(step=1000, loss=3.20, eval_loss=4.50),
        LossRecord(step=2000, loss=3.10, eval_loss=4.40),
        LossRecord(step=3000, loss=3.45, eval_loss=4.85),  # +0.35 / +0.45 jump
    ]
    analyzer = LossTrajectoryAnalyzer(regression_threshold=0.15)
    analysis = analyzer.analyze(records)
    assert analysis.status == RegressionStatus.REGRESSION_SPIKE
    assert analysis.is_regressing is True
    assert analysis.last_delta > 0.15


def test_trajectory_analyzer_persistent_regression():
    records = [
        LossRecord(step=1000, loss=3.00, eval_loss=4.00),
        LossRecord(step=2000, loss=3.10, eval_loss=4.10),
        LossRecord(step=3000, loss=3.20, eval_loss=4.20),
        LossRecord(step=4000, loss=3.30, eval_loss=4.30),  # 3 consecutive rises
    ]
    analyzer = LossTrajectoryAnalyzer(regression_threshold=0.15)
    analysis = analyzer.analyze(records)
    assert analysis.status == RegressionStatus.PERSISTENT_REGRESSION
    assert analysis.consecutive_regressions >= 3
    assert analysis.is_regressing is True


def test_trajectory_analyzer_plateau():
    records = [
        LossRecord(step=1000, loss=3.200, eval_loss=4.500),
        LossRecord(step=2000, loss=3.201, eval_loss=4.502),
        LossRecord(step=3000, loss=3.199, eval_loss=4.501),
        LossRecord(step=4000, loss=3.200, eval_loss=4.500),
    ]
    analyzer = LossTrajectoryAnalyzer(regression_threshold=0.15, plateau_threshold=0.005)
    analysis = analyzer.analyze(records)
    assert analysis.status == RegressionStatus.PLATEAU


def test_render_ascii_sparkline():
    vals = [5.0, 4.5, 4.0, 3.5, 3.0, 3.2, 2.9]
    spark = render_ascii_sparkline(vals, width=7)
    assert len(spark) == 7
    # Values decreasing should show decreasing block heights
    assert spark[0] != spark[-1]


def test_evaluate_checkpoint_loss_fast():
    cfg = ModelConfigV5(**PRESETS_V5["v5_test"])
    model = CerebroSpheroidV5(cfg)
    device = torch.device("cpu")

    eval_batches = [
        (
            torch.randint(1, cfg.vocab_size, (2, 8)),
            torch.randint(1, cfg.vocab_size, (2, 16)),
            torch.randint(1, cfg.vocab_size, (2, 16)),
        )
    ]

    metrics = evaluate_checkpoint_loss(
        model=model,
        eval_batches=eval_batches,
        pad_id=0,
        device=device,
        use_amp=False,
    )
    assert "val_loss" in metrics
    assert "perplexity" in metrics
    assert math.isfinite(metrics["val_loss"]) and metrics["val_loss"] > 0.0


def test_colab_loss_tracker_notebook_valid():
    """Verifies that colab/CerebroSpheroid_V5_Loss_Tracker.ipynb is valid JSON and syntactically sound."""
    import json
    nb_path = os.path.join(os.path.dirname(__file__), "..", "colab", "CerebroSpheroid_V5_Loss_Tracker.ipynb")
    assert os.path.exists(nb_path)

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb.get("nbformat") == 4
    assert len(nb.get("cells", [])) >= 6

    # Verify code cells compile cleanly
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            clean_src = "\n".join("# " + l if l.strip().startswith(("%", "!")) else l for l in src.splitlines())
            compile(clean_src, nb_path, "exec")


def test_cloud_sync_loss_history_roundtrip(monkeypatch):
    """Verifies push_remote_loss_history and fetch_remote_loss_history serialization and roundtrip."""
    from unittest.mock import MagicMock
    import json
    import urllib.request
    from src.cloud_sync import push_remote_loss_history, fetch_remote_loss_history

    records = [
        LossRecord(step=1000, loss=3.5, eval_loss=4.2, status=RegressionStatus.SMOOTH_DESCENT),
        LossRecord(step=2000, loss=3.2, eval_loss=3.9, status=RegressionStatus.SMOOTH_DESCENT),
    ]

    # Test push
    push_called = {}
    def mock_urlopen(req, timeout=15):
        m = MagicMock()
        if "action=save_loss_history" in req.full_url:
            push_called["payload"] = json.loads(req.data.decode("utf-8"))
            m.read.return_value = json.dumps({"success": True, "status": "saved", "total_records": 2}).encode("utf-8")
        elif "action=get_loss_history" in req.full_url:
            m.read.return_value = json.dumps({
                "status": "found",
                "model": "v5_small",
                "count": 2,
                "records": [r.to_dict() for r in records],
            }).encode("utf-8")
        m.__enter__.return_value = m
        return m

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    ok = push_remote_loss_history(model="v5_small", records=records)
    assert ok is True
    assert "payload" in push_called
    assert push_called["payload"]["model"] == "v5_small"
    assert len(push_called["payload"]["records"]) == 2
    assert push_called["payload"]["records"][0]["step"] == 1000

    # Test fetch
    fetched = fetch_remote_loss_history(model="v5_small")
    assert fetched is not None
    assert len(fetched) == 2
    assert fetched[1]["step"] == 2000
    assert fetched[1]["loss"] == pytest.approx(3.2)


def test_run_loss_tracker_syncs_with_website_data(tmp_path, monkeypatch):
    """Verifies run_loss_tracker fetches remote history, merges it, and pushes new steps to website data."""
    from unittest.mock import MagicMock
    from scripts.track_loss_regression import run_loss_tracker
    import src.cloud_sync as cs

    remote_store = [
        {"step": 1000, "loss": 4.0, "status": "SMOOTH_DESCENT", "timestamp": "2026-11-12T01:00:00Z"},
        {"step": 2000, "loss": 3.7, "status": "SMOOTH_DESCENT", "timestamp": "2026-11-12T02:00:00Z"},
    ]
    pushed_records = []

    def mock_fetch(model, url=None, token=None):
        return list(remote_store)

    def mock_push(model, records, url=None, token=None):
        pushed_records.clear()
        pushed_records.extend(records)
        return True

    def mock_manifest(model=None, url=None, token=None):
        return {
            "models": {
                "v5_small": {
                    "latest_manifest": {
                        "model": "v5_small",
                        "step": 3000,
                        "loss": 3.4,
                        "source": "tpu",
                        "updated_at": "2026-11-12T03:00:00Z",
                        "md5": "md5step3000",
                    }
                }
            }
        }

    monkeypatch.setattr(cs, "fetch_remote_loss_history", mock_fetch)
    monkeypatch.setattr(cs, "push_remote_loss_history", mock_push)
    monkeypatch.setattr(cs, "check_remote_manifest", mock_manifest)

    hist_file = str(tmp_path / "loss_hist_test.json")
    records, analysis = run_loss_tracker(
        model="v5_small",
        watch=False,
        history_file=hist_file,
        save_to_cloud=True,
    )

    assert len(records) == 3
    assert records[0].step == 1000
    assert records[1].step == 2000
    assert records[2].step == 3000
    assert records[2].loss == pytest.approx(3.4)
    assert len(pushed_records) == 3
    assert pushed_records[-1].step == 3000
    assert os.path.exists(hist_file)

