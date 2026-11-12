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

