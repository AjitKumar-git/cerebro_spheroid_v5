"""
Unit tests for CerebroSpheroid V5 AGI Emergence & Manifold Safety Diagnostic Suite.
"""

import pytest
import torch
import torch.nn as nn

from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, PRESETS_V5
from src.emergence_detector import AGIEmergenceAuditor, EmergenceReport


@pytest.fixture
def test_model():
    """Constructs a small fast V5 model for unit testing."""
    cfg = ModelConfigV5(
        vocab_size=128,
        surface_dim=64,
        radial_dim=32,
        num_shells=8,
        num_shortcut_tracts=8,
        num_heads=2,
        thinking_cycles=2,
        max_thinking_cycles=4,
        use_adaptive_thinking=True,
    )
    model = CerebroSpheroidV5(cfg)
    model.eval()
    return model


def test_auditor_initialization(test_model):
    auditor = AGIEmergenceAuditor(test_model, device=torch.device("cpu"))
    assert auditor.model is test_model
    assert auditor.device == torch.device("cpu")


def test_audit_laminar_hierarchy(test_model):
    auditor = AGIEmergenceAuditor(test_model, device=torch.device("cpu"))
    sample_ids = torch.randint(1, 100, (2, 16))
    res = auditor.audit_laminar_hierarchy(sample_ids)

    assert "shell_rms_energy" in res
    assert len(res["shell_rms_energy"]) == 8
    assert "effective_dimensional_rank" in res
    assert res["effective_dimensional_rank"] > 0.0
    assert "inversion_sensitivity_delta" in res
    assert res["inversion_sensitivity_delta"] >= 0.0
    assert "laminar_protobrain_score" in res
    assert 0.0 <= res["laminar_protobrain_score"] <= 100.0


def test_audit_connectome_topology(test_model):
    auditor = AGIEmergenceAuditor(test_model, device=torch.device("cpu"))
    res = auditor.audit_connectome_topology()

    assert "num_tracts" in res
    assert res["num_tracts"] == 8
    assert "mean_tract_norm" in res
    assert res["mean_tract_norm"] > 0.0
    assert "top_tracts" in res
    assert len(res["top_tracts"]) > 0
    assert "macro_circuit_emergence_score" in res
    assert 0.0 <= res["macro_circuit_emergence_score"] <= 100.0


def test_audit_proprioceptive_autonomy(test_model):
    auditor = AGIEmergenceAuditor(test_model, device=torch.device("cpu"))
    sample_ids = torch.randint(1, 100, (2, 16))
    res = auditor.audit_proprioceptive_autonomy(sample_ids)

    assert "corollary_discharge_ratio" in res
    assert res["corollary_discharge_ratio"] >= 0.0
    assert "autonomy_classification" in res
    assert isinstance(res["autonomy_classification"], str)
    assert "proprioceptive_autonomy_score" in res
    assert 0.0 <= res["proprioceptive_autonomy_score"] <= 100.0


def test_audit_deliberation_dynamics(test_model):
    auditor = AGIEmergenceAuditor(test_model, device=torch.device("cpu"))
    sample_ids = torch.randint(1, 100, (2, 16))
    res = auditor.audit_deliberation_dynamics(sample_ids, max_cycles=4)

    assert "step_trajectory_drift" in res
    assert len(res["step_trajectory_drift"]) == 4
    assert "total_deliberation_drift" in res
    assert res["total_deliberation_drift"] >= 0.0
    assert "halting_probabilities" in res


def test_audit_lyapunov_and_spectral_stability(test_model):
    auditor = AGIEmergenceAuditor(test_model, device=torch.device("cpu"))
    sample_ids = torch.randint(1, 100, (1, 16))
    res = auditor.audit_lyapunov_and_spectral_stability(sample_ids, divergence_steps=4)

    assert "peak_spectral_radius" in res
    assert res["peak_spectral_radius"] > 0.0
    assert "finite_time_lyapunov_exponent" in res
    assert "dynamical_regime" in res


def test_compute_full_agi_emergence_report(test_model):
    auditor = AGIEmergenceAuditor(test_model, device=torch.device("cpu"))
    sample_ids = torch.randint(1, 100, (2, 16))
    report = auditor.compute_full_agi_emergence_report(
        checkpoint_path="test_ckpt.pt",
        sample_input_ids=sample_ids,
    )

    assert isinstance(report, EmergenceReport)
    assert 0.0 <= report.agi_emergence_score <= 100.0
    assert isinstance(report.is_safe_to_scale, bool)
    assert len(report.scale_up_recommendations) > 0
    assert report.total_params > 0
