"""
Phase I Mathematical Validation: Negative-Covariance Regularizer Analytic Gradient.
Compares closed-form analytic gradient against PyTorch torch.autograd.grad.
"""

from typing import Optional

import pytest
import torch


def compute_analytic_negative_covariance_gradient(
    W_S: torch.Tensor,
    mu_V: Optional[torch.Tensor] = None,
    margin: float = 0.10,
    eps: float = 1e-6,
    use_local_centering_proj: bool = False,
) -> torch.Tensor:
    """
    Computes the exact analytic gradient of R_geom wrt W_S without autograd.
    """
    B, D = W_S.shape
    denom = B * (B - 1)

    # 1. Centering (Global Reference Centroid vs Local Projection)
    if use_local_centering_proj:
        X_S = W_S - W_S.mean(dim=0, keepdim=True)
    else:
        if mu_V is None:
            mu_V = torch.zeros(1, D, device=W_S.device)
        X_S = W_S - mu_V

    # 2. Row Normalization
    norms = X_S.norm(dim=1, keepdim=True).clamp_min(eps)
    U = X_S / norms

    # 3. Cosine Gram Matrix
    G = U @ U.t()

    # 4. Asymmetric Positive Margin Mask (exclude diagonal)
    mask = ~torch.eye(B, dtype=torch.bool, device=W_S.device)
    H = torch.clamp(G - margin, min=0.0) * mask.float()

    # 5. Gradient wrt U
    grad_U = (2.0 / denom) * (H @ U)

    # 6. Tangent Space Projection Through Row Normalization
    u_dot_g = (U * grad_U).sum(dim=1, keepdim=True)
    grad_X = (grad_U - U * u_dot_g) / norms

    # 7. Centering Derivative
    if use_local_centering_proj:
        grad_W = grad_X - grad_X.mean(dim=0, keepdim=True)
    else:
        grad_W = grad_X

    return grad_W


def test_analytic_gradient_vs_autograd_global_reference():
    torch.manual_seed(42)
    B = 32
    D = 64
    V = 128
    m = 0.10
    eps = 1e-6

    # Vocabulary embedding matrix
    W = torch.randn(V, D, dtype=torch.float32)
    indices = torch.randperm(V)[:B]
    W_S = W[indices].clone().detach().requires_grad_(True)

    # Global centroid treated as non-differentiable reference
    mu_V = W.mean(dim=0, keepdim=True).detach()

    # Autograd Forward
    X_S = W_S - mu_V
    norms = X_S.norm(dim=1, keepdim=True).clamp_min(eps)
    U = X_S / norms
    G = U @ U.t()
    mask = ~torch.eye(B, dtype=torch.bool)
    H = torch.clamp(G - m, min=0.0) * mask.float()
    denom = B * (B - 1)
    loss = 0.5 * (H ** 2).sum() / denom

    grad_autograd = torch.autograd.grad(loss, W_S)[0]

    # Analytic gradient
    grad_analytic = compute_analytic_negative_covariance_gradient(
        W_S=W_S.detach(),
        mu_V=mu_V,
        margin=m,
        eps=eps,
        use_local_centering_proj=False,
    )

    diff = (grad_autograd - grad_analytic).abs().max().item()
    rel_err = ((grad_autograd - grad_analytic).norm() / (grad_autograd.norm() + 1e-9)).item()

    assert rel_err < 1e-5, f"Relative error {rel_err} exceeded tolerance 1e-5 (max diff: {diff})"
    assert diff < 1e-7, f"Max diff {diff} exceeded tolerance 1e-7"


def test_analytic_gradient_vs_autograd_local_centering():
    torch.manual_seed(123)
    B = 24
    D = 48
    m = 0.15
    eps = 1e-6

    W_S = torch.randn(B, D, dtype=torch.float32, requires_grad=True)

    # Local centering forward
    X_S = W_S - W_S.mean(dim=0, keepdim=True)
    norms = X_S.norm(dim=1, keepdim=True).clamp_min(eps)
    U = X_S / norms
    G = U @ U.t()
    mask = ~torch.eye(B, dtype=torch.bool)
    H = torch.clamp(G - m, min=0.0) * mask.float()
    denom = B * (B - 1)
    loss = 0.5 * (H ** 2).sum() / denom

    grad_autograd = torch.autograd.grad(loss, W_S)[0]

    # Analytic gradient with local centering projector
    grad_analytic = compute_analytic_negative_covariance_gradient(
        W_S=W_S.detach(),
        mu_V=torch.zeros(1, D),
        margin=m,
        eps=eps,
        use_local_centering_proj=True,
    )

    diff = (grad_autograd - grad_analytic).abs().max().item()
    rel_err = ((grad_autograd - grad_analytic).norm() / (grad_autograd.norm() + 1e-9)).item()

    assert rel_err < 1e-5, f"Relative error {rel_err} exceeded tolerance 1e-5"
    assert diff < 1e-7, f"Max diff {diff} exceeded tolerance 1e-7"


def test_zero_penalty_on_orthogonal_or_negative_pairs():
    """Verify that orthogonal and negative cosine pairs incur exactly zero loss and zero gradient."""
    B = 4
    D = 4
    # Mutually orthogonal vectors (identity matrix)
    W_S = torch.eye(B, D, dtype=torch.float32)
    mu_V = torch.zeros(1, D)

    grad = compute_analytic_negative_covariance_gradient(
        W_S=W_S,
        mu_V=mu_V,
        margin=0.05,
        eps=1e-6,
    )

    # Because all off-diagonal cosines are 0.0 <= margin (0.05), gradient must be identically zero
    assert torch.allclose(grad, torch.zeros_like(grad)), "Orthogonal pairs must yield zero gradient penalty!"
