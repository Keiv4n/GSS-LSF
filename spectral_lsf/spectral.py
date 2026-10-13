from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_graph(signal: torch.Tensor, k: int = 5) -> torch.Tensor:

    signal = signal.reshape(-1, 1).to(dtype=torch.float32)
    n = signal.shape[0]
    if n < 2:
        raise ValueError("At least two sensors are required.")
    if not 1 <= k <= n - 1:
        raise ValueError(f"k must be in [1, {n - 1}], got {k}.")

    distances = torch.cdist(signal, signal)
    weights = 1.0 / (distances + 1e-5)
    weights.fill_diagonal_(0)

    for i in range(n):
        _, idx = torch.topk(weights[i], k=k)
        mask = torch.ones(n, dtype=torch.bool, device=weights.device)
        mask[idx] = False
        weights[i, mask] = 0

    weights = 0.5 * (weights + weights.T)
    degree = torch.diag(weights.sum(dim=1))
    return degree - weights


def snr_db(x_true, x_pred) -> float:
    x_true = np.asarray(x_true, dtype=float).reshape(-1)
    x_pred = np.asarray(x_pred, dtype=float).reshape(-1)
    numerator = np.sum(x_true**2)
    denominator = np.sum((x_true - x_pred) ** 2) + 1e-12
    return float(10.0 * np.log10(numerator / denominator))


def graph_fourier_vectors(laplacian: torch.Tensor) -> torch.Tensor:
    """Return non-constant Laplacian eigenvectors, matching [:, 1:]."""
    _, eigvecs = torch.linalg.eigh(laplacian)
    return eigvecs[:, 1:]


def closed_form_spectral_from_bases(
    u1_all: torch.Tensor,
    u2_all: torch.Tensor,
    mixed: torch.Tensor,
    k1: int,
    k2: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_k1 = u1_all.shape[1]
    max_k2 = u2_all.shape[1]
    if not 1 <= k1 <= max_k1:
        raise ValueError(f"k1 must be in [1, {max_k1}], got {k1}.")
    if not 1 <= k2 <= max_k2:
        raise ValueError(f"k2 must be in [1, {max_k2}], got {k2}.")

    u1 = u1_all[:, :k1]
    u2 = u2_all[:, :k2]
    design = torch.cat([u1, u2], dim=1)
    coeff = torch.linalg.lstsq(design, mixed).solution
    x1_hat = u1 @ coeff[:k1]
    x2_hat = u2 @ coeff[k1:]
    return x1_hat.detach(), x2_hat.detach()


def closed_form_spectral(
    l1: torch.Tensor,
    l2: torch.Tensor,
    mixed: torch.Tensor,
    k1: int,
    k2: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return closed_form_spectral_from_bases(
        graph_fourier_vectors(l1),
        graph_fourier_vectors(l2),
        mixed,
        k1,
        k2,
    )


@dataclass(frozen=True)
class SpectralSearchResult:
    temperature: torch.Tensor
    pressure: torch.Tensor
    k_temperature: int
    k_pressure: int
    snr_temperature: float
    snr_pressure: float

    @property
    def average_snr(self) -> float:
        return 0.5 * (self.snr_temperature + self.snr_pressure)


def spectral_grid_search(
    l1: torch.Tensor,
    l2: torch.Tensor,
    mixed: torch.Tensor,
    gt1: torch.Tensor,
    gt2: torch.Tensor,
    *,
    k_min: int = 2,
    k_max: int = 50,
) -> SpectralSearchResult:
    u1_all = graph_fourier_vectors(l1)
    u2_all = graph_fourier_vectors(l2)
    upper = min(k_max, u1_all.shape[1], u2_all.shape[1])
    if k_min > upper:
        raise ValueError(f"k_min={k_min} exceeds available spectral dimension {upper}.")

    gt1_np = gt1.detach().cpu().numpy()
    gt2_np = gt2.detach().cpu().numpy()

    best: SpectralSearchResult | None = None
    for k1 in range(k_min, upper + 1):
        for k2 in range(k_min, upper + 1):
            x1_hat, x2_hat = closed_form_spectral_from_bases(
                u1_all, u2_all, mixed, k1, k2
            )
            s1 = snr_db(gt1_np, x1_hat.cpu().numpy())
            s2 = snr_db(gt2_np, x2_hat.cpu().numpy())
            candidate = SpectralSearchResult(
                temperature=x1_hat,
                pressure=x2_hat,
                k_temperature=k1,
                k_pressure=k2,
                snr_temperature=s1,
                snr_pressure=s2,
            )
            if best is None or candidate.average_snr > best.average_snr:
                best = candidate

    assert best is not None
    return best


def make_noisy_mixture(
    temperature: torch.Tensor,
    pressure: torch.Tensor,
    *,
    noise_std: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    clean = temperature + pressure
    noise = noise_std * torch.randn_like(clean)
    return clean + noise, noise


@dataclass(frozen=True)
class LearnableSpectralResult:
    """GSS-LSF estimated components, learned gains and optimization history."""
    temperature: torch.Tensor
    pressure: torch.Tensor
    gains_temperature: torch.Tensor
    gains_pressure: torch.Tensor
    losses: tuple[float, ...]
    input_seed: int


def learnable_spectral_filters(
    l1: torch.Tensor,
    l2: torch.Tensor,
    mixed: torch.Tensor,
    k1: int,
    k2: int,
    *,
    steps: int = 2500,
    lr: float = 0.05,
    seed: int = 42,
    center_mixture: bool = True,
) -> LearnableSpectralResult:
    """Unsupervised GSS-LSF using the paper's fixed-input spectral-gain model.

    x_p = U_p[:, :k_p] @ (alpha_p * (U_p[:, :k_p].T @ z)),
    with one shared, frozen random z and trainable gain vectors alpha_p.
    Adam minimizes ||m - x_1 - x_2||_2^2 (per-element mean).

    The graph Laplacians must be supplied externally; this function does not
    construct them from the unknown sources. The graph bases exclude the first
    eigenvector, and outputs are zero mean. Consequently center_mixture=True
    subtracts the mixture mean and only the centered mixture is reconstructed.
    """
    if steps < 1 or lr <= 0:
        raise ValueError("steps must be >= 1 and lr must be positive")
    if l1.shape != l2.shape or l1.ndim != 2 or l1.shape[0] != l1.shape[1]:
        raise ValueError("l1 and l2 must be square matrices of equal size")
    n = l1.shape[0]
    if mixed.numel() != n:
        raise ValueError("mixed must contain one observation per graph node")
    if not (1 <= k1 < n and 1 <= k2 < n):
        raise ValueError("k1 and k2 must be between 1 and N-1")

    orig_shape = mixed.shape
    dtype = l1.dtype
    device = l1.device
    basis1 = graph_fourier_vectors(l1)[:, :k1].detach()
    basis2 = graph_fourier_vectors(l2.to(device=device, dtype=dtype))[:, :k2].detach()
    y = mixed.to(device=device, dtype=dtype).reshape(-1)
    if center_mixture:
        y = y - y.mean()

    # Dedicated seeded generator: no changes to the caller's random state.
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    z = torch.randn(n, generator=generator, device=device, dtype=dtype)
    z1 = basis1.T @ z
    z2 = basis2.T @ z
    alpha1 = torch.nn.Parameter(torch.ones(k1, device=device, dtype=dtype))
    alpha2 = torch.nn.Parameter(torch.ones(k2, device=device, dtype=dtype))
    optimizer = torch.optim.Adam([alpha1, alpha2], lr=lr)
    history = []
    for _ in range(steps):
        optimizer.zero_grad()
        x1 = basis1 @ (alpha1 * z1)
        x2 = basis2 @ (alpha2 * z2)
        # Zero-DC constraint also works for numerically approximate eigenvectors.
        x1 = x1 - x1.mean()
        x2 = x2 - x2.mean()
        loss = torch.mean((y - x1 - x2) ** 2)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    with torch.no_grad():
        x1 = basis1 @ (alpha1 * z1)
        x2 = basis2 @ (alpha2 * z2)
        x1 = (x1 - x1.mean()).reshape(orig_shape)
        x2 = (x2 - x2.mean()).reshape(orig_shape)
        return LearnableSpectralResult(
            x1.detach(), x2.detach(), alpha1.detach().clone(),
            alpha2.detach().clone(), tuple(history), seed,
        )
