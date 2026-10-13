from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_separation(
    temperature_true,
    pressure_true,
    mixed,
    temperature_hat,
    pressure_hat,
    *,
    snr_temperature: float,
    snr_pressure: float,
    save_path: str | None = None,
):
    t = np.asarray(temperature_true).reshape(-1)
    p = np.asarray(pressure_true).reshape(-1)
    m = np.asarray(mixed).reshape(-1)
    th = np.asarray(temperature_hat).reshape(-1)
    ph = np.asarray(pressure_hat).reshape(-1)
    idx = np.arange(len(t))

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    axes[0].plot(idx, m, marker="o")
    axes[0].set_title("Mixed Signal")
    axes[0].set_xlabel("Sensor index")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(idx, t, marker="o", label="Ground Truth")
    axes[1].plot(idx, th, marker="^", linestyle="--", label=f"GSS-LSF ({snr_temperature:.2f} dB)")
    axes[1].set_title("Temperature")
    axes[1].set_xlabel("Sensor index")
    axes[1].set_ylabel("Normalized value")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(idx, p, marker="o", label="Ground Truth")
    axes[2].plot(idx, ph, marker="^", linestyle="--", label=f"GSS-LSF ({snr_pressure:.2f} dB)")
    axes[2].set_title("Pressure")
    axes[2].set_xlabel("Sensor index")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes
