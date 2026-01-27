import numpy as np
import torch
import matplotlib.pyplot as plt
import timesfm
import os

# ============================================================
# 0. DOSSIER RESULTS
# ============================================================

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 1. SIMULATION PHYSIQUE (DIFFUSION DE LA CHALEUR 2D)
# ============================================================

def simulate_heat_diffusion(nx=64, ny=64, nt=200, dx=1.0, dy=1.0, dt=0.1, alpha=1.0):
    T = np.zeros((nt, nx, ny), dtype=np.float32)

    cx, cy = nx // 2, ny // 2
    T[0, cx-2:cx+2, cy-2:cy+2] = 100.0

    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    hole = (X - cx)**2 + (Y - cy)**2 < (nx // 6)**2

    for t in range(nt - 1):
        laplacian = (
            np.roll(T[t], 1, axis=0) +
            np.roll(T[t], -1, axis=0) +
            np.roll(T[t], 1, axis=1) +
            np.roll(T[t], -1, axis=1) -
            4 * T[t]
        ) / (dx * dy)

        T[t+1] = T[t] + alpha * dt * laplacian
        T[t+1][hole] = 0.0

    return T


# ============================================================
# 2. VOLUME <-> SERIES TEMPORELLES
# ============================================================

def volume_to_timeseries(volume):
    T, X, Y = volume.shape
    return volume.reshape(T, X * Y).T


def timeseries_to_volume(timeseries, spatial_shape):
    N, T = timeseries.shape
    X, Y = spatial_shape
    return timeseries.T.reshape(T, X, Y)


# ============================================================
# 3. FORECASTING ZERO-SHOT AVEC TIMESFM (TORCH)
# ============================================================

class TimesFMForecaster:
    def __init__(self, max_context=1024, max_horizon=256):
        torch.set_float32_matmul_precision("high")

        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )

        self.model.compile(
            timesfm.ForecastConfig(
                max_context=max_context,
                max_horizon=max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

    def forecast(self, timeseries, horizon, batch_size=64):
        N = len(timeseries)
        point_all, quantile_all = [], []

        for i in range(0, N, batch_size):
            batch = timeseries[i:i + batch_size]
            inputs = [series.astype(np.float32) for series in batch]

            points, quantiles = self.model.forecast(
                horizon=horizon,
                inputs=inputs
            )

            point_all.append(points)
            quantile_all.append(quantiles)

        return np.vstack(point_all), np.vstack(quantile_all)


# ============================================================
# 4. VISUALISATION (SAVE TO DISK)
# ============================================================

def show_slice(volume, timestep, title="", filename=None):
    plt.figure(figsize=(5, 5))
    plt.imshow(volume[timestep], cmap="inferno")
    plt.colorbar()
    plt.title(title)
    plt.axis("off")

    if filename is None:
        filename = title.replace(" ", "_").lower()

    path = os.path.join(RESULTS_DIR, f"{filename}.png")
    plt.savefig(path, dpi=150)
    plt.close()


def compare_true_vs_pred(true_vol, pred_vol, timestep, filename="comparison"):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(true_vol[timestep], cmap="inferno")
    axs[0].set_title("Ground Truth")

    axs[1].imshow(pred_vol[timestep], cmap="inferno")
    axs[1].set_title("Prediction")

    axs[2].imshow(true_vol[timestep] - pred_vol[timestep], cmap="coolwarm")
    axs[2].set_title("Error")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{filename}.png")
    plt.savefig(path, dpi=150)
    plt.close()

def plot_temperature_at_point(true_vol, pred_vol, point, title="Temperature at point",filename="temp_evo"):
    """
    Trace la température vraie et prédite à un point spécifique au fil du temps.
    
    true_vol : (T_true, X, Y)
    pred_vol : (T_pred, X, Y)
    point : tuple (x, y)
    """
    x, y = point

    true_series = true_vol[:, x, y]
    pred_series = pred_vol[:, x, y]

    print ("temperature réelle : ", true_series[-1])
    print ("temperature prédite : ", pred_series[-1])

    T_true = len(true_series)
    T_pred = len(pred_series)

    # On trace la simulation complète
    plt.figure(figsize=(8,4))
    plt.plot(range(T_true), true_series, label="True Temp", linewidth=2)
    # On trace seulement la portion prédite disponible
    plt.plot(range(T_pred), pred_series, '--', label="Predicted Temp", linewidth=2)

    plt.xlabel("Time step")
    plt.ylabel("Temperature")
    plt.title(f"{title} at point ({x}, {y})")
    plt.legend()
    plt.grid(True)
    if filename is None:
        filename = title.replace(" ", "_").lower()

    path = os.path.join(RESULTS_DIR, f"{filename}.png")
    plt.savefig(path, dpi=150)
    plt.close()
