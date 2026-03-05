"""
=============================================================================
Forecast Volumétrique Météo avec TimesFM vs Chronos — Benchmark Complet
=============================================================================
Deux approches de modélisation :
  1/ Forecast 3D direct d'un nuage (application forecaster voxel par voxel)
  2/ Forecast en espace latent (3D CNN Encoder → Forecaster → Decoder)

Deux forecasters comparés :
  - TimesFM (Google, 2.5-200M)
  - Chronos (Amazon, T5 variants)
  + Baseline linéaire (substitut si modèles indisponibles)

Dataset : ERA5 (Copernicus) ou données synthétiques si non disponible
Sorties  : GIF 3D comparatif + courbes voxel + métriques PSNR/SSIM/physique
           → dossier resultat_meteo_compare/
=============================================================================
"""

import os
import warnings
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import time
import traceback

warnings.filterwarnings('ignore')

# ─── Dépendances optionnelles ─────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    HAS_TORCH = False
    DEVICE = "cpu"
    print("[WARN] PyTorch non disponible.")

# ── TimesFM ──────────────────────────────────────────────────────────────────
try:
    import timesfm
    if hasattr(timesfm, 'TimesFM_2p5_200M_torch'):
        TIMESFM_VERSION = 2
    elif hasattr(timesfm, 'TimesFm'):
        TIMESFM_VERSION = 1
    else:
        TIMESFM_VERSION = 0
    HAS_TIMESFM = TIMESFM_VERSION > 0
    if HAS_TIMESFM:
        print(f"[OK] TimesFM détecté (API v{TIMESFM_VERSION}).")
    else:
        print("[WARN] timesfm importé mais API non reconnue.")
except ImportError:
    HAS_TIMESFM = False
    TIMESFM_VERSION = 0
    print("[WARN] TimesFM non disponible → substitut linéaire utilisé.")

# ── Chronos ───────────────────────────────────────────────────────────────────
try:
    from chronos import ChronosPipeline
    HAS_CHRONOS = True
    print("[OK] Chronos détecté.")
except ImportError:
    HAS_CHRONOS = False
    print("[WARN] Chronos non disponible → substitut linéaire utilisé.")

# ── Autres ───────────────────────────────────────────────────────────────────
try:
    import cdsapi
    HAS_CDS = True
except ImportError:
    HAS_CDS = False

try:
    from skimage.metrics import structural_similarity as ssim_2d
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ─── Configuration globale ────────────────────────────────────────────────────
OUTPUT_DIR = Path("resultat_meteo_compare")
OUTPUT_DIR.mkdir(exist_ok=True)

NX, NY, NZ = 16, 16, 8
N_HIST     = 24
N_PRED     = 6
LATENT_DIM = 32
BATCH_SIZE = 4
N_EPOCHS   = 30


# =============================================================================
# 1. GÉNÉRATION / CHARGEMENT DES DONNÉES
# =============================================================================

def download_era5(out_path: Path) -> bool:
    import cdsapi
    dataset = "reanalysis-era5-pressure-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": ["fraction_of_cloud_cover",
                     "specific_cloud_liquid_water_content",
                     "specific_cloud_ice_water_content"],
        "pressure_level": ["500", "600", "700", "750", "800", "850", "925", "1000"],
        "year": ["2023"], "month": ["10"],
        "day": ["27", "28", "29"],
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf",
        "area": [51.1, 5.2, 43.3, 9.6],
    }
    try:
        client = cdsapi.Client()
        print("   Connexion CDS OK, téléchargement en cours...")
        client.retrieve(dataset, request).download(str(out_path))
        print(f"   [OK] ERA5 téléchargé → {out_path}")
        return True
    except Exception as e:
        print(f"   [WARN] Téléchargement ERA5 échoué : {e}")
        return False


def load_era5_netcdf(path: Path) -> Optional[np.ndarray]:
    try:
        import netCDF4 as nc
        from scipy.ndimage import zoom
        ds = nc.Dataset(path)
        var = None
        for name in ['cc', 'fraction_of_cloud_cover', 'cldfra']:
            if name in ds.variables:
                var = ds.variables[name][:]
                break
        if var is None:
            return None
        data = np.array(var)
        if data.ndim == 4:
            data = data.transpose(0, 2, 3, 1)
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        T = data.shape[0]
        factors = (1, NX / data.shape[1], NY / data.shape[2], NZ / data.shape[3])
        data = zoom(data, factors, order=1)
        return data[:N_HIST + N_PRED]
    except Exception as e:
        print(f"[WARN] Lecture NetCDF échouée : {e}")
        return None


def generate_synthetic_cloud(
    nx=NX, ny=NY, nz=NZ,
    n_steps=None, seed=42
) -> np.ndarray:
    """Génère un nuage synthétique ERA5-like animé."""
    if n_steps is None:
        n_steps = N_HIST + N_PRED
    np.random.seed(seed)
    data = np.zeros((n_steps, nx, ny, nz), dtype=np.float32)
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    cx0, cy0, cz0 = 0.0, -0.5, 0.4
    vx, vy = 0.04, 0.02
    sigma_xy, sigma_z = 0.35, 0.25
    freq = np.random.randn(3, 4, 8) * 0.1

    for t in range(n_steps):
        cx = cx0 + vx * t + 0.05 * np.sin(0.3 * t)
        cy = cy0 + vy * t + 0.03 * np.cos(0.2 * t)
        cz = cz0 + 0.01 * np.sin(0.15 * t)
        sig_xy = sigma_xy * (1 + 0.1 * np.sin(0.25 * t))
        sig_z  = sigma_z  * (1 + 0.05 * np.cos(0.18 * t))

        cloud = np.exp(
            -((X - cx)**2 + (Y - cy)**2) / (2 * sig_xy**2)
            -((Z - cz)**2) / (2 * sig_z**2)
        )
        cx2 = -0.3 + 0.03 * t
        cy2 = 0.3 - 0.01 * t
        cloud2 = 0.4 * np.exp(
            -((X - cx2)**2 + (Y - cy2)**2) / (2 * (sigma_xy * 0.7)**2)
            -((Z - 0.6)**2) / (2 * sig_z**2)
        )
        turb = np.zeros((nx, ny, nz))
        for k in range(4):
            fk = (k + 1) * 2
            turb += 0.08 / (k + 1) * np.sin(
                fk * X[:, :, 0:1] + freq[0, k % 4, 0] * t
            ) * np.sin(
                fk * Y[:, :, 0:1] + freq[1, k % 4, 1] * t
            ) * np.cos(
                fk * Z[0:1, :, :] + freq[2, k % 4, 2] * t
            )
        data[t] = np.clip(cloud + cloud2 + 0.15 * turb, 0, 1)
    return data


# =============================================================================
# 2. MESURE RESSOURCES
# =============================================================================

class ResourceMonitor:
    """Mesure CPU, RAM, GPU et temps d'exécution."""

    def __init__(self, label: str):
        self.label = label
        self.t_start = None
        self.t_end = None
        self.ram_before = None
        self.ram_after = None
        self.gpu_mem_peak = 0.0

    def __enter__(self):
        self.t_start = time.perf_counter()
        if HAS_PSUTIL:
            proc = psutil.Process()
            self.ram_before = proc.memory_info().rss / 1024**2  # MB
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *args):
        self.t_end = time.perf_counter()
        if HAS_PSUTIL:
            proc = psutil.Process()
            self.ram_after = proc.memory_info().rss / 1024**2
        if HAS_TORCH and torch.cuda.is_available():
            self.gpu_mem_peak = torch.cuda.max_memory_allocated() / 1024**2  # MB

    @property
    def elapsed(self) -> float:
        return (self.t_end or time.perf_counter()) - self.t_start

    @property
    def ram_delta(self) -> float:
        if self.ram_before is None or self.ram_after is None:
            return 0.0
        return self.ram_after - self.ram_before

    def summary(self) -> Dict:
        return {
            "elapsed_s": round(self.elapsed, 3),
            "ram_delta_mb": round(self.ram_delta, 1),
            "gpu_peak_mb": round(self.gpu_mem_peak, 1),
        }


# =============================================================================
# 3. FORECASTERS
# =============================================================================

class SimpleForecaster:
    """Baseline : régression linéaire + amortissement vers la moyenne."""
    name = "Baseline"

    def __init__(self, horizon=N_PRED):
        self.horizon = horizon

    def forecast_series(self, series: np.ndarray) -> np.ndarray:
        T = len(series)
        t = np.arange(T)
        coeffs = np.polyfit(t[-min(T, 12):], series[-min(T, 12):], deg=1)
        future_t = np.arange(T, T + self.horizon)
        pred = np.polyval(coeffs, future_t)
        mu = series[-8:].mean()
        alpha = np.linspace(0, 0.5, self.horizon)
        return np.clip(pred * (1 - alpha) + mu * alpha, 0, 1).astype(np.float32)

    def forecast_batch(self, series_list: List[np.ndarray]) -> np.ndarray:
        return np.stack([self.forecast_series(s) for s in series_list])


class TimesFMForecaster:
    """Wrapper TimesFM (v1 ou v2)."""
    name = "TimesFM"

    def __init__(self, horizon=N_PRED, max_context=512):
        if not HAS_TIMESFM:
            raise RuntimeError("TimesFM non disponible")
        if HAS_TORCH:
            torch.set_float32_matmul_precision("high")
        try:
            if TIMESFM_VERSION == 2:
                self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                    "google/timesfm-2.5-200m-pytorch"
                )
                self.model.compile(
                    timesfm.ForecastConfig(
                        max_context=max_context,
                        max_horizon=horizon,
                        normalize_inputs=True,
                        use_continuous_quantile_head=False,
                        infer_is_positive=True,
                    )
                )
            else:
                self.model = timesfm.TimesFm(
                    hparams=timesfm.TimesFmHparams(
                        backend="gpu" if (HAS_TORCH and torch.cuda.is_available()) else "cpu",
                        per_core_batch_size=32,
                        horizon_len=horizon,
                        num_layers=20,
                        model_dims=1280,
                    ),
                    checkpoint=timesfm.TimesFmCheckpoint(
                        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
                    ),
                )
            print(f"[OK] TimesFM v{TIMESFM_VERSION} chargé.")
        except Exception as e:
            raise RuntimeError(f"Impossible de charger TimesFM : {e}")

    def forecast_batch(self, series_list: List[np.ndarray]) -> np.ndarray:
        inputs = [s.astype(np.float32).tolist() for s in series_list]
        if TIMESFM_VERSION == 2:
            points, _ = self.model.forecast(horizon=N_PRED, inputs=inputs)
            return np.array(points, dtype=np.float32)
        else:
            _, preds = self.model.forecast([s.tolist() for s in series_list], freq=[0] * len(series_list))
            return np.array(preds, dtype=np.float32)

    def forecast_series(self, series: np.ndarray) -> np.ndarray:
        return self.forecast_batch([series])[0]


class ChronosForecaster:
    """Wrapper Chronos (Amazon)."""
    name = "Chronos"

    def __init__(self, horizon=N_PRED,
                 model_name="amazon/chronos-t5-small",
                 num_samples=20):
        if not HAS_CHRONOS:
            raise RuntimeError("Chronos non disponible")
        if not HAS_TORCH:
            raise RuntimeError("PyTorch requis pour Chronos")
        self.horizon = horizon
        self.num_samples = num_samples
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Chargement Chronos ({model_name}) sur {device}...")
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        print(f"[OK] Chronos chargé.")

    def forecast_batch(self, series_list: List[np.ndarray],
                       batch_size: int = 8) -> np.ndarray:
        N = len(series_list)
        results = np.zeros((N, self.horizon), dtype=np.float32)
        for i in range(0, N, batch_size):
            batch = [torch.tensor(s, dtype=torch.float32)
                     for s in series_list[i:i + batch_size]]
            try:
                with torch.no_grad():
                    fc = self.pipeline.predict(
                        batch,
                        prediction_length=self.horizon,
                        num_samples=self.num_samples,
                    )
                results[i:i + len(batch)] = np.median(fc.numpy(), axis=1)
                del fc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print("[WARN] OOM Chronos → mode série...")
                for j, s in enumerate(batch):
                    with torch.no_grad():
                        fc = self.pipeline.predict(
                            [s], prediction_length=self.horizon,
                            num_samples=self.num_samples
                        )
                    results[i + j] = np.median(fc.numpy(), axis=1)[0]
                    del fc
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        return results

    def forecast_series(self, series: np.ndarray) -> np.ndarray:
        return self.forecast_batch([series])[0]


def build_forecasters(horizon=N_PRED,
                      chronos_model="amazon/chronos-t5-small") -> Dict:
    """Construit tous les forecasters disponibles."""
    forecasters = {}

    # TimesFM
    if HAS_TIMESFM:
        try:
            forecasters["TimesFM"] = TimesFMForecaster(horizon=horizon)
        except Exception as e:
            print(f"[WARN] TimesFM skippé : {e}")
            forecasters["TimesFM"] = SimpleForecaster(horizon=horizon)
    else:
        forecasters["TimesFM"] = SimpleForecaster(horizon=horizon)

    # Chronos
    if HAS_CHRONOS and HAS_TORCH:
        try:
            forecasters["Chronos"] = ChronosForecaster(
                horizon=horizon, model_name=chronos_model
            )
        except Exception as e:
            print(f"[WARN] Chronos skippé : {e}")
            forecasters["Chronos"] = SimpleForecaster(horizon=horizon)
    else:
        forecasters["Chronos"] = SimpleForecaster(horizon=horizon)

    return forecasters


# =============================================================================
# 4. APPROCHE 1 : FORECAST DIRECT VOXEL PAR VOXEL
# =============================================================================

def forecast_direct(
    data_hist: np.ndarray,
    forecaster,
    label: str = "forecaster",
    chronos_batch_size: int = 64,
) -> Tuple[np.ndarray, Dict]:
    """
    Forecast direct voxel par voxel.
    Retourne (pred, resource_stats).
    """
    print(f"\n[Direct] Forecast avec {label}...")
    T_hist, nx, ny, nz = data_hist.shape
    pred = np.zeros((N_PRED, nx, ny, nz), dtype=np.float32)

    # Reshape en liste de séries
    series_list = [data_hist[:, ix, iy, iz]
                   for ix in range(nx)
                   for iy in range(ny)
                   for iz in range(nz)]
    N_vox = len(series_list)

    with ResourceMonitor(label) as mon:
        # Batch forecast si disponible
        if isinstance(forecaster, ChronosForecaster):
            preds_flat = forecaster.forecast_batch(series_list,
                                                   batch_size=chronos_batch_size)
        elif isinstance(forecaster, TimesFMForecaster):
            # TimesFM gère les batches natifs
            TFMBATCH = 256
            preds_flat = np.zeros((N_vox, N_PRED), dtype=np.float32)
            for i in range(0, N_vox, TFMBATCH):
                batch = series_list[i:i + TFMBATCH]
                preds_flat[i:i + len(batch)] = forecaster.forecast_batch(batch)
                if i % 500 == 0:
                    print(f"   {i}/{N_vox} voxels...", end='\r')
        else:
            # Baseline série par série
            preds_flat = np.stack(
                [forecaster.forecast_series(s) for s in series_list]
            )

    # Reshape vers (N_PRED, nx, ny, nz)
    for k, (ix, iy, iz) in enumerate(
        [(ix, iy, iz)
         for ix in range(nx)
         for iy in range(ny)
         for iz in range(nz)]
    ):
        pred[:, ix, iy, iz] = preds_flat[k]

    stats = mon.summary()
    print(f"   [OK] Direct {label} : {stats['elapsed_s']:.2f}s, "
          f"ΔRAM={stats['ram_delta_mb']:.0f}MB, "
          f"GPU peak={stats['gpu_peak_mb']:.0f}MB")
    return pred, stats


# =============================================================================
# 5. APPROCHE 2 : FORECAST EN ESPACE LATENT
# =============================================================================

if HAS_TORCH:
    class Encoder3D(nn.Module):
        def __init__(self, latent_dim=LATENT_DIM):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(1, 16, 3, padding=1), nn.LeakyReLU(0.2),
                nn.MaxPool3d(2),
                nn.Conv3d(16, 32, 3, padding=1), nn.LeakyReLU(0.2),
                nn.MaxPool3d(2),
                nn.Conv3d(32, 64, 3, padding=1), nn.LeakyReLU(0.2),
                nn.AdaptiveAvgPool3d(2),
            )
            self.fc = nn.Linear(64 * 8, latent_dim)

        def forward(self, x):
            h = self.net(x)
            return self.fc(h.view(h.size(0), -1))

    class Decoder3D(nn.Module):
        def __init__(self, latent_dim=LATENT_DIM, nx=NX, ny=NY, nz=NZ):
            super().__init__()
            self.nx, self.ny, self.nz = nx, ny, nz
            self.fc = nn.Linear(latent_dim, 64 * 8)
            self.net = nn.Sequential(
                nn.ConvTranspose3d(64, 32, 3, padding=1), nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                nn.ConvTranspose3d(32, 16, 3, padding=1), nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                nn.ConvTranspose3d(16, 1, 3, padding=1),
                nn.Sigmoid(),
            )

        def forward(self, z):
            h = self.fc(z).view(-1, 64, 2, 2, 2)
            out = self.net(h)
            return F.interpolate(out, size=(self.nx, self.ny, self.nz),
                                 mode='trilinear', align_corners=False)

    class Autoencoder3D(nn.Module):
        def __init__(self, latent_dim=LATENT_DIM):
            super().__init__()
            self.encoder = Encoder3D(latent_dim)
            self.decoder = Decoder3D(latent_dim)

        def forward(self, x):
            return self.decoder(self.encoder(x))

        def encode(self, x):
            return self.encoder(x)

        def decode(self, z):
            return self.decoder(z)


def train_autoencoder(data: np.ndarray, n_epochs=N_EPOCHS) -> "Autoencoder3D":
    print("\n[Autoencoder] Entraînement...")
    model = Autoencoder3D(LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    X_tensor = torch.tensor(data[:, np.newaxis, ...], dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=BATCH_SIZE, shuffle=True)

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        model.train()
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            loss = F.mse_loss(model(batch), batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg = epoch_loss / len(loader)
        losses.append(avg)
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1:3d}/{n_epochs}  Loss={avg:.6f}")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(losses, color='steelblue', lw=2)
    ax.set_title("Courbe de perte — Autoencoder 3D")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "autoencoder_loss.png", dpi=120)
    plt.close(fig)
    print("[OK] Autoencoder entraîné.")
    return model


def forecast_latent(
    data_hist: np.ndarray,
    autoencoder: "Autoencoder3D",
    forecaster,
    label: str = "forecaster",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Forecast en espace latent. Retourne (pred, latents, stats)."""
    print(f"\n[Latent] Forecast avec {label}...")
    autoencoder.eval()

    with torch.no_grad():
        frames = torch.tensor(data_hist[:, np.newaxis, ...], dtype=torch.float32).to(DEVICE)
        latents = autoencoder.encode(frames).cpu().numpy()

    series_list = [latents[:, d] for d in range(LATENT_DIM)]

    with ResourceMonitor(label) as mon:
        if isinstance(forecaster, ChronosForecaster):
            latent_pred_T = forecaster.forecast_batch(series_list)
        elif isinstance(forecaster, TimesFMForecaster):
            latent_pred_T = forecaster.forecast_batch(series_list)
        else:
            latent_pred_T = np.stack(
                [forecaster.forecast_series(s) for s in series_list]
            )

    # shape (LATENT_DIM, N_PRED) → (N_PRED, LATENT_DIM)
    latent_pred = latent_pred_T.T if latent_pred_T.shape[0] == LATENT_DIM else latent_pred_T

    with torch.no_grad():
        z_tensor = torch.tensor(latent_pred, dtype=torch.float32).to(DEVICE)
        volumes = autoencoder.decode(z_tensor).cpu().numpy()

    pred = np.clip(volumes[:, 0, ...], 0, 1)
    stats = mon.summary()
    print(f"   [OK] Latent {label} : {stats['elapsed_s']:.2f}s, "
          f"ΔRAM={stats['ram_delta_mb']:.0f}MB, "
          f"GPU peak={stats['gpu_peak_mb']:.0f}MB")
    return pred, latents, stats


# =============================================================================
# 6. MÉTRIQUES
# =============================================================================

def psnr_volume(gt: np.ndarray, pred: np.ndarray) -> float:
    mse = np.mean((gt - pred) ** 2)
    return 100.0 if mse < 1e-10 else float(10 * np.log10(1.0 / mse))


def ssim_3d_approx(gt: np.ndarray, pred: np.ndarray) -> float:
    if not HAS_SKIMAGE:
        mu1, mu2 = gt.mean(), pred.mean()
        s1, s2 = gt.std(), pred.std()
        cov = np.mean((gt - mu1) * (pred - mu2))
        c1, c2 = 0.01**2, 0.03**2
        return float((2*mu1*mu2 + c1) * (2*cov + c2) /
                     ((mu1**2 + mu2**2 + c1) * (s1**2 + s2**2 + c2)))
    scores = []
    for t in range(gt.shape[0]):
        for iz in range(gt.shape[3]):
            scores.append(ssim_2d(gt[t, :, :, iz], pred[t, :, :, iz], data_range=1.0))
    return float(np.mean(scores))


def physical_error(gt: np.ndarray, pred: np.ndarray) -> Dict:
    mass_gt   = gt.mean(axis=(1, 2, 3))
    mass_pred = pred.mean(axis=(1, 2, 3))
    mass_err  = float(np.mean(np.abs(mass_gt - mass_pred) / (mass_gt + 1e-8)))
    grad_err  = float(np.mean(np.abs(np.gradient(gt, axis=1) - np.gradient(pred, axis=1))))
    dissip    = float(np.var(gt - pred))
    return {"conservation_masse": mass_err, "erreur_gradient": grad_err, "dissipation": dissip}


def temporal_error(gt: np.ndarray, pred: np.ndarray) -> Dict:
    mae_per_t = np.mean(np.abs(gt - pred), axis=(1, 2, 3))
    ix, iy, iz = NX//2, NY//2, NZ//2
    corr = float(np.corrcoef(gt[:, ix, iy, iz], pred[:, ix, iy, iz])[0, 1])
    return {"MAE_par_pas": mae_per_t.tolist(), "correlation_temporelle": corr}


def compute_metrics_for_pred(gt, pred, resource_stats=None) -> Dict:
    m = {
        "PSNR": psnr_volume(gt, pred),
        "SSIM_3D": ssim_3d_approx(gt, pred),
        **physical_error(gt, pred),
        **temporal_error(gt, pred),
    }
    if resource_stats:
        m.update(resource_stats)
    return m


# =============================================================================
# 7. VISUALISATION
# =============================================================================

# Palette couleurs par modèle
MODEL_COLORS = {
    "TimesFM_Direct":  "#E07B39",
    "Chronos_Direct":  "#3A7DC9",
    "TimesFM_Latent":  "#F5A623",
    "Chronos_Latent":  "#7B68EE",
    "Baseline_Direct": "#888888",
}

def volume_to_projection(vol: np.ndarray) -> np.ndarray:
    return vol.max(axis=-1)  # MIP axe Z → (NX, NY)


def make_comparison_gif(
    data_gt: np.ndarray,
    preds: Dict[str, np.ndarray],
    data_hist: np.ndarray,
    filename: str,
    fps: int = 3,
):
    """GIF animé comparant vérité terrain + toutes les prédictions."""
    T_hist = data_hist.shape[0]
    T_pred = data_gt.shape[0]
    all_gt = np.concatenate([data_hist, data_gt], axis=0)

    pred_keys = list(preds.keys())
    n_cols = 1 + len(pred_keys)
    cmaps = ["Blues"] + [
        "Oranges", "Blues", "YlOrBr", "Purples",
        "Greens", "Reds"
    ][:len(pred_keys)]

    fig, axes = plt.subplots(1, n_cols, figsize=(3.8 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    ims = []
    for t in range(T_hist + T_pred):
        row_ims = []
        is_future = t >= T_hist

        # Vérité terrain
        frame_gt = volume_to_projection(all_gt[t])
        im0 = axes[0].imshow(frame_gt.T, origin='lower', cmap="Blues",
                             vmin=0, vmax=1, animated=True)
        lbl = "HISTORIQUE" if not is_future else f"FUTUR t+{t-T_hist+1}"
        txt0 = axes[0].text(0.5, 1.02, lbl, transform=axes[0].transAxes,
                            ha='center', fontsize=7,
                            color='gray' if not is_future else 'black', animated=True)
        row_ims += [im0, txt0]

        # Prédictions
        for k, key in enumerate(pred_keys):
            ax = axes[k + 1]
            cmap = cmaps[k + 1]
            if is_future:
                frame = volume_to_projection(preds[key][t - T_hist])
                txt_lbl = f"PRÉDIT t+{t-T_hist+1}"
                txt_col = MODEL_COLORS.get(key, "gray")
            else:
                frame = volume_to_projection(data_hist[t])
                txt_lbl = "HISTORIQUE"
                txt_col = "gray"
            im = ax.imshow(frame.T, origin='lower', cmap=cmap,
                           vmin=0, vmax=1, animated=True)
            txt = ax.text(0.5, 1.02, txt_lbl, transform=ax.transAxes,
                          ha='center', fontsize=7, color=txt_col, animated=True)
            row_ims += [im, txt]

        ims.append(row_ims)

    titles = ["Vérité terrain"] + pred_keys
    for ax, title in zip(axes, titles):
        ax.set_title(title.replace("_", "\n"), fontsize=8, fontweight='bold', pad=18)
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("Rendu volumique MIP — Comparaison TimesFM vs Chronos", fontsize=10, y=1.05)
    fig.tight_layout()
    ani = animation.ArtistAnimation(fig, ims, interval=1000 // fps, blit=True)
    ani.save(OUTPUT_DIR / filename, writer='pillow', fps=fps, dpi=100)
    plt.close(fig)
    print(f"   [OK] GIF : {OUTPUT_DIR / filename}")


def plot_voxel_timeseries_all(
    data_hist, data_gt, preds: Dict[str, np.ndarray],
    voxel: Tuple[int, int, int],
):
    ix, iy, iz = voxel
    T_hist = data_hist.shape[0]
    T_pred = data_gt.shape[0]
    t_hist = np.arange(T_hist)
    t_pred = np.arange(T_hist, T_hist + T_pred)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axvspan(T_hist - 0.5, T_hist + T_pred - 0.5, alpha=0.08, color='gold',
               label='Zone forecast')
    ax.plot(t_hist, data_hist[:, ix, iy, iz], 'k-', lw=2.5,
            label='Historique', zorder=6)
    ax.plot(t_pred, data_gt[:, ix, iy, iz], 'k--', lw=2.5,
            label='Vérité terrain', zorder=6)

    markers = ['o', 's', '^', 'D', 'v', 'P']
    for k, (key, pred) in enumerate(preds.items()):
        color = MODEL_COLORS.get(key, f"C{k}")
        ax.plot(t_pred, pred[:, ix, iy, iz],
                markers[k % len(markers)] + '-',
                color=color, lw=2, ms=5, label=key, zorder=5)

    ax.axvline(T_hist - 0.5, color='gray', ls=':', lw=1.5)
    ax.set_xlim(0, T_hist + T_pred - 1); ax.set_ylim(-0.05, 1.2)
    ax.set_xlabel("Pas de temps (h)"); ax.set_ylabel("Fraction nuageuse")
    ax.set_title(f"Série temporelle — Voxel ({ix},{iy},{iz})",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "voxel_timeseries_all.png", dpi=150)
    plt.close(fig)
    print(f"   [OK] Voxel timeseries : {OUTPUT_DIR / 'voxel_timeseries_all.png'}")


def plot_benchmark_dashboard(metrics: Dict[str, Dict]):
    """
    Dashboard de benchmark 2×3 :
    PSNR | SSIM | Erreur masse | MAE temporelle | Temps forecast | RAM / GPU
    """
    keys = list(metrics.keys())
    colors = [MODEL_COLORS.get(k, f"C{i}") for i, k in enumerate(keys)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ── PSNR ────────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    vals = [metrics[k]["PSNR"] for k in keys]
    bars = ax.bar(keys, vals, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_title("PSNR Volumique (dB)  ↑ meilleur", fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(vals) * 1.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.2f}",
                ha='center', fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3); ax.tick_params(axis='x', rotation=30)

    # ── SSIM ─────────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    vals = [metrics[k]["SSIM_3D"] for k in keys]
    bars = ax.bar(keys, vals, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_title("SSIM 3D  ↑ meilleur", fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.4f}",
                ha='center', fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3); ax.tick_params(axis='x', rotation=30)

    # ── Conservation masse ───────────────────────────────────────────────────
    ax = axes[0, 2]
    vals = [metrics[k]["conservation_masse"] for k in keys]
    bars = ax.bar(keys, vals, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_title("Erreur conservation masse  ↓ meilleur", fontsize=11, fontweight='bold')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.02, f"{v:.4f}",
                ha='center', fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3); ax.tick_params(axis='x', rotation=30)

    # ── MAE par pas de temps ─────────────────────────────────────────────────
    ax = axes[1, 0]
    for k, color in zip(keys, colors):
        mae = metrics[k]["MAE_par_pas"]
        ax.plot(range(1, len(mae)+1), mae, 'o-', color=color,
                label=k, lw=2, ms=6)
    ax.set_title("MAE par pas de forecast  ↓ meilleur", fontsize=11, fontweight='bold')
    ax.set_xlabel("Pas de forecast (h)"); ax.set_ylabel("MAE")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Temps d'exécution ────────────────────────────────────────────────────
    ax = axes[1, 1]
    vals = [metrics[k].get("elapsed_s", 0.0) for k in keys]
    bars = ax.bar(keys, vals, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_title("Temps de forecast (s)  ↓ meilleur", fontsize=11, fontweight='bold')
    ax.set_ylabel("Secondes")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.02, f"{v:.2f}s",
                ha='center', fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3); ax.tick_params(axis='x', rotation=30)

    # ── RAM / GPU ────────────────────────────────────────────────────────────
    ax = axes[1, 2]
    ram_vals = [metrics[k].get("ram_delta_mb", 0.0) for k in keys]
    gpu_vals = [metrics[k].get("gpu_peak_mb", 0.0) for k in keys]
    x_pos = np.arange(len(keys))
    width = 0.35
    bars_ram = ax.bar(x_pos - width/2, ram_vals, width, label='ΔRAM (MB)',
                      color=colors, alpha=0.7, edgecolor='white')
    bars_gpu = ax.bar(x_pos + width/2, gpu_vals, width, label='GPU peak (MB)',
                      color=colors, alpha=0.4, edgecolor='black', hatch='//')
    ax.set_xticks(x_pos); ax.set_xticklabels(keys, rotation=30, ha='right', fontsize=9)
    ax.set_title("Ressources mémoire", fontsize=11, fontweight='bold')
    ax.set_ylabel("MB")
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    plt.suptitle("Benchmark Forecast Volumétrique Nuages ERA5\nTimesFM vs Chronos — Approches Directe & Latent",
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "benchmark_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   [OK] Dashboard : {OUTPUT_DIR / 'benchmark_dashboard.png'}")


def plot_radar_chart(metrics: Dict[str, Dict]):
    """Radar chart — comparaison synthétique normalisée."""
    # Critères (inversés si ↓ meilleur)
    criteria = ["PSNR", "SSIM_3D", "Masse\n(inv)", "Grad\n(inv)", "Vitesse\n(inv)", "CorrT"]

    def normalize(key, val_dict):
        vals = np.array([val_dict[k][key] for k in val_dict])
        mn, mx = vals.min(), vals.max()
        if mx - mn < 1e-8:
            return {k: 0.5 for k in val_dict}
        return {k: (val_dict[k][key] - mn) / (mx - mn) for k in val_dict}

    psnr_n  = normalize("PSNR", metrics)
    ssim_n  = normalize("SSIM_3D", metrics)
    mass_n  = {k: 1 - v for k, v in normalize("conservation_masse", metrics).items()}
    grad_n  = {k: 1 - v for k, v in normalize("erreur_gradient", metrics).items()}
    speed_n = {k: 1 - v for k, v in normalize("elapsed_s", metrics).items()}
    corr_n  = normalize("correlation_temporelle", metrics)

    N = len(criteria)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})

    keys = list(metrics.keys())
    for k, color in zip(keys, [MODEL_COLORS.get(kk, f"C{i}") for i, kk in enumerate(keys)]):
        vals = [
            psnr_n[k], ssim_n[k], mass_n[k],
            grad_n[k], speed_n[k], corr_n[k]
        ]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', color=color, lw=2, label=k)
        ax.fill(angles, vals, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title("Comparaison synthétique normalisée\n(plus loin = meilleur)",
                 fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "radar_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   [OK] Radar chart : {OUTPUT_DIR / 'radar_comparison.png'}")


def plot_latent_space(latents_dict: Dict[str, np.ndarray]):
    """PCA 2D de l'espace latent pour chaque forecaster."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return
    n = len(latents_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (key, latents) in zip(axes, latents_dict.items()):
        pca = PCA(n_components=2)
        z2d = pca.fit_transform(latents)
        sc = ax.scatter(z2d[:, 0], z2d[:, 1], c=np.arange(len(z2d)),
                        cmap='viridis', s=50, edgecolors='k', linewidths=0.4)
        plt.colorbar(sc, ax=ax, label='Pas de temps')
        ax.set_title(f"Espace latent — {key}", fontsize=11)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "latent_pca_all.png", dpi=150)
    plt.close(fig)
    print(f"   [OK] PCA latent : {OUTPUT_DIR / 'latent_pca_all.png'}")


def save_report(metrics: Dict, data_shape: tuple):
    path = OUTPUT_DIR / "rapport_benchmark.txt"
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RAPPORT BENCHMARK — FORECAST VOLUMÉTRIQUE MÉTÉO\n")
        f.write("TimesFM vs Chronos  |  Direct vs Espace Latent\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Résolution volumique : {NX} × {NY} × {NZ}\n")
        f.write(f"Historique           : {N_HIST} pas de temps\n")
        f.write(f"Horizon              : {N_PRED} pas de temps\n")
        f.write(f"Dimension latente    : {LATENT_DIM}\n")
        f.write(f"TimesFM disponible   : {HAS_TIMESFM}\n")
        f.write(f"Chronos disponible   : {HAS_CHRONOS}\n")
        f.write(f"Device               : {DEVICE}\n\n")

        # Tableau de comparaison
        header = f"{'Approche':<25} {'PSNR':>8} {'SSIM':>8} {'Masse':>8} {'CorrT':>8} {'Temps(s)':>10} {'RAM(MB)':>9} {'GPU(MB)':>9}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")
        for approach, vals in metrics.items():
            f.write(
                f"{approach:<25} "
                f"{vals['PSNR']:>8.3f} "
                f"{vals['SSIM_3D']:>8.5f} "
                f"{vals['conservation_masse']:>8.5f} "
                f"{vals['correlation_temporelle']:>8.4f} "
                f"{vals.get('elapsed_s', 0):>10.3f} "
                f"{vals.get('ram_delta_mb', 0):>9.1f} "
                f"{vals.get('gpu_peak_mb', 0):>9.1f}\n"
            )

        f.write("\n")
        # Meilleurs par critère
        for crit, best_fn, label in [
            ("PSNR", max, "PSNR ↑"),
            ("SSIM_3D", max, "SSIM ↑"),
            ("conservation_masse", min, "Masse ↓"),
            ("elapsed_s", min, "Vitesse ↓"),
        ]:
            try:
                best_key = best_fn(metrics, key=lambda k: metrics[k].get(crit, 0 if best_fn == max else 1e9))
                f.write(f"Meilleur {label:<15}: {best_key}  ({metrics[best_key].get(crit, 0):.4f})\n")
            except Exception:
                pass
    print(f"   [OK] Rapport : {path}")


# =============================================================================
# 8. PIPELINE PRINCIPALE
# =============================================================================

def run_pipeline(
    use_era5: bool = False,
    skip_latent: bool = False,
    chronos_model: str = "amazon/chronos-t5-small",
    n_epochs: int = N_EPOCHS,
):
    print("=" * 70)
    print("  BENCHMARK FORECAST VOLUMÉTRIQUE MÉTÉO — TimesFM vs Chronos")
    print("=" * 70)
    print(f"  Volume: {NX}×{NY}×{NZ}  |  Hist={N_HIST}h  |  Forecast={N_PRED}h")
    print(f"  Device: {DEVICE}  |  TimesFM={HAS_TIMESFM}  |  Chronos={HAS_CHRONOS}")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print("=" * 70)

    # ── Données ───────────────────────────────────────────────────────────────
    print("\n[STEP 1] Données...")
    data = None
    if use_era5:
        era5_path = OUTPUT_DIR / "era5_cloud.nc"
        if not era5_path.exists():
            download_era5(era5_path)
        if era5_path.exists():
            data = load_era5_netcdf(era5_path)

    if data is None:
        print("   → Données synthétiques ERA5-like générées.")
        data = generate_synthetic_cloud()

    print(f"   shape={data.shape}  min={data.min():.3f}  max={data.max():.3f}")
    data_hist = data[:N_HIST]
    data_gt   = data[N_HIST:N_HIST + N_PRED]

    # ── Forecasters ───────────────────────────────────────────────────────────
    print("\n[STEP 2] Initialisation des forecasters...")
    forecasters = build_forecasters(horizon=N_PRED, chronos_model=chronos_model)
    for name, f in forecasters.items():
        print(f"   {name}: {type(f).__name__}")

    # ── Autoencoder (partagé entre les deux forecasters) ──────────────────────
    autoencoder = None
    if HAS_TORCH and not skip_latent:
        print("\n[STEP 3] Entraînement Autoencoder 3D (partagé)...")
        autoencoder = train_autoencoder(data_hist, n_epochs=n_epochs)

    # ── Forecasts + métriques ─────────────────────────────────────────────────
    all_preds: Dict[str, np.ndarray] = {}
    all_metrics: Dict[str, Dict] = {}
    all_latents: Dict[str, np.ndarray] = {}

    for fname, forecaster in forecasters.items():
        # --- Approche Directe ---
        key_d = f"{fname}_Direct"
        try:
            pred_d, stats_d = forecast_direct(data_hist, forecaster, label=key_d)
            all_preds[key_d]   = pred_d
            all_metrics[key_d] = compute_metrics_for_pred(data_gt, pred_d, stats_d)
        except Exception as e:
            print(f"[WARN] {key_d} échoué : {e}")
            traceback.print_exc()

        # --- Approche Latente ---
        if autoencoder is not None:
            key_l = f"{fname}_Latent"
            try:
                pred_l, latents, stats_l = forecast_latent(
                    data_hist, autoencoder, forecaster, label=key_l
                )
                all_preds[key_l]    = pred_l
                all_metrics[key_l]  = compute_metrics_for_pred(data_gt, pred_l, stats_l)
                all_latents[key_l]  = latents
            except Exception as e:
                print(f"[WARN] {key_l} échoué : {e}")
                traceback.print_exc()

    # ── Résumé métriques console ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RÉSUMÉ DES MÉTRIQUES")
    print("=" * 70)
    print(f"{'Approche':<25} {'PSNR':>8} {'SSIM':>8} {'CorrT':>7} {'Temps(s)':>10}")
    print("-" * 65)
    for key, m in all_metrics.items():
        print(f"{key:<25} {m['PSNR']:>8.3f} {m['SSIM_3D']:>8.4f} "
              f"{m['correlation_temporelle']:>7.4f} {m.get('elapsed_s', 0):>10.2f}")

    # ── Visualisations ────────────────────────────────────────────────────────
    print("\n[STEP 4] Génération des visualisations...")

    # GIF comparatif
    if all_preds:
        make_comparison_gif(data_gt, all_preds, data_hist,
                            filename="comparaison_timesfm_vs_chronos.gif")

    # Voxel à max variance
    var_map = data.var(axis=0)
    ix, iy, iz = np.unravel_index(var_map.argmax(), var_map.shape)
    print(f"   Voxel max variance : ({ix}, {iy}, {iz})")
    if all_preds:
        plot_voxel_timeseries_all(data_hist, data_gt, all_preds, voxel=(ix, iy, iz))

    # Dashboard benchmark
    if all_metrics:
        plot_benchmark_dashboard(all_metrics)
        plot_radar_chart(all_metrics)

    # Espace latent PCA
    if all_latents:
        plot_latent_space(all_latents)

    # Rapport texte
    save_report(all_metrics, data.shape)

    # ── Résumé fichiers ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  RÉSULTATS : {OUTPUT_DIR.resolve()}")
    print("=" * 70)
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"   {f.name:<45}  {f.stat().st_size // 1024:5d} Ko")
    print("=" * 70)
    print("\n[DONE] Pipeline terminée.")


# =============================================================================
# 9. POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark forecast volumétrique météo : TimesFM vs Chronos"
    )
    parser.add_argument('--era5', action='store_true',
                        help='Télécharger ERA5 via cdsapi')
    parser.add_argument('--skip-latent', action='store_true',
                        help="Ne pas entraîner l'autoencoder (approche latente)")
    parser.add_argument('--nx', type=int, default=NX)
    parser.add_argument('--ny', type=int, default=NY)
    parser.add_argument('--nz', type=int, default=NZ)
    parser.add_argument('--hist', type=int, default=N_HIST,
                        help='Nombre de pas historiques')
    parser.add_argument('--pred', type=int, default=N_PRED,
                        help='Horizon de forecast')
    parser.add_argument('--latent-dim', type=int, default=LATENT_DIM)
    parser.add_argument('--epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--chronos-model', type=str,
                        default="amazon/chronos-t5-small",
                        choices=[
                            "amazon/chronos-t5-tiny",
                            "amazon/chronos-t5-small",
                            "amazon/chronos-t5-base",
                            "amazon/chronos-t5-large",
                        ],
                        help='Variante du modèle Chronos')
    args = parser.parse_args()

    # Override globals
    NX, NY, NZ = args.nx, args.ny, args.nz
    N_HIST, N_PRED = args.hist, args.pred
    LATENT_DIM = args.latent_dim
    N_EPOCHS = args.epochs

    run_pipeline(
        use_era5=args.era5,
        skip_latent=args.skip_latent,
        chronos_model=args.chronos_model,
        n_epochs=args.epochs,
    )