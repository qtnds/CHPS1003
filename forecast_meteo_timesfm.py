"""
=============================================================================
Forecast Volumétrique Météo avec TimesFM
=============================================================================
Deux approches :
  1/ Forecast 3D direct d'un nuage (application TimesFM voxel par voxel)
  2/ Forecast en espace latent (3D CNN Encoder → TimesFM → Decoder)

Dataset : ERA5 (Copernicus) ou données synthétiques si non disponible
Sorties  : GIF 3D comparatif + courbes voxel + métriques PSNR/SSIM/physique
           → dossier resultat_meteo/
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

warnings.filterwarnings('ignore')

# ─── Dépendances optionnelles ─────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch non disponible. Seul le mode synthétique sera possible.")

try:
    import timesfm
    # Vérifier quelle API est disponible (1.x vs 2.x)
    if hasattr(timesfm, 'TimesFM_2p5_200M_torch'):
        TIMESFM_VERSION = 2
    elif hasattr(timesfm, 'TimesFm'):
        TIMESFM_VERSION = 1
    else:
        TIMESFM_VERSION = 0
    HAS_TIMESFM = TIMESFM_VERSION > 0
    if HAS_TIMESFM:
        print(f"[OK] Module TimesFM détecté (API v{TIMESFM_VERSION}).")
    else:
        print("[WARN] Module timesfm importé mais API non reconnue → substitut utilisé.")
except ImportError:
    HAS_TIMESFM = False
    TIMESFM_VERSION = 0
    print("[WARN] TimesFM non disponible → utilisation d'un forecaster simple de substitution.")

try:
    import cdsapi
    HAS_CDS = True
except ImportError:
    HAS_CDS = False
    print("[WARN] cdsapi non disponible → données synthétiques ERA5 générées localement.")

try:
    from skimage.metrics import structural_similarity as ssim_2d
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# ─── Configuration globale ────────────────────────────────────────────────────
OUTPUT_DIR = Path("resultat_meteo")
OUTPUT_DIR.mkdir(exist_ok=True)

# Paramètres de simulation
NX, NY, NZ = 16, 16, 8        # Résolution volumique (réduite pour démo)
N_HIST     = 24                # Pas de temps historiques (heures)
N_PRED     = 6                 # Pas de temps à prédire
LATENT_DIM = 32                # Dimension de l'espace latent
BATCH_SIZE = 4
N_EPOCHS   = 30                # Epochs pour l'autoencoder
DEVICE     = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"


# =============================================================================
# 1. GÉNÉRATION / CHARGEMENT DES DONNÉES
# =============================================================================

def download_era5(out_path: Path) -> bool:
    import cdsapi

    dataset = "reanalysis-era5-pressure-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "fraction_of_cloud_cover",
            "specific_cloud_liquid_water_content",
            "specific_cloud_ice_water_content",
        ],
        # Niveaux couvrant la troposphère (où se forment les nuages)
        "pressure_level": ["500", "600", "700", "750", "800", "850", "925", "1000"],
        "year": ["2023"],
        "month": ["10"],
        "day": ["27", "28", "29"],   # 3 jours = 72h historique + forecast
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf",     # ← NetCDF pas GRIB
        "area": [51.1, 5.2, 43.3, 9.6],  # Nord-Est France / Alsace
    }

    try:
        client = cdsapi.Client()
        print("   Connexion CDS OK, téléchargement en cours (peut prendre 1-2 min)...")
        client.retrieve(dataset, request).download(str(out_path))
        print(f"   [OK] ERA5 téléchargé → {out_path}")
        return True
    except Exception as e:
        if 'Missing/incomplete configuration' in str(e):
            print("   [WARN] ~/.cdsapirc manquant → https://cds.climate.copernicus.eu/how-to-api")
        else:
            print(f"   [WARN] Téléchargement ERA5 échoué : {e}")
        return False


def load_era5_netcdf(path: Path) -> Optional[np.ndarray]:
    """Charge un fichier NetCDF ERA5 → tableau (T, X, Y, Z)."""
    try:
        import netCDF4 as nc
        ds = nc.Dataset(path)
        # Variable fraction_of_cloud_cover ou cc
        var = None
        for name in ['cc', 'fraction_of_cloud_cover', 'cldfra']:
            if name in ds.variables:
                var = ds.variables[name][:]
                break
        if var is None:
            return None
        # shape (time, level, lat, lon) → (T, lat, lon, level)
        data = np.array(var)
        if data.ndim == 4:
            data = data.transpose(0, 2, 3, 1)
        # Normalisation [0, 1]
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        # Resampling spatial vers NX × NY × NZ
        from scipy.ndimage import zoom
        T = data.shape[0]
        factors = (1, NX / data.shape[1], NY / data.shape[2], NZ / data.shape[3])
        data = zoom(data, factors, order=1)
        return data[:N_HIST + N_PRED]
    except Exception as e:
        print(f"[WARN] Lecture NetCDF échouée : {e}")
        return None


def generate_synthetic_cloud(
    nx: int = NX, ny: int = NY, nz: int = NZ,
    n_steps: int = N_HIST + N_PRED,
    seed: int = 42
) -> np.ndarray:
    """
    Génère un nuage synthétique V(x, y, z, t) animé :
    - Forme gaussienne se déplaçant + se dilatant
    - Bruit turbulent basse fréquence
    - Physique simplifiée (advection + diffusion)
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    data = np.zeros((n_steps, nx, ny, nz), dtype=np.float32)

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Paramètres du nuage
    cx0, cy0, cz0 = 0.0, -0.5, 0.4          # Centre initial
    vx, vy = 0.04, 0.02                       # Vitesse d'advection
    sigma_xy, sigma_z = 0.35, 0.25           # Taille

    # Bruit de fond turbulent (Perlin-like via fréquences multiples)
    freq = np.random.randn(3, 4, 8) * 0.1

    for t in range(n_steps):
        # Centre du nuage
        cx = cx0 + vx * t + 0.05 * np.sin(0.3 * t)
        cy = cy0 + vy * t + 0.03 * np.cos(0.2 * t)
        cz = cz0 + 0.01 * np.sin(0.15 * t)
        sig_xy = sigma_xy * (1 + 0.1 * np.sin(0.25 * t))
        sig_z  = sigma_z  * (1 + 0.05 * np.cos(0.18 * t))

        # Nuage principal (Gaussienne 3D)
        cloud = np.exp(
            -((X - cx)**2 + (Y - cy)**2) / (2 * sig_xy**2)
            -((Z - cz)**2) / (2 * sig_z**2)
        )

        # Second nuage (interaction physique)
        cx2 = -0.3 + 0.03 * t
        cy2 = 0.3 - 0.01 * t
        cloud2 = 0.4 * np.exp(
            -((X - cx2)**2 + (Y - cy2)**2) / (2 * (sigma_xy * 0.7)**2)
            -((Z - 0.6)**2) / (2 * sig_z**2)
        )

        # Turbulence basse fréquence
        turb = np.zeros((nx, ny, nz))
        for k in range(4):
            fk = (k + 1) * 2
            amp = freq[:, :, k*2] if k < 3 else np.zeros((3, 4))
            turb += 0.08 / (k + 1) * np.sin(
                fk * X[:, :, 0:1] + freq[0, k % 4, 0] * t
            ) * np.sin(
                fk * Y[:, :, 0:1] + freq[1, k % 4, 1] * t
            ) * np.cos(
                fk * Z[0:1, :, :] + freq[2, k % 4, 2] * t
            )

        frame = cloud + cloud2 + 0.15 * turb
        frame = np.clip(frame, 0, 1)
        data[t] = frame

    return data


# =============================================================================
# 2. FORECASTER DE SUBSTITUTION (si TimesFM absent)
# =============================================================================

class SimpleForecaster:
    """Régression linéaire + bruit → substitut de TimesFM."""
    def __init__(self, horizon: int = N_PRED):
        self.horizon = horizon

    def forecast(self, series: np.ndarray) -> np.ndarray:
        """series : (T,) → prédiction (horizon,)"""
        T = len(series)
        t = np.arange(T)
        coeffs = np.polyfit(t[-min(T, 12):], series[-min(T, 12):], deg=1)
        future_t = np.arange(T, T + self.horizon)
        pred = np.polyval(coeffs, future_t)
        # Amortissement vers la moyenne
        mu = series[-8:].mean()
        alpha = np.linspace(0, 0.5, self.horizon)
        pred = pred * (1 - alpha) + mu * alpha
        return np.clip(pred, 0, 1).astype(np.float32)


def build_timesfm_forecaster():
    """Initialise TimesFM (v1 ou v2) ou un substitut linéaire."""
    if not HAS_TIMESFM:
        return SimpleForecaster(horizon=N_PRED)

    try:
        import torch as _torch
        _torch.set_float32_matmul_precision("high")

        if TIMESFM_VERSION == 2:
            # ── TimesFM 2.x API (comme dans le code OPF de référence) ──────
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                "google/timesfm-2.5-200m-pytorch"
            )
            model.compile(
                timesfm.ForecastConfig(
                    max_context=512,
                    max_horizon=N_PRED,
                    normalize_inputs=True,
                    use_continuous_quantile_head=False,
                    infer_is_positive=True,
                )
            )
            print("[OK] TimesFM 2.x chargé et compilé.")
            return model

        else:
            # ── TimesFM 1.x API ─────────────────────────────────────────────
            tfm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu" if _torch.cuda.is_available() else "cpu",
                    per_core_batch_size=32,
                    horizon_len=N_PRED,
                    num_layers=20,
                    model_dims=1280,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
                ),
            )
            print("[OK] TimesFM 1.x chargé avec succès.")
            return tfm

    except Exception as e:
        print(f"[WARN] Impossible de charger TimesFM : {e}\n      → Substitut utilisé.")
    return SimpleForecaster(horizon=N_PRED)


def forecast_series(model, series: np.ndarray) -> np.ndarray:
    """Applique le modèle de forecast sur une série 1D → prédiction (N_PRED,)."""
    if not HAS_TIMESFM:
        return model.forecast(series)

    if TIMESFM_VERSION == 2:
        # TimesFM 2.x : model.forecast(horizon=..., inputs=[...])
        # Retourne (point_preds, quantile_preds)
        points, _ = model.forecast(
            horizon=N_PRED,
            inputs=[series.astype(np.float32).tolist()],
        )
        return np.array(points[0], dtype=np.float32)

    elif TIMESFM_VERSION == 1:
        # TimesFM 1.x : model.forecast([series], freq=[0])
        _, preds = model.forecast(
            [series.tolist()],
            freq=[0],
        )
        return np.array(preds[0], dtype=np.float32)

    else:
        return model.forecast(series)


# =============================================================================
# 3. APPROCHE 1 : FORECAST 3D DIRECT (TimesFM voxel par voxel)
# =============================================================================

def forecast_direct(
    data_hist: np.ndarray,  # (T_hist, NX, NY, NZ)
    forecaster,
) -> np.ndarray:
    """
    Forecast direct : applique TimesFM sur chaque voxel indépendamment.
    Entrée  : historique volumique (T_hist, X, Y, Z)
    Sortie  : prédiction (T_pred, X, Y, Z)
    """
    print("\n[1/2] Forecast direct voxel par voxel...")
    T_hist, nx, ny, nz = data_hist.shape
    pred = np.zeros((N_PRED, nx, ny, nz), dtype=np.float32)

    total = nx * ny * nz
    done = 0
    t0 = time.time()

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                series = data_hist[:, ix, iy, iz]
                pred[:, ix, iy, iz] = forecast_series(forecaster, series)
                done += 1
                if done % (total // 10 + 1) == 0:
                    pct = 100 * done / total
                    elapsed = time.time() - t0
                    print(f"   {pct:.0f}% ({done}/{total}) — {elapsed:.1f}s")

    print(f"   Forecast direct terminé en {time.time()-t0:.1f}s")
    return pred


# =============================================================================
# 4. APPROCHE 2 : FORECAST EN ESPACE LATENT (Encoder → TimesFM → Decoder)
# =============================================================================

if HAS_TORCH:
    class Encoder3D(nn.Module):
        """CNN 3D : volume (1, X, Y, Z) → vecteur latent (LATENT_DIM,)"""
        def __init__(self, latent_dim: int = LATENT_DIM):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(1, 16, kernel_size=3, padding=1), nn.LeakyReLU(0.2),
                nn.MaxPool3d(2),                             # /2
                nn.Conv3d(16, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.2),
                nn.MaxPool3d(2),                             # /4
                nn.Conv3d(32, 64, kernel_size=3, padding=1), nn.LeakyReLU(0.2),
                nn.AdaptiveAvgPool3d(2),                     # → (64, 2, 2, 2)
            )
            self.fc = nn.Linear(64 * 8, latent_dim)

        def forward(self, x):
            # x : (B, 1, X, Y, Z)
            h = self.net(x)
            h = h.view(h.size(0), -1)
            return self.fc(h)

    class Decoder3D(nn.Module):
        """Vecteur latent (LATENT_DIM,) → volume (1, X, Y, Z)"""
        def __init__(self, latent_dim: int = LATENT_DIM,
                     nx: int = NX, ny: int = NY, nz: int = NZ):
            super().__init__()
            self.nx, self.ny, self.nz = nx, ny, nz
            self.fc = nn.Linear(latent_dim, 64 * 2 * 2 * 2)
            self.net = nn.Sequential(
                nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1), nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                nn.ConvTranspose3d(16, 1, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )

        def forward(self, z):
            h = self.fc(z).view(-1, 64, 2, 2, 2)
            out = self.net(h)
            return F.interpolate(out, size=(self.nx, self.ny, self.nz),
                                 mode='trilinear', align_corners=False)

    class Autoencoder3D(nn.Module):
        def __init__(self, latent_dim: int = LATENT_DIM):
            super().__init__()
            self.encoder = Encoder3D(latent_dim)
            self.decoder = Decoder3D(latent_dim)

        def forward(self, x):
            return self.decoder(self.encoder(x))

        def encode(self, x):
            return self.encoder(x)

        def decode(self, z):
            return self.decoder(z)


def train_autoencoder(
    data: np.ndarray,  # (T_total, NX, NY, NZ)
    n_epochs: int = N_EPOCHS,
) -> "Autoencoder3D":
    """Entraîne l'autoencoder 3D sur les données historiques."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch requis pour l'approche espace latent.")

    print("\n[Autoencoder] Entraînement...")
    model = Autoencoder3D(LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Dataset : chaque frame volumique est un exemple d'entraînement
    T = data.shape[0]
    X_tensor = torch.tensor(data[:, np.newaxis, ...], dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        model.train()
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            loss = F.mse_loss(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg = epoch_loss / len(loader)
        losses.append(avg)
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1:3d}/{n_epochs}  Loss = {avg:.6f}")

    # Sauvegarde courbe de loss
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(losses, color='steelblue', lw=2)
    ax.set_title("Courbe de perte — Autoencoder 3D")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "autoencoder_loss.png", dpi=120)
    plt.close(fig)
    print("[OK] Autoencoder entraîné.")
    return model


def forecast_latent(
    data_hist: np.ndarray,  # (T_hist, NX, NY, NZ)
    autoencoder: "Autoencoder3D",
    forecaster,
) -> np.ndarray:
    """
    Forecast en espace latent :
      1. Encode chaque frame historique → (T_hist, LATENT_DIM)
      2. Pour chaque dimension latente, applique TimesFM → (T_pred, LATENT_DIM)
      3. Décode chaque vecteur latent prédit → (T_pred, NX, NY, NZ)
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch requis pour l'approche espace latent.")

    print("\n[2/2] Forecast en espace latent...")
    autoencoder.eval()

    # 1. Encodage
    with torch.no_grad():
        frames = torch.tensor(data_hist[:, np.newaxis, ...], dtype=torch.float32).to(DEVICE)
        latents = autoencoder.encode(frames).cpu().numpy()  # (T_hist, LATENT_DIM)

    print(f"   Espace latent : {latents.shape}  (T_hist × LATENT_DIM)")

    # 2. Forecast dans l'espace latent (une série par dimension)
    latent_pred = np.zeros((N_PRED, LATENT_DIM), dtype=np.float32)
    for d in range(LATENT_DIM):
        latent_pred[:, d] = forecast_series(forecaster, latents[:, d])

    # 3. Décodage
    with torch.no_grad():
        z_tensor = torch.tensor(latent_pred, dtype=torch.float32).to(DEVICE)
        volumes = autoencoder.decode(z_tensor).cpu().numpy()  # (T_pred, 1, NX, NY, NZ)

    pred = volumes[:, 0, ...]  # (T_pred, NX, NY, NZ)
    pred = np.clip(pred, 0, 1)
    print(f"   Forecast latent terminé.")
    return pred, latents


# =============================================================================
# 5. MÉTRIQUES
# =============================================================================

def psnr_volume(gt: np.ndarray, pred: np.ndarray) -> float:
    """PSNR volumique moyen sur T frames."""
    mse = np.mean((gt - pred) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def ssim_3d_approx(gt: np.ndarray, pred: np.ndarray) -> float:
    """SSIM 3D approximé : moyenne des SSIM sur les coupes axiales."""
    if not HAS_SKIMAGE:
        # Fallback manuel
        mu1, mu2 = gt.mean(), pred.mean()
        s1, s2 = gt.std(), pred.std()
        cov = np.mean((gt - mu1) * (pred - mu2))
        c1, c2 = 0.01**2, 0.03**2
        return float((2*mu1*mu2 + c1) * (2*cov + c2) /
                     ((mu1**2 + mu2**2 + c1) * (s1**2 + s2**2 + c2)))
    scores = []
    for t in range(gt.shape[0]):
        for iz in range(gt.shape[3]):
            s = ssim_2d(gt[t, :, :, iz], pred[t, :, :, iz],
                        data_range=1.0)
            scores.append(s)
    return float(np.mean(scores))


def physical_error(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """
    Erreurs physiques :
    - Conservation de masse (∫ρ dV) relative
    - Gradient spatial moyen (continuité)
    - Dissipation : variance de la différence
    """
    mass_gt   = gt.mean(axis=(1, 2, 3))
    mass_pred = pred.mean(axis=(1, 2, 3))
    mass_err  = float(np.mean(np.abs(mass_gt - mass_pred) / (mass_gt + 1e-8)))

    grad_gt   = np.gradient(gt, axis=1)
    grad_pred = np.gradient(pred, axis=1)
    grad_err  = float(np.mean(np.abs(grad_gt - grad_pred)))

    dissip = float(np.var(gt - pred))

    return {
        "conservation_masse": mass_err,
        "erreur_gradient": grad_err,
        "dissipation": dissip,
    }


def temporal_error(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """
    Erreurs temporelles :
    - MAE par pas de temps
    - Correlation temporelle d'un voxel représentatif
    """
    mae_per_t = np.mean(np.abs(gt - pred), axis=(1, 2, 3))
    ix, iy, iz = NX//2, NY//2, NZ//2
    corr = float(np.corrcoef(
        gt[:, ix, iy, iz], pred[:, ix, iy, iz]
    )[0, 1])
    return {
        "MAE_par_pas": mae_per_t.tolist(),
        "correlation_temporelle": corr,
    }


def compute_all_metrics(
    gt: np.ndarray, pred1: np.ndarray, pred2: Optional[np.ndarray]
) -> Dict:
    metrics = {
        "approche1_directe": {
            "PSNR": psnr_volume(gt, pred1),
            "SSIM_3D": ssim_3d_approx(gt, pred1),
            **physical_error(gt, pred1),
            **temporal_error(gt, pred1),
        }
    }
    if pred2 is not None:
        metrics["approche2_latent"] = {
            "PSNR": psnr_volume(gt, pred2),
            "SSIM_3D": ssim_3d_approx(gt, pred2),
            **physical_error(gt, pred2),
            **temporal_error(gt, pred2),
        }
    return metrics


# =============================================================================
# 6. VISUALISATION
# =============================================================================

def volume_to_projection(vol: np.ndarray) -> np.ndarray:
    """Projection MIP (Maximum Intensity Projection) sur l'axe Z."""
    return vol.max(axis=-1)  # (NX, NY)


def make_gif_3d(
    data_gt: np.ndarray,    # (T, NX, NY, NZ)
    pred1: np.ndarray,      # (T_pred, NX, NY, NZ)
    pred2: Optional[np.ndarray],
    data_hist: np.ndarray,  # (T_hist, NX, NY, NZ)
    filename: str,
    fps: int = 4,
):
    """
    Crée un GIF animé comparant :
    - Vérité terrain (historique + futur)
    - Prédiction directe (approche 1)
    - Prédiction latente (approche 2)
    """
    T_hist = data_hist.shape[0]
    T_pred = pred1.shape[0]

    n_cols = 3 if pred2 is not None else 2
    titles = ["Vérité terrain", "Approche 1 (Direct)", "Approche 2 (Latent)"][:n_cols]

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    # Combine hist + gt pour les T_pred derniers steps
    all_gt = np.concatenate([data_hist, data_gt], axis=0)

    ims = []
    for t in range(T_hist + T_pred):
        row_ims = []

        # GT
        frame_gt = volume_to_projection(all_gt[t])
        im0 = axes[0].imshow(frame_gt.T, origin='lower', cmap='Blues',
                              vmin=0, vmax=1, animated=True)
        label = "HISTORIQUE" if t < T_hist else f"FUTUR t+{t-T_hist+1}"
        txt0 = axes[0].text(0.5, 1.02, label, transform=axes[0].transAxes,
                             ha='center', fontsize=8,
                             color='gray' if t < T_hist else 'black', animated=True)
        row_ims += [im0, txt0]

        # Prédiction 1
        if t < T_hist:
            frame_p1 = volume_to_projection(data_hist[t])
            txt1_label = "HISTORIQUE"
        else:
            frame_p1 = volume_to_projection(pred1[t - T_hist])
            txt1_label = f"PRÉDIT t+{t-T_hist+1}"
        im1 = axes[1].imshow(frame_p1.T, origin='lower', cmap='Oranges',
                              vmin=0, vmax=1, animated=True)
        txt1 = axes[1].text(0.5, 1.02, txt1_label, transform=axes[1].transAxes,
                             ha='center', fontsize=8,
                             color='gray' if t < T_hist else 'darkorange', animated=True)
        row_ims += [im1, txt1]

        # Prédiction 2
        if pred2 is not None:
            if t < T_hist:
                frame_p2 = volume_to_projection(data_hist[t])
                txt2_label = "HISTORIQUE"
            else:
                frame_p2 = volume_to_projection(pred2[t - T_hist])
                txt2_label = f"PRÉDIT t+{t-T_hist+1}"
            im2 = axes[2].imshow(frame_p2.T, origin='lower', cmap='Greens',
                                  vmin=0, vmax=1, animated=True)
            txt2 = axes[2].text(0.5, 1.02, txt2_label, transform=axes[2].transAxes,
                                 ha='center', fontsize=8,
                                 color='gray' if t < T_hist else 'darkgreen', animated=True)
            row_ims += [im2, txt2]

        ims.append(row_ims)

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=9, fontweight='bold', pad=18)
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("Rendu volumique (MIP) — ERA5 Nuages", fontsize=11, y=1.05)
    fig.tight_layout()

    ani = animation.ArtistAnimation(fig, ims, interval=1000 // fps, blit=True)
    ani.save(OUTPUT_DIR / filename, writer='pillow', fps=fps, dpi=120)
    plt.close(fig)
    print(f"   [OK] GIF sauvegardé : {OUTPUT_DIR / filename}")


def plot_voxel_timeseries(
    data_hist: np.ndarray,   # (T_hist, NX, NY, NZ)
    data_gt: np.ndarray,     # (T_pred, NX, NY, NZ)
    pred1: np.ndarray,       # (T_pred, NX, NY, NZ)
    pred2: Optional[np.ndarray],
    voxel: Tuple[int, int, int] = (NX//2, NY//2, NZ//2),
):
    """Trace les séries temporelles pour un voxel représentatif."""
    ix, iy, iz = voxel
    T_hist = data_hist.shape[0]
    T_pred = data_gt.shape[0]
    T_total = T_hist + T_pred

    t_hist = np.arange(T_hist)
    t_pred = np.arange(T_hist, T_total)

    hist_series = data_hist[:, ix, iy, iz]
    gt_series   = data_gt[:, ix, iy, iz]
    p1_series   = pred1[:, ix, iy, iz]

    fig, ax = plt.subplots(figsize=(11, 4))

    # Zone forecast
    ax.axvspan(T_hist - 0.5, T_total - 0.5, alpha=0.08, color='gold', label='Zone forecast')

    ax.plot(t_hist, hist_series, 'k-', lw=2, label='Historique', zorder=5)
    ax.plot(t_pred, gt_series, 'k--', lw=2, label='Vérité terrain', zorder=5)
    ax.plot(t_pred, p1_series, 'o-', color='darkorange', lw=2, ms=5,
            label='Approche 1 — Direct (TimesFM)')

    if pred2 is not None:
        p2_series = pred2[:, ix, iy, iz]
        ax.plot(t_pred, p2_series, 's-', color='forestgreen', lw=2, ms=5,
                label='Approche 2 — Latent (3D CNN + TimesFM)')

    ax.axvline(T_hist - 0.5, color='gray', ls=':', lw=1.5, label='Coupure hist./forecast')
    ax.set_xlim(0, T_total - 1)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("Pas de temps (heures)", fontsize=11)
    ax.set_ylabel("Fraction nuageuse", fontsize=11)
    ax.set_title(f"Série temporelle — Voxel ({ix}, {iy}, {iz})", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "voxel_timeseries.png", dpi=150)
    plt.close(fig)
    print(f"   [OK] Courbe voxel sauvegardée : {OUTPUT_DIR / 'voxel_timeseries.png'}")


def plot_metrics(metrics: Dict):
    """Visualise les métriques comparatives."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    approaches = list(metrics.keys())
    colors = ['#E07B39', '#3DA35D']

    # PSNR
    psnr_vals = [metrics[a]['PSNR'] for a in approaches]
    axes[0].bar(approaches, psnr_vals, color=colors[:len(approaches)])
    axes[0].set_title("PSNR Volumique (dB)\n(↑ meilleur)", fontsize=10)
    axes[0].set_ylim(0, max(psnr_vals) * 1.3)
    for i, v in enumerate(psnr_vals):
        axes[0].text(i, v + 0.5, f"{v:.2f}", ha='center', fontweight='bold')

    # SSIM
    ssim_vals = [metrics[a]['SSIM_3D'] for a in approaches]
    axes[1].bar(approaches, ssim_vals, color=colors[:len(approaches)])
    axes[1].set_title("SSIM 3D\n(↑ meilleur)", fontsize=10)
    axes[1].set_ylim(0, 1.15)
    for i, v in enumerate(ssim_vals):
        axes[1].text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')

    # Conservation de masse
    mass_vals = [metrics[a]['conservation_masse'] for a in approaches]
    axes[2].bar(approaches, mass_vals, color=colors[:len(approaches)])
    axes[2].set_title("Erreur physique\n(conservation masse ↓)", fontsize=10)
    for i, v in enumerate(mass_vals):
        axes[2].text(i, v + 0.001, f"{v:.4f}", ha='center', fontweight='bold')

    # MAE temporelle
    for k, a in enumerate(approaches):
        mae = metrics[a]['MAE_par_pas']
        axes[3].plot(range(1, len(mae)+1), mae,
                     'o-', color=colors[k], label=a.split("_")[1], lw=2, ms=6)
    axes[3].set_title("MAE par pas de forecast\n(↓ meilleur)", fontsize=10)
    axes[3].set_xlabel("Pas de forecast (h)")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    for ax in axes[:3]:
        ax.set_ylabel("")
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle("Comparaison des métriques — Forecast Volumique Nuages ERA5",
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "metriques_comparatives.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   [OK] Métriques sauvegardées : {OUTPUT_DIR / 'metriques_comparatives.png'}")


def plot_latent_space(latents: np.ndarray):
    """Visualise l'espace latent (PCA 2D)."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return
    pca = PCA(n_components=2)
    z2d = pca.fit_transform(latents)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(z2d[:, 0], z2d[:, 1], c=np.arange(len(z2d)),
                    cmap='viridis', s=40, edgecolors='k', linewidths=0.4)
    plt.colorbar(sc, ax=ax, label='Pas de temps')
    ax.set_title("Trajectoire dans l'espace latent (PCA 2D)", fontsize=11)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "latent_space_pca.png", dpi=150)
    plt.close(fig)
    print(f"   [OK] Espace latent PCA : {OUTPUT_DIR / 'latent_space_pca.png'}")


def save_metrics_report(metrics: Dict, data_shape: tuple):
    """Sauvegarde un rapport texte des métriques."""
    report_path = OUTPUT_DIR / "rapport_metriques.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RAPPORT — FORECAST VOLUMIQUE MÉTÉO (ERA5 Nuages)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Résolution volumique : {NX} × {NY} × {NZ}\n")
        f.write(f"Historique utilisé   : {N_HIST} pas de temps\n")
        f.write(f"Horizon de forecast  : {N_PRED} pas de temps\n")
        f.write(f"Dimension latente    : {LATENT_DIM}\n")
        f.write(f"TimesFM disponible   : {'Oui' if HAS_TIMESFM else 'Non (substitut linéaire)'}\n")
        f.write(f"PyTorch disponible   : {'Oui' if HAS_TORCH else 'Non'}\n\n")

        for approach, vals in metrics.items():
            f.write("-" * 40 + "\n")
            f.write(f"Approche : {approach}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  PSNR volumique       : {vals['PSNR']:.4f} dB\n")
            f.write(f"  SSIM 3D              : {vals['SSIM_3D']:.6f}\n")
            f.write(f"  Conservation masse   : {vals['conservation_masse']:.6f}\n")
            f.write(f"  Erreur gradient      : {vals['erreur_gradient']:.6f}\n")
            f.write(f"  Dissipation          : {vals['dissipation']:.6f}\n")
            f.write(f"  Corr. temporelle     : {vals['correlation_temporelle']:.4f}\n")
            mae = vals['MAE_par_pas']
            f.write(f"  MAE par pas          : {[f'{v:.4f}' for v in mae]}\n\n")

    print(f"   [OK] Rapport : {report_path}")


# =============================================================================
# 7. PIPELINE PRINCIPALE
# =============================================================================

def run_pipeline(use_era5: bool = False, skip_latent: bool = False):
    print("=" * 65)
    print("  FORECAST VOLUMÉTRIQUE MÉTÉO — TimesFM + 3D CNN")
    print("=" * 65)
    print(f"  Résolution : {NX}×{NY}×{NZ}  |  Hist={N_HIST}h  |  Forecast={N_PRED}h")
    print(f"  Device     : {DEVICE}")
    print(f"  Output     : {OUTPUT_DIR.resolve()}")
    print("=" * 65)

    # ── 1. Données ──────────────────────────────────────────────────────────
    print("\n[STEP 1] Chargement des données...")
    data = None

    if use_era5:
        era5_path = OUTPUT_DIR / "era5_cloud.nc"
        if not era5_path.exists():
            download_era5(era5_path)
        if era5_path.exists():
            data = load_era5_netcdf(era5_path)

    if data is None:
        print("   → Données synthétiques ERA5-like générées.")
        data = generate_synthetic_cloud()  # (T_total, NX, NY, NZ)

    T_total = data.shape[0]
    print(f"   Données : shape={data.shape}  min={data.min():.3f}  max={data.max():.3f}")

    data_hist = data[:N_HIST]
    data_gt   = data[N_HIST:N_HIST + N_PRED]

    # ── 2. Modèle de forecast ───────────────────────────────────────────────
    print("\n[STEP 2] Initialisation du forecaster...")
    forecaster = build_timesfm_forecaster()

    # ── 3. Approche 1 : Direct ──────────────────────────────────────────────
    print("\n[STEP 3] Approche 1 — Forecast direct...")
    pred1 = forecast_direct(data_hist, forecaster)

    # ── 4. Approche 2 : Espace latent ───────────────────────────────────────
    pred2 = None
    latents = None

    if HAS_TORCH and not skip_latent:
        print("\n[STEP 4] Approche 2 — Espace latent...")
        autoencoder = train_autoencoder(data_hist)
        pred2, latents = forecast_latent(data_hist, autoencoder, forecaster)
    else:
        if not HAS_TORCH:
            print("\n[STEP 4] Approche 2 ignorée (PyTorch non disponible).")
        else:
            print("\n[STEP 4] Approche 2 ignorée (--skip-latent).")

    # ── 5. Métriques ─────────────────────────────────────────────────────────
    print("\n[STEP 5] Calcul des métriques...")
    metrics = compute_all_metrics(data_gt, pred1, pred2)
    for approach, vals in metrics.items():
        print(f"   {approach}:")
        print(f"     PSNR   = {vals['PSNR']:.2f} dB")
        print(f"     SSIM3D = {vals['SSIM_3D']:.4f}")
        print(f"     Masse  = {vals['conservation_masse']:.6f}")
        print(f"     CorrT  = {vals['correlation_temporelle']:.4f}")

    # ── 6. Visualisations ───────────────────────────────────────────────────
    print("\n[STEP 6] Génération des visualisations...")

    # GIF comparatif
    print("   Génération du GIF 3D comparatif...")
    make_gif_3d(data_gt, pred1, pred2, data_hist,
                filename="comparaison_3d.gif", fps=3)

    # Courbe voxel (voxel le plus intéressant = max variance)
    var_map = data.var(axis=0)
    ix, iy, iz = np.unravel_index(var_map.argmax(), var_map.shape)
    print(f"   Voxel sélectionné (max variance) : ({ix}, {iy}, {iz})")
    plot_voxel_timeseries(data_hist, data_gt, pred1, pred2, voxel=(ix, iy, iz))

    # Métriques comparatives
    plot_metrics(metrics)

    # Espace latent PCA
    if latents is not None:
        plot_latent_space(latents)

    # Rapport texte
    save_metrics_report(metrics, data.shape)

    # ── 7. Résumé final ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RÉSULTATS SAUVEGARDÉS DANS :", OUTPUT_DIR.resolve())
    print("=" * 65)
    files = sorted(OUTPUT_DIR.iterdir())
    for f in files:
        print(f"   {f.name:40s}  {f.stat().st_size // 1024:5d} Ko")
    print("=" * 65)
    print("\n[DONE] Pipeline terminée avec succès.")


# =============================================================================
# 8. POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forecast volumétrique météo avec TimesFM"
    )
    parser.add_argument('--era5', action='store_true',
                        help='Télécharger les données ERA5 réelles via cdsapi')
    parser.add_argument('--skip-latent', action='store_true',
                        help="Ne pas exécuter l'approche espace latent (3D CNN)")
    parser.add_argument('--nx', type=int, default=NX, help='Résolution X')
    parser.add_argument('--ny', type=int, default=NY, help='Résolution Y')
    parser.add_argument('--nz', type=int, default=NZ, help='Résolution Z')
    parser.add_argument('--hist', type=int, default=N_HIST, help='Historique (h)')
    parser.add_argument('--pred', type=int, default=N_PRED, help='Horizon forecast (h)')
    parser.add_argument('--latent-dim', type=int, default=LATENT_DIM)
    parser.add_argument('--epochs', type=int, default=N_EPOCHS)
    args = parser.parse_args()

    # Override globals si précisé
    NX, NY, NZ = args.nx, args.ny, args.nz
    N_HIST, N_PRED = args.hist, args.pred
    LATENT_DIM = args.latent_dim
    N_EPOCHS = args.epochs

    run_pipeline(use_era5=args.era5, skip_latent=args.skip_latent)