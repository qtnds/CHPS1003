import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import timesfm
import os
import argparse

# ============================================================
# 0. DOSSIER RESULTS
# ============================================================

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# PROPRIÉTÉS PHYSIQUES DES MATÉRIAUX
# ============================================================

MATERIALS = {
    "steel": {
        "name": "Acier (Steel)",
        "thermal_conductivity": 50.0,  # W/(m·K)
        "density": 7850.0,              # kg/m³
        "specific_heat": 490.0,         # J/(kg·K)
        "thermal_diffusivity": None     # Sera calculé
    },
    "copper": {
        "name": "Cuivre (Copper)",
        "thermal_conductivity": 401.0,
        "density": 8960.0,
        "specific_heat": 385.0,
        "thermal_diffusivity": None
    },
    "aluminum": {
        "name": "Aluminium (Aluminum)",
        "thermal_conductivity": 237.0,
        "density": 2700.0,
        "specific_heat": 897.0,
        "thermal_diffusivity": None
    }
}

def calculate_thermal_diffusivity(material_dict):
    """
    Calcule la diffusivité thermique α = k / (ρ * c_p)
    
    k: conductivité thermique [W/(m·K)]
    ρ: masse volumique [kg/m³]
    c_p: capacité thermique spécifique [J/(kg·K)]
    α: diffusivité thermique [m²/s]
    """
    k = material_dict["thermal_conductivity"]
    rho = material_dict["density"]
    cp = material_dict["specific_heat"]
    
    alpha = k / (rho * cp)
    material_dict["thermal_diffusivity"] = alpha
    return alpha

# Calculer les diffusivités pour tous les matériaux
for material in MATERIALS.values():
    calculate_thermal_diffusivity(material)

# ============================================================
# 1. SIMULATION PHYSIQUE (DIFFUSION DE LA CHALEUR 3D)
# ============================================================

def simulate_heat_diffusion_3d(nx=32, ny=32, nz=32, nt=200, 
                              dx=0.01, dy=0.01, dz=0.01, dt=0.1, 
                              material="steel", disruption=None):
    """
    Simule la diffusion de chaleur en 3D avec une source à 100°C sur la surface droite.
    Utilise des conditions aux limites de Neumann (flux nul) sur les bords,
    sauf sur la surface droite où on impose T = 100°C (Dirichlet).
    
    nx, ny, nz: dimensions spatiales (nombre de points en x, y et z)
    nt: nombre de pas de temps
    dx, dy, dz: pas d'espace [m]
    dt: pas de temps [s]
    material: type de matériau ("steel", "copper", "aluminum")
    disruption: dict avec clés 'x', 'y', 'z', 'temp', 'instant' pour ajouter une perturbation
    """
    # Récupérer la diffusivité thermique du matériau
    if material not in MATERIALS:
        raise ValueError(f"Matériau '{material}' inconnu. Choix: {list(MATERIALS.keys())}")
    
    alpha = MATERIALS[material]["thermal_diffusivity"]
    material_name = MATERIALS[material]["name"]
    
    print(f"\nMatériau: {material_name}")
    print(f"Diffusivité thermique α = {alpha:.2e} m²/s")
    print(f"Dimensions: {nx*dx:.3f}m × {ny*dy:.3f}m × {nz*dz:.3f}m")
    print(f"Pas de temps dt = {dt:.3f}s")
    
    # Vérification de la stabilité (critère CFL pour méthode explicite en 3D)
    dt_max = 1.0 / (2 * alpha * (1/dx**2 + 1/dy**2 + 1/dz**2))
    if dt > dt_max:
        print(f"⚠️  ATTENTION: dt={dt:.3e}s > dt_max={dt_max:.3e}s")
        print(f"   Le schéma peut être instable! Considérez réduire dt.")
    else:
        print(f"✓ Stabilité: dt={dt:.3e}s < dt_max={dt_max:.3e}s")
    
    T = np.zeros((nt, nx, ny, nz), dtype=np.float32)
    
    # Condition initiale: température 0°C partout
    # Source de chaleur: surface droite (x=nx-1) à 100°C
    T[0, -1, :, :] = 100.0
    
    for t in range(nt - 1):
        T_current = T[t].copy()
        laplacian = np.zeros_like(T_current)
        
        # Appliquer la perturbation si on atteint l'instant défini
        if disruption is not None and t == disruption['instant']:
            T_current[disruption['x'], disruption['y'], disruption['z']] = disruption['temp']
            print(f"\n⚡ Perturbation appliquée à t={t}:")
            print(f"   Position ({disruption['x']}, {disruption['y']}, {disruption['z']})")
            print(f"   Température imposée: {disruption['temp']}°C")
        
        # Intérieur du domaine
        laplacian[1:-1, 1:-1, 1:-1] = (
            (T_current[2:, 1:-1, 1:-1] - 2*T_current[1:-1, 1:-1, 1:-1] + T_current[:-2, 1:-1, 1:-1]) / (dx**2) +
            (T_current[1:-1, 2:, 1:-1] - 2*T_current[1:-1, 1:-1, 1:-1] + T_current[1:-1, :-2, 1:-1]) / (dy**2) +
            (T_current[1:-1, 1:-1, 2:] - 2*T_current[1:-1, 1:-1, 1:-1] + T_current[1:-1, 1:-1, :-2]) / (dz**2)
        )
        
        # Faces (6 faces du cube)
        # Face gauche (x=0)
        laplacian[0, 1:-1, 1:-1] = (
            (T_current[1, 1:-1, 1:-1] - T_current[0, 1:-1, 1:-1]) / (dx**2) +
            (T_current[0, 2:, 1:-1] - 2*T_current[0, 1:-1, 1:-1] + T_current[0, :-2, 1:-1]) / (dy**2) +
            (T_current[0, 1:-1, 2:] - 2*T_current[0, 1:-1, 1:-1] + T_current[0, 1:-1, :-2]) / (dz**2)
        )
        
        # Face avant (y=0)
        laplacian[1:-1, 0, 1:-1] = (
            (T_current[2:, 0, 1:-1] - 2*T_current[1:-1, 0, 1:-1] + T_current[:-2, 0, 1:-1]) / (dx**2) +
            (T_current[1:-1, 1, 1:-1] - T_current[1:-1, 0, 1:-1]) / (dy**2) +
            (T_current[1:-1, 0, 2:] - 2*T_current[1:-1, 0, 1:-1] + T_current[1:-1, 0, :-2]) / (dz**2)
        )
        
        # Face arrière (y=ny-1)
        laplacian[1:-1, -1, 1:-1] = (
            (T_current[2:, -1, 1:-1] - 2*T_current[1:-1, -1, 1:-1] + T_current[:-2, -1, 1:-1]) / (dx**2) +
            (T_current[1:-1, -1, 1:-1] - T_current[1:-1, -2, 1:-1]) / (dy**2) +
            (T_current[1:-1, -1, 2:] - 2*T_current[1:-1, -1, 1:-1] + T_current[1:-1, -1, :-2]) / (dz**2)
        )
        
        # Face bas (z=0)
        laplacian[1:-1, 1:-1, 0] = (
            (T_current[2:, 1:-1, 0] - 2*T_current[1:-1, 1:-1, 0] + T_current[:-2, 1:-1, 0]) / (dx**2) +
            (T_current[1:-1, 2:, 0] - 2*T_current[1:-1, 1:-1, 0] + T_current[1:-1, :-2, 0]) / (dy**2) +
            (T_current[1:-1, 1:-1, 1] - T_current[1:-1, 1:-1, 0]) / (dz**2)
        )
        
        # Face haut (z=nz-1)
        laplacian[1:-1, 1:-1, -1] = (
            (T_current[2:, 1:-1, -1] - 2*T_current[1:-1, 1:-1, -1] + T_current[:-2, 1:-1, -1]) / (dx**2) +
            (T_current[1:-1, 2:, -1] - 2*T_current[1:-1, 1:-1, -1] + T_current[1:-1, :-2, -1]) / (dy**2) +
            (T_current[1:-1, 1:-1, -1] - T_current[1:-1, 1:-1, -2]) / (dz**2)
        )
        
        # Mise à jour temporelle
        T[t+1] = T[t] + alpha * dt * laplacian
        
        # Imposer la condition de Dirichlet sur la surface droite
        T[t+1, -1, :, :] = 100.0
    
    return T


# ============================================================
# 2. VOLUME <-> SERIES TEMPORELLES
# ============================================================

def volume_to_timeseries(volume):
    """Convertit (T, X, Y, Z) -> (N, T) où N = X*Y*Z séries temporelles"""
    T, X, Y, Z = volume.shape
    return volume.reshape(T, X * Y * Z).T


def timeseries_to_volume(timeseries, spatial_shape):
    """Convertit (N, T) -> (T, X, Y, Z)"""
    N, T = timeseries.shape
    X, Y, Z = spatial_shape
    return timeseries.T.reshape(T, X, Y, Z)


# ============================================================
# 3. FORECASTING ZERO-SHOT AVEC TIMESFM (TORCH)
# ============================================================

class TimesFMForecaster:
    def __init__(self, max_context=1024, max_horizon=256):
        torch.set_float32_matmul_precision("high")
        
        print("\nChargement du modèle TimesFM...")
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
        print("✓ Modèle chargé et compilé")
    
    def forecast(self, timeseries, horizon, batch_size=64):
        """
        Prédit les 'horizon' prochains pas de temps pour chaque série.
        
        timeseries: array (N, T_context) - historique
        horizon: nombre de pas à prédire
        
        Returns: point_forecast (N, horizon), quantile_forecast (N, horizon, quantiles)
        """
        N = len(timeseries)
        point_all, quantile_all = [], []
        
        print(f"\nPrédiction en cours ({N} séries temporelles)...")
        for i in range(0, N, batch_size):
            batch = timeseries[i:i + batch_size]
            inputs = [series.astype(np.float32) for series in batch]
            
            points, quantiles = self.model.forecast(
                horizon=horizon,
                inputs=inputs
            )
            
            point_all.append(points)
            quantile_all.append(quantiles)
            
            progress = min(i + batch_size, N)
            print(f"  [{progress}/{N}] séries traitées", end='\r')
        
        print(f"\n✓ Prédiction terminée!")
        return np.vstack(point_all), np.vstack(quantile_all)


# ============================================================
# 4. VISUALISATION 3D (SAVE TO DISK)
# ============================================================

def show_3d_volume(volume, timestep, title="", filename=None, show_isosurface=True):
    """Affiche le volume 3D avec coupe transversale et isosurface optionnelle"""
    fig = plt.figure(figsize=(15, 5))
    
    # Sous-plot 1: Coupe au milieu (plan Y-Z à x=nx//2)
    ax1 = fig.add_subplot(131)
    mid_x = volume.shape[1] // 2
    im1 = ax1.imshow(volume[timestep, mid_x, :, :].T, cmap="inferno", origin="lower", aspect="auto")
    ax1.set_title(f"Coupe Y-Z (x={mid_x})")
    ax1.set_xlabel("Y")
    ax1.set_ylabel("Z")
    plt.colorbar(im1, ax=ax1, label="°C")
    
    # Sous-plot 2: Coupe horizontale (plan X-Y à z=nz//2)
    ax2 = fig.add_subplot(132)
    mid_z = volume.shape[3] // 2
    im2 = ax2.imshow(volume[timestep, :, :, mid_z].T, cmap="inferno", origin="lower", aspect="auto")
    ax2.set_title(f"Coupe X-Y (z={mid_z})")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    plt.colorbar(im2, ax=ax2, label="°C")
    
    # Sous-plot 3: Vue 3D avec points de température
    ax3 = fig.add_subplot(133, projection='3d')
    
    data = volume[timestep]
    X, Y, Z = data.shape
    
    # Créer un sous-échantillonnage pour la visualisation
    stride = max(1, X // 15)
    x, y, z = np.meshgrid(
        np.arange(0, X, stride),
        np.arange(0, Y, stride),
        np.arange(0, Z, stride),
        indexing='ij'
    )
    
    temps = data[::stride, ::stride, ::stride].flatten()
    colors = cm.inferno(temps / 100.0)
    
    ax3.scatter(x.flatten(), y.flatten(), z.flatten(), 
               c=colors, marker='o', s=20, alpha=0.6)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Vue 3D')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if filename is None:
        filename = title.replace(" ", "_").lower()
    
    path = os.path.join(RESULTS_DIR, f"{filename}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def compare_true_vs_pred_3d(true_vol, pred_vol, timestep, filename="comparison_3d"):
    """Compare les volumes 3D vrais et prédits"""
    fig = plt.figure(figsize=(18, 5))
    
    mid_x = true_vol.shape[1] // 2
    
    # Ground Truth
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(true_vol[timestep, mid_x, :, :].T, cmap="inferno", origin="lower")
    ax1.set_title("Ground Truth (Coupe Y-Z)", fontsize=12)
    ax1.set_xlabel("Y")
    ax1.set_ylabel("Z")
    plt.colorbar(im1, ax=ax1, label="°C")
    
    # Prediction
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(pred_vol[timestep, mid_x, :, :].T, cmap="inferno", origin="lower")
    ax2.set_title("TimesFM Prediction (Coupe Y-Z)", fontsize=12)
    ax2.set_xlabel("Y")
    ax2.set_ylabel("Z")
    plt.colorbar(im2, ax=ax2, label="°C")
    
    # Error
    ax3 = fig.add_subplot(133)
    error = np.abs(true_vol[timestep, mid_x, :, :] - pred_vol[timestep, mid_x, :, :])
    im3 = ax3.imshow(error.T, cmap="hot", origin="lower")
    ax3.set_title("Absolute Error (Coupe Y-Z)", fontsize=12)
    ax3.set_xlabel("Y")
    ax3.set_ylabel("Z")
    plt.colorbar(im3, ax=ax3, label="°C")
    
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{filename}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_temperature_at_point(true_vol, pred_vol, context_len, point, 
                               title="Temperature Evolution", filename="temp_evo"):
    """
    Trace l'évolution de température (vraie et prédite) en un point 3D.
    """
    x, y, z = point
    
    true_series = true_vol[:, x, y, z]
    pred_series = pred_vol[:, x, y, z]
    
    T_true = len(true_series)
    T_pred = len(pred_series)
    
    print(f"\nPoint ({x}, {y}, {z}):")
    print(f"  Température réelle finale: {true_series[-1]:.2f}°C")
    if T_pred > 0:
        print(f"  Température prédite finale: {pred_series[-1]:.2f}°C")
        print(f"  Erreur absolue: {abs(true_series[context_len + T_pred - 1] - pred_series[-1]):.2f}°C")
    
    plt.figure(figsize=(10, 5))
    
    plt.plot(range(context_len), true_series[:context_len], 
             'b-', linewidth=2, label="Historical Context", alpha=0.7)
    
    plt.plot(range(context_len - 1, T_true), true_series[context_len - 1:], 
             'g-', linewidth=2, label="True Future")
    
    if T_pred > 0:
        plt.plot(range(context_len, context_len + T_pred), pred_series, 
                 'r--', linewidth=2, label="TimesFM Prediction")
    
    plt.axvline(x=context_len, color='gray', linestyle=':', 
                linewidth=1.5, label="Forecast Start")
    
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.title(f"{title} at Point ({x}, {y}, {z})", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if filename is None:
        filename = title.replace(" ", "_").lower()
    
    path = os.path.join(RESULTS_DIR, f"{filename}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# SCRIPT PRINCIPAL
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Simulation 3D de diffusion de chaleur avec prédiction TimesFM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--total", type=int, default=200,
                       help="Nombre total de pas de temps")
    parser.add_argument("--horizon", type=int, default=50,
                       help="Horizon de prédiction")
    parser.add_argument("--material", type=str, default="steel",
                       choices=list(MATERIALS.keys()),
                       help="Matériau de la plaque")
    parser.add_argument("--nx", type=int, default=24,
                       help="Points de grille en x")
    parser.add_argument("--ny", type=int, default=24,
                       help="Points de grille en y")
    parser.add_argument("--nz", type=int, default=24,
                       help="Points de grille en z")
    parser.add_argument("--dx", type=float, default=0.01,
                       help="Pas spatial en x [m]")
    parser.add_argument("--dy", type=float, default=0.01,
                       help="Pas spatial en y [m]")
    parser.add_argument("--dz", type=float, default=0.01,
                       help="Pas spatial en z [m]")
    parser.add_argument("--dt", type=float, default=0.1,
                       help="Pas de temps [s]")
    parser.add_argument("--disruption", type=str, default=None,
                       help="Perturbation: [x,y,z,temp,instant] ex: [10,12,12,150,50]")
    
    args = parser.parse_args()
    
    # Parser la perturbation si fournie
    disruption = None
    if args.disruption:
        try:
            # Enlever les crochets et parser
            disruption_str = args.disruption.strip('[]')
            disruption_values = [float(x.strip()) for x in disruption_str.split(',')]
            
            if len(disruption_values) != 5:
                raise ValueError("La perturbation doit avoir exactement 5 valeurs")
            
            disruption = {
                'x': int(disruption_values[0]),
                'y': int(disruption_values[1]),
                'z': int(disruption_values[2]),
                'temp': disruption_values[3],
                'instant': int(disruption_values[4])
            }
            
            # Validation
            if not (0 <= disruption['x'] < args.nx):
                raise ValueError(f"x doit être entre 0 et {args.nx-1}")
            if not (0 <= disruption['y'] < args.ny):
                raise ValueError(f"y doit être entre 0 et {args.ny-1}")
            if not (0 <= disruption['z'] < args.nz):
                raise ValueError(f"z doit être entre 0 et {args.nz-1}")
            if not (0 <= disruption['instant'] < args.total):
                raise ValueError(f"instant doit être entre 0 et {args.total-1}")
            
            print(f"\n🔥 Perturbation configurée:")
            print(f"   Position: ({disruption['x']}, {disruption['y']}, {disruption['z']})")
            print(f"   Température: {disruption['temp']}°C")
            print(f"   Instant de départ: t={disruption['instant']}")
            
        except Exception as e:
            raise ValueError(f"Erreur de parsing de --disruption: {e}")
    
    if args.horizon >= args.total:
        raise ValueError(f"L'horizon ({args.horizon}) doit être < total ({args.total})")
    
    context_len = args.total - args.horizon
    
    print("=" * 70)
    print("SIMULATION 3D DE DIFFUSION THERMIQUE AVEC TIMESFM")
    print("=" * 70)
    print(f"\nParamètres:")
    print(f"  Total: {args.total}, Contexte: {context_len}, Horizon: {args.horizon}")
    print(f"  Grille: {args.nx} × {args.ny} × {args.nz}")
    print(f"  Pas spatiaux: dx={args.dx}m, dy={args.dy}m, dz={args.dz}m")
    print(f"  Pas temporel: dt={args.dt}s")
    
    if disruption:
        print(f"\n🌡️  Perturbation active à t={disruption['instant']}")
        print(f"     Position: ({disruption['x']}, {disruption['y']}, {disruption['z']})")
        print(f"     Température: {disruption['temp']}°C")
    
    # 1. SIMULATION 3D
    volume = simulate_heat_diffusion_3d(
        nx=args.nx, ny=args.ny, nz=args.nz,
        nt=args.total,
        dx=args.dx, dy=args.dy, dz=args.dz,
        dt=args.dt,
        material=args.material,
        disruption=disruption
    )
    
    print(f"\n✓ Simulation 3D terminée: shape = {volume.shape} (T, X, Y, Z)")
    
    # 2. SÉPARATION
    volume_context = volume[:context_len]
    volume_future = volume[context_len:context_len + args.horizon]
    
    # 3. CONVERSION
    print("\n" + "=" * 70)
    print("PRÉPARATION DES DONNÉES")
    print("=" * 70)
    
    timeseries_context = volume_to_timeseries(volume_context)
    print(f"Séries temporelles: {timeseries_context.shape[0]} (une par voxel)")
    print(f"Longueur: {timeseries_context.shape[1]} pas")
    
    # 4. FORECASTING
    print("\n" + "=" * 70)
    print("PRÉDICTION AVEC TIMESFM")
    print("=" * 70)
    
    forecaster = TimesFMForecaster(max_context=context_len, max_horizon=args.horizon)
    point_forecast, quantile_forecast = forecaster.forecast(
        timeseries_context,
        horizon=args.horizon,
        batch_size=32
    )
    
    # 5. RECONSTRUCTION
    predicted_volume = timeseries_to_volume(
        point_forecast,
        spatial_shape=(args.nx, args.ny, args.nz)
    )
    
    print(f"\nVolume prédit: {predicted_volume.shape}")
    
    # 6. VISUALISATION 3D
    print("\n" + "=" * 70)
    print("GÉNÉRATION DES VISUALISATIONS 3D")
    print("=" * 70)
    
    t_vis = min(10, args.horizon - 1)
    
    show_3d_volume(volume_context, -1,
                   title=f"Last Context Frame (t={context_len})",
                   filename="last_context_3d")
    
    show_3d_volume(volume_future, t_vis,
                   title=f"Ground Truth 3D (t={context_len + t_vis})",
                   filename="ground_truth_3d")
    
    show_3d_volume(predicted_volume, t_vis,
                   title=f"TimesFM Prediction 3D (t={context_len + t_vis})",
                   filename="prediction_3d")
    
    compare_true_vs_pred_3d(volume_future, predicted_volume, t_vis,
                           filename="comparison_3d")
    
    # 7. ÉVOLUTION TEMPORELLE
    points = [
        (args.nx // 4, args.ny // 2, args.nz // 2, "Left Quarter"),
        (args.nx // 2, args.ny // 2, args.nz // 2, "Center"),
        (3 * args.nx // 4, args.ny // 2, args.nz // 2, "Right Quarter"),
    ]
    
    print("\nTracé des courbes d'évolution:")
    for x, y, z, label in points:
        plot_temperature_at_point(
            true_vol=volume,
            pred_vol=predicted_volume,
            context_len=context_len,
            point=(x, y, z),
            title=f"Temperature: {label}",
            filename=f"temp_evo_{label.lower().replace(' ', '_')}"
        )
    
    # 8. MÉTRIQUES
    print("\n" + "=" * 70)
    print("MÉTRIQUES D'ERREUR")
    print("=" * 70)
    
    mae = np.mean(np.abs(volume_future - predicted_volume))
    rmse = np.sqrt(np.mean((volume_future - predicted_volume)**2))
    max_error = np.max(np.abs(volume_future - predicted_volume))
    
    print(f"\nErreur globale:")
    print(f"  MAE: {mae:.4f}°C")
    print(f"  RMSE: {rmse:.4f}°C")
    print(f"  Max: {max_error:.4f}°C")
    
    print("\n" + "=" * 70)
    print(f"✓ Résultats sauvegardés dans ./{RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()