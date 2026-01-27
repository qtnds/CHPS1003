import numpy as np
import torch
import matplotlib.pyplot as plt
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
# 1. SIMULATION PHYSIQUE (DIFFUSION DE LA CHALEUR 2D)
# ============================================================

def simulate_heat_diffusion(nx=64, ny=64, nt=200, dx=0.01, dy=0.01, dt=0.1, 
                           material="steel"):
    """
    Simule la diffusion de chaleur avec une source à 100°C sur l'arête droite.
    Utilise des conditions aux limites de Neumann (flux nul) sur les bords,
    sauf sur l'arête droite où on impose T = 100°C (Dirichlet).
    
    nx, ny: dimensions spatiales (nombre de points en x et y)
    nt: nombre de pas de temps
    dx, dy: pas d'espace [m]
    dt: pas de temps [s]
    material: type de matériau ("steel", "copper", "aluminum")
    """
    # Récupérer la diffusivité thermique du matériau
    if material not in MATERIALS:
        raise ValueError(f"Matériau '{material}' inconnu. Choix: {list(MATERIALS.keys())}")
    
    alpha = MATERIALS[material]["thermal_diffusivity"]
    material_name = MATERIALS[material]["name"]
    
    print(f"\nMatériau: {material_name}")
    print(f"Diffusivité thermique α = {alpha:.2e} m²/s")
    print(f"Dimensions: {nx*dx:.3f}m × {ny*dy:.3f}m")
    print(f"Pas de temps dt = {dt:.3f}s")
    
    # Vérification de la stabilité (critère CFL pour méthode explicite)
    # dt < dx²dy² / (2α(dx² + dy²))
    dt_max = (dx**2 * dy**2) / (2 * alpha * (dx**2 + dy**2))
    if dt > dt_max:
        print(f"⚠️  ATTENTION: dt={dt:.3e}s > dt_max={dt_max:.3e}s")
        print(f"   Le schéma peut être instable! Considérez réduire dt.")
    else:
        print(f"✓ Stabilité: dt={dt:.3e}s < dt_max={dt_max:.3e}s")
    
    T = np.zeros((nt, nx, ny), dtype=np.float32)
    
    # Condition initiale: température 0°C partout
    # Source de chaleur: arête droite à 100°C
    T[0, -1, :] = 100.0  # Dernière colonne (x=nx-1) = arête droite
    
    for t in range(nt - 1):
        # Créer une copie pour le calcul
        T_current = T[t].copy()
        
        # Calcul du Laplacien avec différences finies
        # en respectant les conditions aux limites
        laplacian = np.zeros_like(T_current)
        
        # Intérieur du domaine (pas sur les bords)
        laplacian[1:-1, 1:-1] = (
            (T_current[2:, 1:-1] - 2*T_current[1:-1, 1:-1] + T_current[:-2, 1:-1]) / (dx**2) +
            (T_current[1:-1, 2:] - 2*T_current[1:-1, 1:-1] + T_current[1:-1, :-2]) / (dy**2)
        )
        
        # Bord gauche (x=0) : condition de Neumann (dT/dx = 0)
        laplacian[0, 1:-1] = (
            (T_current[1, 1:-1] - T_current[0, 1:-1]) / (dx**2) +
            (T_current[0, 2:] - 2*T_current[0, 1:-1] + T_current[0, :-2]) / (dy**2)
        )
        
        # Bord droit (x=nx-1) : condition de Dirichlet (T = 100°C)
        # On ne calcule pas le Laplacien ici car on va imposer T
        
        # Bord bas (y=0) : condition de Neumann (dT/dy = 0)
        laplacian[1:-1, 0] = (
            (T_current[2:, 0] - 2*T_current[1:-1, 0] + T_current[:-2, 0]) / (dx**2) +
            (T_current[1:-1, 1] - T_current[1:-1, 0]) / (dy**2)
        )
        
        # Bord haut (y=ny-1) : condition de Neumann (dT/dy = 0)
        laplacian[1:-1, -1] = (
            (T_current[2:, -1] - 2*T_current[1:-1, -1] + T_current[:-2, -1]) / (dx**2) +
            (T_current[1:-1, -1] - T_current[1:-1, -2]) / (dy**2)
        )
        
        # Coins (conditions de Neumann sur les deux directions)
        # Coin bas-gauche
        laplacian[0, 0] = (
            (T_current[1, 0] - T_current[0, 0]) / (dx**2) +
            (T_current[0, 1] - T_current[0, 0]) / (dy**2)
        )
        
        # Coin haut-gauche
        laplacian[0, -1] = (
            (T_current[1, -1] - T_current[0, -1]) / (dx**2) +
            (T_current[0, -1] - T_current[0, -2]) / (dy**2)
        )
        
        # Les coins droits sont imposés à 100°C (pas besoin de calculer le Laplacien)
        
        # Mise à jour temporelle (équation de la chaleur)
        T[t+1] = T[t] + alpha * dt * laplacian
        
        # Imposer les conditions aux limites de Dirichlet
        # Arête droite constante à 100°C
        T[t+1, -1, :] = 100.0
    
    return T


# ============================================================
# 2. VOLUME <-> SERIES TEMPORELLES
# ============================================================

def volume_to_timeseries(volume):
    """Convertit (T, X, Y) -> (N, T) où N = X*Y séries temporelles"""
    T, X, Y = volume.shape
    return volume.reshape(T, X * Y).T


def timeseries_to_volume(timeseries, spatial_shape):
    """Convertit (N, T) -> (T, X, Y)"""
    N, T = timeseries.shape
    X, Y = spatial_shape
    return timeseries.T.reshape(T, X, Y)


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
            
            # Afficher la progression
            progress = min(i + batch_size, N)
            print(f"  [{progress}/{N}] séries traitées", end='\r')
        
        print(f"\n✓ Prédiction terminée!")
        return np.vstack(point_all), np.vstack(quantile_all)


# ============================================================
# 4. VISUALISATION (SAVE TO DISK)
# ============================================================

def show_slice(volume, timestep, title="", filename=None):
    plt.figure(figsize=(6, 5))
    plt.imshow(volume[timestep].T, cmap="inferno", origin="lower", aspect="auto")
    plt.colorbar(label="Temperature (°C)")
    plt.title(title)
    plt.xlabel("Width (x)")
    plt.ylabel("Height (y)")
    
    if filename is None:
        filename = title.replace(" ", "_").lower()
    
    path = os.path.join(RESULTS_DIR, f"{filename}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def compare_true_vs_pred(true_vol, pred_vol, timestep, filename="comparison"):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    im0 = axs[0].imshow(true_vol[timestep].T, cmap="inferno", origin="lower")
    axs[0].set_title("Ground Truth", fontsize=14)
    axs[0].set_xlabel("Width")
    axs[0].set_ylabel("Height")
    plt.colorbar(im0, ax=axs[0], label="°C")
    
    im1 = axs[1].imshow(pred_vol[timestep].T, cmap="inferno", origin="lower")
    axs[1].set_title("TimesFM Prediction", fontsize=14)
    axs[1].set_xlabel("Width")
    axs[1].set_ylabel("Height")
    plt.colorbar(im1, ax=axs[1], label="°C")
    
    error = np.abs(true_vol[timestep] - pred_vol[timestep])
    im2 = axs[2].imshow(error.T, cmap="hot", origin="lower")
    axs[2].set_title("Absolute Error", fontsize=14)
    axs[2].set_xlabel("Width")
    axs[2].set_ylabel("Height")
    plt.colorbar(im2, ax=axs[2], label="°C")
    
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{filename}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_temperature_at_point(true_vol, pred_vol, context_len, point, 
                               title="Temperature Evolution", filename="temp_evo"):
    """
    Trace l'évolution de température (vraie et prédite) en un point.
    
    true_vol: (T_total, X, Y) - simulation complète
    pred_vol: (T_pred, X, Y) - prédictions
    context_len: nombre de pas utilisés pour le contexte
    point: (x, y)
    """
    x, y = point
    
    # Extraire les séries temporelles au point d'intérêt
    true_series = true_vol[:, x, y]
    pred_series = pred_vol[:, x, y]
    
    T_true = len(true_series)
    T_pred = len(pred_series)
    
    # Afficher les valeurs finales
    print(f"\nPoint ({x}, {y}):")
    print(f"  Température réelle finale: {true_series[-1]:.2f}°C")
    if T_pred > 0:
        print(f"  Température prédite finale: {pred_series[-1]:.2f}°C")
        print(f"  Erreur absolue: {abs(true_series[context_len + T_pred - 1] - pred_series[-1]):.2f}°C")
    
    # Visualisation
    plt.figure(figsize=(10, 5))
    
    # Contexte (historique)
    plt.plot(range(context_len), true_series[:context_len], 
             'b-', linewidth=2, label="Historical Context", alpha=0.7)
    
    # Continuation vraie (ground truth)
    plt.plot(range(context_len - 1, T_true), true_series[context_len - 1:], 
             'g-', linewidth=2, label="True Future")
    
    # Prédiction
    if T_pred > 0:
        plt.plot(range(context_len, context_len + T_pred), pred_series, 
                 'r--', linewidth=2, label="TimesFM Prediction")
    
    # Ligne verticale pour séparer contexte/prédiction
    plt.axvline(x=context_len, color='gray', linestyle=':', 
                linewidth=1.5, label="Forecast Start")
    
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.title(f"{title} at Point ({x}, {y})", fontsize=14)
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
    # ============================================================
    # ARGUMENTS EN LIGNE DE COMMANDE
    # ============================================================
    parser = argparse.ArgumentParser(
        description="Simulation de diffusion de chaleur avec prédiction TimesFM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--total", 
        type=int, 
        default=200,
        help="Nombre total de pas de temps de la simulation"
    )
    
    parser.add_argument(
        "--horizon", 
        type=int, 
        default=50,
        help="Horizon de prédiction TimesFM (nombre de pas futurs)"
    )
    
    parser.add_argument(
        "--material",
        type=str,
        default="steel",
        choices=list(MATERIALS.keys()),
        help="Matériau de la plaque"
    )
    
    parser.add_argument(
        "--nx",
        type=int,
        default=32,
        help="Nombre de points de grille en x"
    )
    
    parser.add_argument(
        "--ny",
        type=int,
        default=32,
        help="Nombre de points de grille en y"
    )
    
    parser.add_argument(
        "--dx",
        type=float,
        default=0.01,
        help="Pas spatial en x [m]"
    )
    
    parser.add_argument(
        "--dy",
        type=float,
        default=0.01,
        help="Pas spatial en y [m]"
    )
    
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Pas de temps [s]"
    )
    
    args = parser.parse_args()
    
    # Validation des arguments
    if args.horizon >= args.total:
        raise ValueError(f"L'horizon ({args.horizon}) doit être inférieur au total ({args.total})")
    
    context_len = args.total - args.horizon
    
    if context_len < 50:
        print(f"⚠️  ATTENTION: Contexte très court ({context_len} pas). Considérez augmenter --total.")
    
    # ============================================================
    # 1. GENERATION DU DATASET
    # ============================================================
    print("=" * 70)
    print("SIMULATION DE DIFFUSION THERMIQUE AVEC TIMESFM")
    print("=" * 70)
    print(f"\nParamètres de simulation:")
    print(f"  Total de pas de temps: {args.total}")
    print(f"  Contexte (historique): {context_len}")
    print(f"  Horizon de prédiction: {args.horizon}")
    print(f"  Grille spatiale: {args.nx} × {args.ny}")
    print(f"  Pas spatiaux: dx={args.dx}m, dy={args.dy}m")
    print(f"  Pas temporel: dt={args.dt}s")
    
    volume = simulate_heat_diffusion(
        nx=args.nx, 
        ny=args.ny, 
        nt=args.total, 
        dx=args.dx,
        dy=args.dy,
        dt=args.dt,
        material=args.material
    )
    
    print(f"\n✓ Simulation terminée: shape = {volume.shape} (T, X, Y)")
    print(f"\nTempérature sur l'arête droite (source) au dernier pas:")
    print(f"  Min: {volume[-1, -1, :].min():.2f}°C")
    print(f"  Max: {volume[-1, -1, :].max():.2f}°C")
    print(f"  Moyenne: {volume[-1, -1, :].mean():.2f}°C")
    
    # ============================================================
    # 2. SEPARATION CONTEXTE / FUTUR
    # ============================================================
    volume_context = volume[:context_len]
    volume_future = volume[context_len:context_len + args.horizon]
    
    # ============================================================
    # 3. CONVERSION EN SERIES TEMPORELLES
    # ============================================================
    print("\n" + "=" * 70)
    print("PRÉPARATION DES DONNÉES POUR TIMESFM")
    print("=" * 70)
    
    timeseries_context = volume_to_timeseries(volume_context)
    print(f"Nombre de séries temporelles: {timeseries_context.shape[0]} (une par pixel)")
    print(f"Longueur de chaque série: {timeseries_context.shape[1]} pas de temps")
    
    # ============================================================
    # 4. FORECASTING AVEC TIMESFM
    # ============================================================
    print("\n" + "=" * 70)
    print("PRÉDICTION AVEC TIMESFM")
    print("=" * 70)
    
    forecaster = TimesFMForecaster(max_context=context_len, max_horizon=args.horizon)
    point_forecast, quantile_forecast = forecaster.forecast(
        timeseries_context,
        horizon=args.horizon,
        batch_size=32
    )
    
    print(f"\nRésultats de prédiction:")
    print(f"  Point forecast shape: {point_forecast.shape}")
    print(f"  Quantile forecast shape: {quantile_forecast.shape}")
    
    # ============================================================
    # 5. RECONSTRUCTION DU VOLUME
    # ============================================================
    predicted_volume = timeseries_to_volume(
        point_forecast,
        spatial_shape=(args.nx, args.ny)
    )
    
    print(f"  Predicted volume shape: {predicted_volume.shape}")
    
    # ============================================================
    # 6. VISUALISATION
    # ============================================================
    print("\n" + "=" * 70)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("=" * 70)
    
    t_vis = min(10, args.horizon - 1)
    
    show_slice(volume_context, -1, 
               title=f"Last Context Frame (t={context_len})", 
               filename="last_context")
    
    show_slice(volume_future, t_vis, 
               title=f"Ground Truth (t={context_len + t_vis})", 
               filename="ground_truth")
    
    show_slice(predicted_volume, t_vis, 
               title=f"TimesFM Prediction (t={context_len + t_vis})", 
               filename="prediction")
    
    compare_true_vs_pred(volume_future, predicted_volume, t_vis, 
                         filename="comparison")
    
    # ============================================================
    # 7. ÉVOLUTION TEMPORELLE EN PLUSIEURS POINTS
    # ============================================================
    points = [
        (args.nx // 4, args.ny // 2, "Left Quarter"),
        (args.nx // 2, args.ny // 2, "Center"),
        (3 * args.nx // 4, args.ny // 2, "Right Quarter"),
    ]
    
    print("\nTracé des courbes d'évolution temporelle:")
    for x, y, label in points:
        plot_temperature_at_point(
            true_vol=volume,
            pred_vol=predicted_volume,
            context_len=context_len,
            point=(x, y),
            title=f"Temperature: {label}",
            filename=f"temp_evo_{label.lower().replace(' ', '_')}"
        )
    
    # ============================================================
    # 8. MÉTRIQUES D'ERREUR
    # ============================================================
    print("\n" + "=" * 70)
    print("MÉTRIQUES D'ERREUR")
    print("=" * 70)
    
    mae = np.mean(np.abs(volume_future - predicted_volume))
    rmse = np.sqrt(np.mean((volume_future - predicted_volume)**2))
    max_error = np.max(np.abs(volume_future - predicted_volume))
    
    print(f"\nErreur globale sur l'horizon de prédiction:")
    print(f"  MAE (Mean Absolute Error): {mae:.4f}°C")
    print(f"  RMSE (Root Mean Square Error): {rmse:.4f}°C")
    print(f"  Erreur maximale: {max_error:.4f}°C")
    
    print("\n" + "=" * 70)
    print(f"✓ Tous les résultats sauvegardés dans ./{RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()