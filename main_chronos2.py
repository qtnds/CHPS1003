import argparse
import numpy as np
import torch
from chronos import ChronosPipeline
from utilities import (
    MATERIALS,
    simulate_heat_diffusion_3d,
    volume_to_timeseries,
    timeseries_to_volume,
    TimesFMForecaster,
    show_3d_volume,
    compare_true_vs_pred_3d,
    plot_temperature_at_point,
    create_surface_mask,
    RESULTS_DIR
)
import matplotlib.pyplot as plt
import os


# ============================================================
# CHRONOS FORECASTER
# ============================================================

class ChronosForecaster:
    def __init__(self, model_name="amazon/chronos-t5-small", device="cuda"):
        print(f"\n🕰️  Chargement Chronos ({model_name}) …")
        self.device = device
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        print("✓ Chronos prêt")

    def forecast(self, timeseries, horizon, batch_size=8, num_samples=20):
        N = len(timeseries)
        predictions = np.zeros((N, horizon), dtype=np.float32)
        print(f"\n🔮 Prédiction Chronos ({N} séries, batch={batch_size}) …")

        for i in range(0, N, batch_size):
            batch_end = min(i + batch_size, N)
            batch = [torch.tensor(series, dtype=torch.float32)
                     for series in timeseries[i:batch_end]]
            try:
                with torch.no_grad():
                    forecast = self.pipeline.predict(
                        batch,
                        prediction_length=horizon,
                        num_samples=num_samples,
                    )
                predictions[i:batch_end] = np.median(forecast.numpy(), axis=1)
                del forecast
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"\n⚠️  OOM au batch {i}, passage en mode unitaire …")
                for j in range(i, batch_end):
                    single = [torch.tensor(timeseries[j], dtype=torch.float32)]
                    with torch.no_grad():
                        fc = self.pipeline.predict(
                            single,
                            prediction_length=horizon,
                            num_samples=num_samples,
                        )
                    predictions[j] = np.median(fc.numpy(), axis=1)[0]
                    del fc
                    torch.cuda.empty_cache()

            if batch_end % 500 == 0:
                print(f"   [{batch_end}/{N}]")

        print("✓ Chronos terminé !")
        return predictions


# ============================================================
# VISUALISATIONS MULTI-MODÈLES
# ============================================================

def plot_model_comparison(errors_dict, filename="model_comparison"):
    """Bar chart MAE / RMSE pour tous les modèles."""
    models    = list(errors_dict.keys())
    mae_vals  = [errors_dict[m]['mae']  for m in models]
    rmse_vals = [errors_dict[m]['rmse'] for m in models]
    colors    = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, vals, label in zip(axes, [mae_vals, rmse_vals], ['MAE (°C)', 'RMSE (°C)']):
        ax.bar(models, vals, color=colors[:len(models)])
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle('Comparaison des performances de prédiction',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{filename}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Sauvegardé : {filename}.png")


def plot_temperature_all_models(true_vol, predictions_dict, context_len, point,
                                title="Temperature Evolution", filename="temp_evo",
                                disruption_time=None, convection_time=None):
    """
    Évolution temporelle en un point pour tous les modèles sur une seule figure.

    Parameters
    ----------
    predictions_dict : dict  {nom_modèle: volume_prédit (T_horizon, X, Y, Z)}
    """
    x, y, z = point
    fig, ax = plt.subplots(figsize=(14, 6))

    true_series = true_vol[:, x, y, z]
    T_total     = len(true_series)

    # Contexte
    ax.plot(range(context_len), true_series[:context_len],
            'b-', lw=2.5, label="Contexte historique", alpha=0.8)
    # Vérité terrain future
    ax.plot(range(context_len - 1, T_total), true_series[context_len - 1:],
            'g-', lw=2.5, label="Vérité terrain", alpha=0.9)

    # Une courbe par modèle
    style_cycle = [
        ('r',      '--'),
        ('purple', '-.'),
        ('orange', ':'),
        ('cyan',   '--'),
    ]
    for (color, ls), (model_name, pred_vol) in zip(style_cycle, predictions_dict.items()):
        pred_series = pred_vol[:, x, y, z]
        ax.plot(range(context_len, context_len + len(pred_series)),
                pred_series, color=color, linestyle=ls, lw=2,
                label=f"Prédiction {model_name}", alpha=0.85)

    # Lignes verticales événements
    ax.axvline(context_len, color='gray', ls=':', lw=1.5,
               label="Début prédiction", alpha=0.6)
    if disruption_time is not None:
        ax.axvline(disruption_time, color='red', ls='--', lw=1.2,
                   label=f"Perturbation (t={disruption_time})", alpha=0.7)
    if convection_time is not None:
        ax.axvline(convection_time, color='blue', ls='--', lw=1.2,
                   label=f"Convection (t={convection_time})", alpha=0.7)

    ax.set_xlabel("Pas de temps", fontsize=12)
    ax.set_ylabel("Température (°C)", fontsize=12)
    ax.set_title(f"{title} au point ({x}, {y}, {z})", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, f"{filename}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Sauvegardé : {filename}.png")


def compare_all_models_3d(true_vol, predictions_dict, timestep,
                           filename="comparison_all_3d"):
    """
    Grille de coupes X-Y (z central) : Ground Truth + un panneau par modèle.
    Tous les modèles sur une seule figure.
    """
    n_models  = len(predictions_dict)
    n_panels  = 1 + n_models          # vérité terrain + modèles
    ncols     = min(n_panels, 3)
    nrows     = (n_panels + ncols - 1) // ncols

    mid_z = true_vol.shape[3] // 2

    # Plage de couleurs commune
    all_vols = [true_vol] + list(predictions_dict.values())
    vmin = min(v[timestep].min() for v in all_vols)
    vmax = max(v[timestep].max() for v in all_vols)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6 * ncols, 5 * nrows),
                              squeeze=False)

    panels = [("Ground Truth", true_vol)] + list(predictions_dict.items())

    for idx, (name, vol) in enumerate(panels):
        r, c = divmod(idx, ncols)
        ax   = axes[r][c]
        im   = ax.imshow(vol[timestep, :, :, mid_z].T,
                         cmap='inferno', origin='lower',
                         vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='°C')

    # Masquer les cases vides si grille non pleine
    for idx in range(len(panels), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    plt.suptitle(f'Comparaison Ground Truth vs Modèles — t={timestep}',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{filename}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Sauvegardé : {filename}.png")


# ============================================================
# PARSING (inchangé)
# ============================================================

def parse_disruption(disruption_str):
    """Parse la chaîne de perturbation [x,y,z,temp,instant]"""
    try:
        disruption_str = disruption_str.strip('[]')
        values = [float(x.strip()) for x in disruption_str.split(',')]
        if len(values) != 5:
            raise ValueError("La perturbation doit avoir exactement 5 valeurs")
        return {
            'x': int(values[0]),
            'y': int(values[1]),
            'z': int(values[2]),
            'temp': values[3],
            'instant': int(values[4])
        }
    except Exception as e:
        raise ValueError(f"Erreur de parsing de --disruption: {e}")


def parse_convection(convection_str, nx, ny, nz):
    """
    Parse la chaîne de convection.
    Format: [T_air,h,surface_type,t_start,params...]
    """
    try:
        convection_str = convection_str.strip('[]')
        parts = convection_str.split(',')
        if len(parts) < 4:
            raise ValueError("Convection doit avoir au moins 4 paramètres de base")

        T_air = float(parts[0])
        h     = float(parts[1])
        surface_type = parts[2].strip()
        t_start      = int(parts[3])

        if surface_type == 'face':
            if len(parts) < 5:
                raise ValueError("Type 'face' nécessite: [T_air,h,face,t_start,face_name]")
            surface_spec = {'type': 'face', 'face': parts[4].strip()}

        elif surface_type == 'rectangle':
            if len(parts) < 9:
                raise ValueError("Type 'rectangle' nécessite 9 paramètres")
            face_name = parts[4].strip()
            if face_name in ['top', 'bottom']:
                surface_spec = {
                    'type': 'rectangle', 'face': face_name,
                    'x_range': (int(parts[5]), int(parts[6])),
                    'y_range': (int(parts[7]), int(parts[8]))
                }
            elif face_name in ['left', 'right']:
                surface_spec = {
                    'type': 'rectangle', 'face': face_name,
                    'y_range': (int(parts[5]), int(parts[6])),
                    'z_range': (int(parts[7]), int(parts[8]))
                }
            else:
                surface_spec = {
                    'type': 'rectangle', 'face': face_name,
                    'x_range': (int(parts[5]), int(parts[6])),
                    'z_range': (int(parts[7]), int(parts[8]))
                }

        elif surface_type == 'circle':
            if len(parts) < 8:
                raise ValueError("Type 'circle' nécessite 8 paramètres")
            surface_spec = {
                'type': 'circle', 'face': parts[4].strip(),
                'center': (int(parts[5]), int(parts[6])),
                'radius': float(parts[7])
            }

        else:
            raise ValueError(f"Type de surface '{surface_type}' non reconnu")

        return {'T_air': T_air, 'h': h, 'surface': surface_spec, 't_start': t_start}

    except Exception as e:
        raise ValueError(f"Erreur de parsing de --convection: {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Simulation 3D de diffusion thermique avec perturbation et convection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Paramètres de simulation (inchangés)
    parser.add_argument("--total",    type=int,   default=200)
    parser.add_argument("--horizon",  type=int,   default=50)
    parser.add_argument("--material", type=str,   default="steel",
                        choices=list(MATERIALS.keys()))
    parser.add_argument("--nx", type=int,   default=24)
    parser.add_argument("--ny", type=int,   default=24)
    parser.add_argument("--nz", type=int,   default=24)
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--dy", type=float, default=0.01)
    parser.add_argument("--dz", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--disruption", type=str, default=None)
    parser.add_argument("--convection", type=str, default=None)

    # Nouveaux arguments Chronos
    parser.add_argument("--skip-chronos",    action="store_true",
                        help="Sauter Chronos")
    parser.add_argument("--chronos-model",   type=str,
                        default="amazon/chronos-t5-small",
                        choices=["amazon/chronos-t5-tiny",
                                 "amazon/chronos-t5-small",
                                 "amazon/chronos-t5-base"],
                        help="Variante Chronos")
    parser.add_argument("--chronos-batch",   type=int, default=8,
                        help="Taille de batch Chronos")
    parser.add_argument("--chronos-samples", type=int, default=20,
                        help="Échantillons Monte-Carlo Chronos")

    args = parser.parse_args()

    if args.horizon >= args.total:
        raise ValueError(f"L'horizon ({args.horizon}) doit être < total ({args.total})")

    context_len = args.total - args.horizon

    # --- Parsing perturbation / convection (inchangé) ---
    disruption = None
    if args.disruption:
        disruption = parse_disruption(args.disruption)
        if not (0 <= disruption['x'] < args.nx):
            raise ValueError(f"x doit être entre 0 et {args.nx-1}")
        if not (0 <= disruption['y'] < args.ny):
            raise ValueError(f"y doit être entre 0 et {args.ny-1}")
        if not (0 <= disruption['z'] < args.nz):
            raise ValueError(f"z doit être entre 0 et {args.nz-1}")
        if not (0 <= disruption['instant'] < args.total):
            raise ValueError(f"instant doit être entre 0 et {args.total-1}")
        print(f"\n🔥 Perturbation ponctuelle:")
        print(f"   Position: ({disruption['x']}, {disruption['y']}, {disruption['z']})")
        print(f"   Température: {disruption['temp']}°C")
        print(f"   Instant: t={disruption['instant']}")

    convection      = None
    convection_mask = None
    if args.convection:
        convection = parse_convection(args.convection, args.nx, args.ny, args.nz)
        if not (0 <= convection['t_start'] < args.total):
            raise ValueError(f"t_start doit être entre 0 et {args.total-1}")
        convection_mask = create_surface_mask(
            args.nx, args.ny, args.nz, convection['surface'])
        print(f"\n🌬️  Convection d'air:")
        print(f"   Température air: {convection['T_air']}°C")
        print(f"   Coefficient h: {convection['h']} W/(m²·K)")
        print(f"   Type: {convection['surface']['type']}")
        print(f"   Face: {convection['surface'].get('face', 'N/A')}")
        print(f"   Surface: {np.sum(convection_mask)} points")
        print(f"   Début: t={convection['t_start']}")

    print("\n" + "=" * 70)
    print("SIMULATION 3D DE DIFFUSION THERMIQUE")
    print("=" * 70)
    print(f"\n  Total: {args.total} | Contexte: {context_len} | Horizon: {args.horizon}")
    print(f"  Grille: {args.nx}×{args.ny}×{args.nz} | Matériau: {args.material}")

    # ============================================================
    # 1. SIMULATION (inchangée)
    # ============================================================
    print("\n" + "=" * 70)
    print("SIMULATION PHYSIQUE")
    print("=" * 70)

    volume = simulate_heat_diffusion_3d(
        nx=args.nx, ny=args.ny, nz=args.nz,
        nt=args.total,
        dx=args.dx, dy=args.dy, dz=args.dz,
        dt=args.dt,
        material=args.material,
        disruption=disruption,
        convection=convection
    )
    print(f"\n✓ Simulation terminée: shape = {volume.shape} (T, X, Y, Z)")

    volume_context = volume[:context_len]
    volume_future  = volume[context_len:context_len + args.horizon]

    timeseries_context = volume_to_timeseries(volume_context)
    spatial_shape      = (args.nx, args.ny, args.nz)
    print(f"\n📈 Séries: {timeseries_context.shape[0]} voxels × {timeseries_context.shape[1]} pas")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    errors_dict = {}

    # ============================================================
    # 2. TIMESFM (inchangé)
    # ============================================================
    print("\n" + "=" * 70)
    print("PRÉDICTION AVEC TIMESFM")
    print("=" * 70)

    forecaster = TimesFMForecaster(max_context=context_len, max_horizon=args.horizon)
    point_forecast, quantile_forecast = forecaster.forecast(
        timeseries_context, horizon=args.horizon, batch_size=32)

    predicted_volume = timeseries_to_volume(point_forecast, spatial_shape)
    print(f"Volume prédit TimesFM: {predicted_volume.shape}")

    mae_tfm  = float(np.mean(np.abs(volume_future - predicted_volume)))
    rmse_tfm = float(np.sqrt(np.mean((volume_future - predicted_volume)**2)))
    errors_dict['TimesFM'] = dict(mae=mae_tfm, rmse=rmse_tfm)

    # ============================================================
    # 3. CHRONOS (nouveau)
    # ============================================================
    predicted_volume_chronos = None
    if not args.skip_chronos:
        print("\n" + "=" * 70)
        print(f"PRÉDICTION AVEC CHRONOS ({args.chronos_model})")
        print("=" * 70)

        forecaster_chronos = ChronosForecaster(
            model_name=args.chronos_model, device=device)
        pred_chronos = forecaster_chronos.forecast(
            timeseries_context, args.horizon,
            batch_size=args.chronos_batch,
            num_samples=args.chronos_samples,
        )
        predicted_volume_chronos = timeseries_to_volume(pred_chronos, spatial_shape)
        print(f"Volume prédit Chronos: {predicted_volume_chronos.shape}")

        mae_ch  = float(np.mean(np.abs(volume_future - predicted_volume_chronos)))
        rmse_ch = float(np.sqrt(np.mean((volume_future - predicted_volume_chronos)**2)))
        errors_dict['Chronos'] = dict(mae=mae_ch, rmse=rmse_ch)
        print(f"✓ Chronos : MAE={mae_ch:.4f}°C  RMSE={rmse_ch:.4f}°C")
    else:
        print("\n⚠️  Chronos ignoré (--skip-chronos)")

    # ============================================================
    # 4. VISUALISATIONS 3D (inchangées)
    # ============================================================
    print("\n" + "=" * 70)
    print("GÉNÉRATION DES VISUALISATIONS 3D")
    print("=" * 70)

    t_vis = min(10, args.horizon - 1)

    show_3d_volume(volume_context, -1,
                   title=f"Last Context Frame (t={context_len})",
                   filename="last_context_3d",
                   convection_mask=convection_mask)

    show_3d_volume(volume_future, t_vis,
                   title=f"Ground Truth 3D (t={context_len + t_vis})",
                   filename="ground_truth_3d",
                   convection_mask=convection_mask)

    # Figures individuelles par modèle (inchangées)
    show_3d_volume(predicted_volume, t_vis,
                   title=f"TimesFM Prediction 3D (t={context_len + t_vis})",
                   filename="prediction_timesfm_3d",
                   convection_mask=convection_mask)
    if predicted_volume_chronos is not None:
        show_3d_volume(predicted_volume_chronos, t_vis,
                       title=f"Chronos Prediction 3D (t={context_len + t_vis})",
                       filename="prediction_chronos_3d",
                       convection_mask=convection_mask)

    # Figure de comparaison unifiée : Ground Truth + tous les modèles côte à côte
    predictions_dict_vis = {'TimesFM': predicted_volume}
    if predicted_volume_chronos is not None:
        predictions_dict_vis['Chronos'] = predicted_volume_chronos
    compare_all_models_3d(volume_future, predictions_dict_vis, t_vis,
                          filename="comparison_all_models_3d")

    # ============================================================
    # 5. ÉVOLUTION TEMPORELLE (inchangée + Chronos)
    # ============================================================
    print("\n" + "=" * 70)
    print("ÉVOLUTION TEMPORELLE AUX POINTS D'INTÉRÊT")
    print("=" * 70)

    points = [
        (args.nx // 4,     args.ny // 2, args.nz // 2, "Left Quarter"),
        (args.nx // 2,     args.ny // 2, args.nz // 2, "Center"),
        (3*args.nx // 4,   args.ny // 2, args.nz // 2, "Right Quarter"),
    ]
    if disruption:
        points.append((disruption['x'], disruption['y'],
                       disruption['z'], "Disruption Point"))
    if convection_mask is not None:
        conv_points = np.argwhere(convection_mask)
        if len(conv_points) > 0:
            mid = conv_points[len(conv_points) // 2]
            points.append((mid[0], mid[1], mid[2], "Convection Surface"))

    disruption_time = disruption['instant'] if disruption else None
    convection_time = convection['t_start'] if convection else None

    # Dictionnaire des prédictions pour les figures unifiées
    predictions_dict_plots = {'TimesFM': predicted_volume}
    if predicted_volume_chronos is not None:
        predictions_dict_plots['Chronos'] = predicted_volume_chronos

    for x, y, z, label in points:
        plot_temperature_all_models(
            true_vol=volume,
            predictions_dict=predictions_dict_plots,
            context_len=context_len,
            point=(x, y, z),
            title=f"Temperature: {label}",
            filename=f"temp_evo_{label.lower().replace(' ', '_')}",
            disruption_time=disruption_time,
            convection_time=convection_time
        )

    # ============================================================
    # 6. MÉTRIQUES D'ERREUR (inchangées + Chronos)
    # ============================================================
    print("\n" + "=" * 70)
    print("MÉTRIQUES D'ERREUR")
    print("=" * 70)

    print(f"\n{'Modèle':<12} {'MAE (°C)':>12} {'RMSE (°C)':>12}")
    print("-" * 38)
    for m, d in errors_dict.items():
        print(f"{m:<12} {d['mae']:>12.4f} {d['rmse']:>12.4f}")

    # Métriques détaillées TimesFM (inchangées)
    max_error = np.max(np.abs(volume_future - predicted_volume))
    print(f"\nTimesFM — Erreur maximale : {max_error:.4f}°C")

    if convection_mask is not None:
        future_conv = volume_future[:, convection_mask]
        pred_conv   = predicted_volume[:, convection_mask]
        print(f"TimesFM — MAE surface convection : "
              f"{np.mean(np.abs(future_conv - pred_conv)):.4f}°C")
        if predicted_volume_chronos is not None:
            pred_conv_ch = predicted_volume_chronos[:, convection_mask]
            print(f"Chronos — MAE surface convection : "
                  f"{np.mean(np.abs(future_conv - pred_conv_ch)):.4f}°C")

    # Bar chart comparatif
    if len(errors_dict) > 1:
        plot_model_comparison(errors_dict, filename="model_comparison")

    print("\n" + "=" * 70)
    print(f"✓ Tous les résultats sauvegardés dans ./{RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()