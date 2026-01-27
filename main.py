import argparse
import numpy as np
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
    
    Exemples:
    - Face complète: [20,10,face,50,top]
    - Rectangle: [20,10,rectangle,50,top,5,20,5,20]  (face,x1,x2,y1,y2)
    - Cercle: [20,10,circle,50,top,12,12,5]  (face,cx,cy,radius)
    """
    try:
        convection_str = convection_str.strip('[]')
        parts = convection_str.split(',')
        
        if len(parts) < 4:
            raise ValueError("Convection doit avoir au moins 4 paramètres de base")
        
        T_air = float(parts[0])
        h = float(parts[1])
        surface_type = parts[2].strip()
        t_start = int(parts[3])
        
        # Parser les paramètres de surface
        if surface_type == 'face':
            if len(parts) < 5:
                raise ValueError("Type 'face' nécessite: [T_air,h,face,t_start,face_name]")
            face_name = parts[4].strip()
            surface_spec = {
                'type': 'face',
                'face': face_name
            }
        
        elif surface_type == 'rectangle':
            if len(parts) < 9:
                raise ValueError("Type 'rectangle' nécessite 9 paramètres")
            face_name = parts[4].strip()
            
            if face_name in ['top', 'bottom']:
                x1, x2 = int(parts[5]), int(parts[6])
                y1, y2 = int(parts[7]), int(parts[8])
                surface_spec = {
                    'type': 'rectangle',
                    'face': face_name,
                    'x_range': (x1, x2),
                    'y_range': (y1, y2)
                }
            elif face_name in ['left', 'right']:
                y1, y2 = int(parts[5]), int(parts[6])
                z1, z2 = int(parts[7]), int(parts[8])
                surface_spec = {
                    'type': 'rectangle',
                    'face': face_name,
                    'y_range': (y1, y2),
                    'z_range': (z1, z2)
                }
            else:  # front, back
                x1, x2 = int(parts[5]), int(parts[6])
                z1, z2 = int(parts[7]), int(parts[8])
                surface_spec = {
                    'type': 'rectangle',
                    'face': face_name,
                    'x_range': (x1, x2),
                    'z_range': (z1, z2)
                }
        
        elif surface_type == 'circle':
            if len(parts) < 8:
                raise ValueError("Type 'circle' nécessite 8 paramètres")
            face_name = parts[4].strip()
            c1 = int(parts[5])
            c2 = int(parts[6])
            radius = float(parts[7])
            
            surface_spec = {
                'type': 'circle',
                'face': face_name,
                'center': (c1, c2),
                'radius': radius
            }
        
        else:
            raise ValueError(f"Type de surface '{surface_type}' non reconnu")
        
        return {
            'T_air': T_air,
            'h': h,
            'surface': surface_spec,
            't_start': t_start
        }
    
    except Exception as e:
        raise ValueError(f"Erreur de parsing de --convection: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Simulation 3D de diffusion thermique avec perturbation et convection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Paramètres de simulation
    parser.add_argument("--total", type=int, default=200,
                       help="Nombre total de pas de temps")
    parser.add_argument("--horizon", type=int, default=50,
                       help="Horizon de prédiction TimesFM")
    parser.add_argument("--material", type=str, default="steel",
                       choices=list(MATERIALS.keys()),
                       help="Matériau de la plaque")
    
    # Paramètres de grille
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
    
    # Perturbations
    parser.add_argument("--disruption", type=str, default=None,
                       help="Perturbation ponctuelle: [x,y,z,temp,instant]")
    parser.add_argument("--convection", type=str, default=None,
                       help="Convection: [T_air,h,type,t_start,params...]. "
                            "Ex: [20,10,face,50,top] ou [20,10,circle,50,top,12,12,5]")
    
    args = parser.parse_args()
    
    # Validation
    if args.horizon >= args.total:
        raise ValueError(f"L'horizon ({args.horizon}) doit être < total ({args.total})")
    
    context_len = args.total - args.horizon
    
    # Parser la perturbation
    disruption = None
    if args.disruption:
        disruption = parse_disruption(args.disruption)
        
        # Validation
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
    
    # Parser la convection
    convection = None
    convection_mask = None
    if args.convection:
        convection = parse_convection(args.convection, args.nx, args.ny, args.nz)
        
        # Validation
        if not (0 <= convection['t_start'] < args.total):
            raise ValueError(f"t_start doit être entre 0 et {args.total-1}")
        
        # Créer le masque pour la visualisation
        convection_mask = create_surface_mask(args.nx, args.ny, args.nz, convection['surface'])
        
        print(f"\n🌬️  Convection d'air:")
        print(f"   Température air: {convection['T_air']}°C")
        print(f"   Coefficient h: {convection['h']} W/(m²·K)")
        print(f"   Type: {convection['surface']['type']}")
        print(f"   Face: {convection['surface'].get('face', 'N/A')}")
        print(f"   Surface: {np.sum(convection_mask)} points")
        print(f"   Début: t={convection['t_start']}")
    
    # Affichage des paramètres
    print("\n" + "=" * 70)
    print("SIMULATION 3D DE DIFFUSION THERMIQUE AVEC TIMESFM")
    print("=" * 70)
    print(f"\nParamètres de simulation:")
    print(f"  Total: {args.total} pas de temps")
    print(f"  Contexte: {context_len} pas")
    print(f"  Horizon: {args.horizon} pas")
    print(f"  Grille: {args.nx} × {args.ny} × {args.nz}")
    print(f"  Pas spatiaux: dx={args.dx}m, dy={args.dy}m, dz={args.dz}m")
    print(f"  Pas temporel: dt={args.dt}s")
    print(f"  Matériau: {args.material}")
    
    # ============================================================
    # 1. SIMULATION 3D
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
    
    # ============================================================
    # 2. SÉPARATION CONTEXTE/FUTUR
    # ============================================================
    volume_context = volume[:context_len]
    volume_future = volume[context_len:context_len + args.horizon]
    
    # ============================================================
    # 3. CONVERSION EN SÉRIES TEMPORELLES
    # ============================================================
    print("\n" + "=" * 70)
    print("PRÉPARATION DES DONNÉES POUR TIMESFM")
    print("=" * 70)
    
    timeseries_context = volume_to_timeseries(volume_context)
    print(f"Séries temporelles: {timeseries_context.shape[0]} (une par voxel)")
    print(f"Longueur: {timeseries_context.shape[1]} pas")
    
    # ============================================================
    # 4. PRÉDICTION AVEC TIMESFM
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
    
    # ============================================================
    # 5. RECONSTRUCTION DU VOLUME
    # ============================================================
    predicted_volume = timeseries_to_volume(
        point_forecast,
        spatial_shape=(args.nx, args.ny, args.nz)
    )
    
    print(f"Volume prédit: {predicted_volume.shape}")
    
    # ============================================================
    # 6. VISUALISATION 3D
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
    
    show_3d_volume(predicted_volume, t_vis,
                   title=f"TimesFM Prediction 3D (t={context_len + t_vis})",
                   filename="prediction_3d",
                   convection_mask=convection_mask)
    
    compare_true_vs_pred_3d(volume_future, predicted_volume, t_vis,
                           filename="comparison_3d")
    
    # ============================================================
    # 7. ÉVOLUTION TEMPORELLE AUX POINTS D'INTÉRÊT
    # ============================================================
    print("\n" + "=" * 70)
    print("ÉVOLUTION TEMPORELLE AUX POINTS D'INTÉRÊT")
    print("=" * 70)
    
    points = [
        (args.nx // 4, args.ny // 2, args.nz // 2, "Left Quarter"),
        (args.nx // 2, args.ny // 2, args.nz // 2, "Center"),
        (3 * args.nx // 4, args.ny // 2, args.nz // 2, "Right Quarter"),
    ]
    
    # Ajouter le point de perturbation s'il existe
    if disruption:
        points.append((
            disruption['x'],
            disruption['y'],
            disruption['z'],
            "Disruption Point"
        ))
    
    # Ajouter un point sur la surface de convection si elle existe
    if convection_mask is not None:
        conv_points = np.argwhere(convection_mask)
        if len(conv_points) > 0:
            # Prendre le point au milieu de la surface
            mid_point = conv_points[len(conv_points) // 2]
            points.append((
                mid_point[0],
                mid_point[1],
                mid_point[2],
                "Convection Surface"
            ))
    
    # Marquer les temps des événements
    disruption_time = disruption['instant'] if disruption else None
    convection_time = convection['t_start'] if convection else None
    
    for x, y, z, label in points:
        plot_temperature_at_point(
            true_vol=volume,
            pred_vol=predicted_volume,
            context_len=context_len,
            point=(x, y, z),
            title=f"Temperature: {label}",
            filename=f"temp_evo_{label.lower().replace(' ', '_')}",
            disruption_time=disruption_time,
            convection_time=convection_time
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
    
    # Métriques spécifiques pour la zone de convection
    if convection_mask is not None:
        # Extraire uniquement les points de la surface de convection
        future_conv = volume_future[:, convection_mask]
        pred_conv = predicted_volume[:, convection_mask]
        
        mae_conv = np.mean(np.abs(future_conv - pred_conv))
        rmse_conv = np.sqrt(np.mean((future_conv - pred_conv)**2))
        
        print(f"\nErreur sur la surface de convection:")
        print(f"  MAE: {mae_conv:.4f}°C")
        print(f"  RMSE: {rmse_conv:.4f}°C")
    
    print("\n" + "=" * 70)
    print(f"✓ Tous les résultats sauvegardés dans ./{RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()