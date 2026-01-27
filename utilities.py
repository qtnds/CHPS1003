import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import timesfm
import os

# ============================================================
# DOSSIER RESULTS
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
# SIMULATION PHYSIQUE (DIFFUSION DE LA CHALEUR 3D)
# ============================================================

def create_surface_mask(nx, ny, nz, surface_spec):
    """
    Crée un masque pour une surface de convection.
    
    surface_spec: dict avec 'type' et paramètres
    Types supportés:
    - 'face': {'face': 'top'/'bottom'/'left'/'right'/'front'/'back'}
    - 'rectangle': {'face': str, 'x_range': (x1,x2), 'y_range': (y1,y2) ou 'z_range': (z1,z2)}
    - 'circle': {'face': str, 'center': (cx, cy ou cz), 'radius': r}
    """
    mask = np.zeros((nx, ny, nz), dtype=bool)
    
    surf_type = surface_spec['type']
    face = surface_spec.get('face', 'top')
    
    if surf_type == 'face':
        # Surface complète
        if face == 'top':
            mask[:, :, -1] = True
        elif face == 'bottom':
            mask[:, :, 0] = True
        elif face == 'left':
            mask[0, :, :] = True
        elif face == 'right':
            mask[-1, :, :] = True
        elif face == 'front':
            mask[:, 0, :] = True
        elif face == 'back':
            mask[:, -1, :] = True
    
    elif surf_type == 'rectangle':
        # Surface rectangulaire sur une face
        if face in ['top', 'bottom']:
            z_idx = -1 if face == 'top' else 0
            x1, x2 = surface_spec.get('x_range', (0, nx))
            y1, y2 = surface_spec.get('y_range', (0, ny))
            mask[x1:x2, y1:y2, z_idx] = True
        elif face in ['left', 'right']:
            x_idx = 0 if face == 'left' else -1
            y1, y2 = surface_spec.get('y_range', (0, ny))
            z1, z2 = surface_spec.get('z_range', (0, nz))
            mask[x_idx, y1:y2, z1:z2] = True
        elif face in ['front', 'back']:
            y_idx = 0 if face == 'front' else -1
            x1, x2 = surface_spec.get('x_range', (0, nx))
            z1, z2 = surface_spec.get('z_range', (0, nz))
            mask[x1:x2, y_idx, z1:z2] = True
    
    elif surf_type == 'circle':
        # Surface circulaire sur une face
        center = surface_spec['center']
        radius = surface_spec['radius']
        
        if face in ['top', 'bottom']:
            z_idx = -1 if face == 'top' else 0
            cx, cy = center
            X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
            circle_mask = (X - cx)**2 + (Y - cy)**2 <= radius**2
            mask[:, :, z_idx] = circle_mask
        elif face in ['left', 'right']:
            x_idx = 0 if face == 'left' else -1
            cy, cz = center
            Y, Z = np.meshgrid(np.arange(ny), np.arange(nz), indexing='ij')
            circle_mask = (Y - cy)**2 + (Z - cz)**2 <= radius**2
            mask[x_idx, :, :] = circle_mask
        elif face in ['front', 'back']:
            y_idx = 0 if face == 'front' else -1
            cx, cz = center
            X, Z = np.meshgrid(np.arange(nx), np.arange(nz), indexing='ij')
            circle_mask = (X - cx)**2 + (Z - cz)**2 <= radius**2
            mask[:, y_idx, :] = circle_mask
    
    return mask


def simulate_heat_diffusion_3d(nx=32, ny=32, nz=32, nt=200, 
                              dx=0.01, dy=0.01, dz=0.01, dt=0.1, 
                              material="steel", disruption=None, 
                              convection=None):
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
    convection: dict avec clés:
        - 'T_air': température de l'air [°C]
        - 'h': coefficient de convection [W/(m²·K)]
        - 't_start': instant de démarrage
        - 'surface': dict définissant la surface (face, x_range, y_range, z_range)
    """
    # Récupérer les propriétés thermiques du matériau
    if material not in MATERIALS:
        raise ValueError(f"Matériau '{material}' inconnu. Choix: {list(MATERIALS.keys())}")
    
    alpha = MATERIALS[material]["thermal_diffusivity"]
    k = MATERIALS[material]["thermal_conductivity"]
    rho = MATERIALS[material]["density"]
    cp = MATERIALS[material]["specific_heat"]
    material_name = MATERIALS[material]["name"]
    
    print(f"\nMatériau: {material_name}")
    print(f"Diffusivité thermique α = {alpha:.2e} m²/s")
    print(f"Conductivité thermique k = {k:.2f} W/(m·K)")
    print(f"Dimensions: {nx*dx:.3f}m × {ny*dy:.3f}m × {nz*dz:.3f}m")
    print(f"Pas de temps dt = {dt:.3f}s")
    
    # Vérification de la stabilité (critère CFL pour méthode explicite en 3D)
    dt_max = 1.0 / (2 * alpha * (1/dx**2 + 1/dy**2 + 1/dz**2))
    if dt > dt_max:
        print(f"⚠️  ATTENTION: dt={dt:.3e}s > dt_max={dt_max:.3e}s")
        print(f"   Le schéma peut être instable! Considérez réduire dt.")
    else:
        print(f"✓ Stabilité: dt={dt:.3e}s < dt_max={dt_max:.3e}s")
    
    # Traitement de la convection
    convection_mask = None
    convection_active = False
    if convection is not None:
        T_air = convection['T_air']
        h = convection['h']
        t_start = convection['t_start']
        surface_def = convection['surface']
        
        convection_mask = create_surface_mask(nx, ny, nz, surface_def)
        n_conv_cells = np.sum(convection_mask)
        
        # Calcul du caractère effectif de longueur pour la cellule de surface
        # Pour une cellule de surface, on prend le pas d'espace perpendiculaire
        face = surface_def['face']
        if 'x' in face:
            delta_n = dx
        elif 'y' in face:
            delta_n = dy
        else:  # z
            delta_n = dz
        
        # Nombre de Biot local: Bi = h * delta_n / k
        Bi = h * delta_n / k
        
        print(f"\n🌬️  Convection configurée:")
        print(f"   Température air: {T_air}°C")
        print(f"   Coefficient h: {h} W/(m²·K)")
        print(f"   Démarrage: t = {t_start}")
        print(f"   Surface: face '{surface_def['face']}'")
        print(f"   Nombre de cellules affectées: {n_conv_cells}")
        print(f"   Nombre de Biot: {Bi:.4f}")
        if Bi > 0.1:
            print(f"   ⚠️  Bi > 0.1: résistance interne significative")
        else:
            print(f"   ✓ Bi < 0.1: température de surface quasi-uniforme")
        
        convection_active = True
    
    T = np.zeros((nt, nx, ny, nz), dtype=np.float32)
    
    # Condition initiale: température 0°C partout
    # Source de chaleur: surface droite (x=nx-1) à 100°C
    T[0, -1, :, :] = 100.0
    
    for t in range(nt - 1):
        T_current = T[t].copy()
        laplacian = np.zeros_like(T_current)
        
        # Appliquer la perturbation ponctuelle si on atteint l'instant défini
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
        
        # Faces (6 faces du cube) - Conditions de Neumann par défaut
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
        
        # Mise à jour temporelle avec diffusion
        T[t+1] = T[t] + alpha * dt * laplacian
        
        # Imposer la condition de Dirichlet sur la surface droite (source chaude)
        T[t+1, -1, :, :] = 100.0
        
        # Appliquer la convection si active et après t_start
        if convection_active and t >= t_start:
            # La convection modifie le bilan thermique à la surface
            # Flux convectif: q_conv = h * (T_surface - T_air) [W/m²]
            # Contribution au terme source: -q_conv / (rho * cp * delta_n)
            # Équivalent à ajouter un terme: -(h / (rho * cp * delta_n)) * (T - T_air)
            
            # On applique cette correction sur les cellules de surface
            face = surface_def['face']
            if 'x' in face:
                delta_n = dx
            elif 'y' in face:
                delta_n = dy
            else:
                delta_n = dz
            
            # Coefficient de convection effectif pour mise à jour explicite
            # dT/dt += -(h / (rho * cp * delta_n)) * (T - T_air)
            conv_coeff = h / (rho * cp * delta_n)
            
            # Vérification de stabilité pour la convection
            dt_max_conv = 1.0 / conv_coeff
            if t == t_start:
                if dt > dt_max_conv:
                    print(f"⚠️  Stabilité convection: dt={dt:.3e}s > dt_max_conv={dt_max_conv:.3e}s")
            
            # Application du terme de convection
            T_surface = T[t+1][convection_mask]
            heat_loss = conv_coeff * (T_surface - T_air) * dt
            T[t+1][convection_mask] -= heat_loss
            
            if t == t_start:
                print(f"\n🌬️  Convection démarrée à t={t}")
                print(f"   Température surface moyenne: {np.mean(T_surface):.2f}°C")
                print(f"   Flux convectif moyen: {h * np.mean(T_surface - T_air):.2f} W/m²")
    
    return T



# ============================================================
# CONVERSION VOLUME <-> SÉRIES TEMPORELLES
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
# FORECASTING AVEC TIMESFM
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
# VISUALISATION 3D
# ============================================================

def show_3d_volume(volume, timestep, title="", filename=None, convection_mask=None):
    """Affiche le volume 3D avec coupes et vue 3D"""
    fig = plt.figure(figsize=(18, 5))
    
    mid_x = volume.shape[1] // 2
    mid_z = volume.shape[3] // 2
    
    # Coupe Y-Z
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(volume[timestep, mid_x, :, :].T, cmap="inferno", origin="lower", aspect="auto")
    ax1.set_title(f"Coupe Y-Z (x={mid_x})")
    ax1.set_xlabel("Y")
    ax1.set_ylabel("Z")
    
    # Surimpression du masque de convection si disponible
    if convection_mask is not None:
        conv_slice = convection_mask[mid_x, :, :]
        ax1.contour(conv_slice.T, colors='cyan', linewidths=2, levels=[0.5])
    
    plt.colorbar(im1, ax=ax1, label="°C")
    
    # Coupe X-Y
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(volume[timestep, :, :, mid_z].T, cmap="inferno", origin="lower", aspect="auto")
    ax2.set_title(f"Coupe X-Y (z={mid_z})")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    
    if convection_mask is not None:
        conv_slice = convection_mask[:, :, mid_z]
        ax2.contour(conv_slice.T, colors='cyan', linewidths=2, levels=[0.5])
    
    plt.colorbar(im2, ax=ax2, label="°C")
    
    # Vue 3D
    ax3 = fig.add_subplot(133, projection='3d')
    data = volume[timestep]
    X, Y, Z = data.shape
    
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
    
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(true_vol[timestep, mid_x, :, :].T, cmap="inferno", origin="lower")
    ax1.set_title("Ground Truth (Coupe Y-Z)", fontsize=12)
    ax1.set_xlabel("Y")
    ax1.set_ylabel("Z")
    plt.colorbar(im1, ax=ax1, label="°C")
    
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(pred_vol[timestep, mid_x, :, :].T, cmap="inferno", origin="lower")
    ax2.set_title("TimesFM Prediction (Coupe Y-Z)", fontsize=12)
    ax2.set_xlabel("Y")
    ax2.set_ylabel("Z")
    plt.colorbar(im2, ax=ax2, label="°C")
    
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
                               title="Temperature Evolution", filename="temp_evo",
                               disruption_time=None, convection_time=None):
    """Trace l'évolution de température avec marqueurs d'événements"""
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
    
    plt.figure(figsize=(12, 5))
    
    plt.plot(range(context_len), true_series[:context_len], 
             'b-', linewidth=2, label="Historical Context", alpha=0.7)
    
    plt.plot(range(context_len - 1, T_true), true_series[context_len - 1:], 
             'g-', linewidth=2, label="True Future")
    
    if T_pred > 0:
        plt.plot(range(context_len, context_len + T_pred), pred_series, 
                 'r--', linewidth=2, label="TimesFM Prediction")
    
    plt.axvline(x=context_len, color='gray', linestyle=':', 
                linewidth=1.5, label="Forecast Start")
    
    # Marquer les événements
    if disruption_time is not None:
        plt.axvline(x=disruption_time, color='orange', linestyle='--', 
                   linewidth=1.5, label="Disruption", alpha=0.7)
    
    if convection_time is not None:
        plt.axvline(x=convection_time, color='cyan', linestyle='--', 
                   linewidth=1.5, label="Convection Start", alpha=0.7)
    
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.title(f"{title} at Point ({x}, {y}, {z})", fontsize=14)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    if filename is None:
        filename = title.replace(" ", "_").lower()
    
    path = os.path.join(RESULTS_DIR, f"{filename}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")