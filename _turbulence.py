import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
import timesfm
import os
import argparse
import time
from statsmodels.tsa.arima.model import ARIMA
from scipy.ndimage import gaussian_filter

# ============================================================
# 0. CONFIGURATION
# ============================================================

RESULTS_DIR = "results_turbulence"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 1. SIMULATION RAYLEIGH-BÉNARD 3D
# ============================================================

class RayleighBenardSimulator:
    """
    Simulateur de convection de Rayleigh-Bénard 3D
    
    Équations couplées:
    - Équation de Navier-Stokes pour la vitesse
    - Équation de transport pour la température
    - Approximation de Boussinesq pour la flottabilité
    """
    
    def __init__(self, nx=32, ny=32, nz=32, dx=0.1, dy=0.1, dz=0.1, dt=0.001,
                 Ra=1000, Pr=0.71, T_hot=100, T_cold=20):
        """
        Parameters:
        -----------
        nx, ny, nz : dimensions de la grille
        dx, dy, dz : pas spatiaux [m]
        dt : pas de temps [s]
        Ra : nombre de Rayleigh (contrôle l'intensité de la convection)
        Pr : nombre de Prandtl (rapport viscosité/diffusivité thermique)
        T_hot : température de la plaque inférieure [°C]
        T_cold : température de la plaque supérieure [°C]
        """
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.dt = dt
        self.Ra = Ra
        self.Pr = Pr
        self.T_hot = T_hot
        self.T_cold = T_cold
        self.dT = T_hot - T_cold
        
        # Initialisation des champs
        self.T = np.zeros((nx, ny, nz), dtype=np.float32)
        self.vx = np.zeros((nx, ny, nz), dtype=np.float32)
        self.vy = np.zeros((nx, ny, nz), dtype=np.float32)
        self.vz = np.zeros((nx, ny, nz), dtype=np.float32)
        
        # Conditions initiales : gradient linéaire + perturbations aléatoires
        for k in range(nz):
            self.T[:, :, k] = T_hot - (T_hot - T_cold) * k / (nz - 1)
        
        # Ajouter des perturbations pour déclencher la convection
        noise = np.random.randn(nx, ny, nz) * self.dT * 0.01
        self.T += gaussian_filter(noise, sigma=2.0)
        
        # Imposer les conditions aux limites
        self.T[:, :, 0] = T_hot    # Plaque chaude en bas
        self.T[:, :, -1] = T_cold  # Plaque froide en haut
        
        print(f"\n🌊 Simulateur Rayleigh-Bénard initialisé:")
        print(f"   Grille: {nx}×{ny}×{nz}")
        print(f"   Nombre de Rayleigh: {Ra:.1f}")
        print(f"   Nombre de Prandtl: {Pr:.2f}")
        print(f"   ΔT = {self.dT}°C (chaud: {T_hot}°C, froid: {T_cold}°C)")
    
    def step(self):
        """Un pas de temps de la simulation"""
        
        # Coefficients effectifs
        alpha = 1.0 / (self.Ra * self.Pr)  # Diffusivité thermique
        nu = self.Pr / self.Ra             # Viscosité cinématique
        g = 1.0                             # Gravité (normalisée)
        
        # 1. Calcul des forces de flottabilité (approximation de Boussinesq)
        # F_z = g * (T - T_mean) / dT
        T_mean = (self.T_hot + self.T_cold) / 2
        buoyancy = g * (self.T - T_mean) / self.dT
        
        # 2. Mise à jour de la vitesse verticale (équation de Navier-Stokes simplifiée)
        # ∂vz/∂t = -vx*∂vz/∂x - vy*∂vz/∂y - vz*∂vz/∂z + ν*∇²vz + buoyancy
        
        # Termes advectifs (simplifiés)
        dvz_dx = np.gradient(self.vz, self.dx, axis=0)
        dvz_dy = np.gradient(self.vz, self.dy, axis=1)
        dvz_dz = np.gradient(self.vz, self.dz, axis=2)
        
        advection_vz = -(self.vx * dvz_dx + self.vy * dvz_dy + self.vz * dvz_dz)
        
        # Laplacien de vz
        laplacian_vz = (
            np.gradient(np.gradient(self.vz, self.dx, axis=0), self.dx, axis=0) +
            np.gradient(np.gradient(self.vz, self.dy, axis=1), self.dy, axis=1) +
            np.gradient(np.gradient(self.vz, self.dz, axis=2), self.dz, axis=2)
        )
        
        # Mise à jour vz
        self.vz += self.dt * (advection_vz + nu * laplacian_vz + buoyancy)
        
        # Conditions aux limites pour vz (pas de glissement)
        self.vz[:, :, 0] = 0
        self.vz[:, :, -1] = 0
        self.vz[0, :, :] = 0
        self.vz[-1, :, :] = 0
        self.vz[:, 0, :] = 0
        self.vz[:, -1, :] = 0
        
        # 3. Mise à jour de la température (équation de transport)
        # ∂T/∂t = -vx*∂T/∂x - vy*∂T/∂y - vz*∂T/∂z + α*∇²T
        
        dT_dx = np.gradient(self.T, self.dx, axis=0)
        dT_dy = np.gradient(self.T, self.dy, axis=1)
        dT_dz = np.gradient(self.T, self.dz, axis=2)
        
        advection_T = -(self.vx * dT_dx + self.vy * dT_dy + self.vz * dT_dz)
        
        laplacian_T = (
            np.gradient(np.gradient(self.T, self.dx, axis=0), self.dx, axis=0) +
            np.gradient(np.gradient(self.T, self.dy, axis=1), self.dy, axis=1) +
            np.gradient(np.gradient(self.T, self.dz, axis=2), self.dz, axis=2)
        )
        
        self.T += self.dt * (advection_T + alpha * laplacian_T)
        
        # Conditions aux limites pour T (Dirichlet)
        self.T[:, :, 0] = self.T_hot
        self.T[:, :, -1] = self.T_cold
        
        # Générer des vitesses horizontales dues à la convection (simplifié)
        # Les panaches chauds montent, créant des mouvements horizontaux
        self.vx = 0.1 * np.gradient(self.T, self.dx, axis=0)
        self.vy = 0.1 * np.gradient(self.T, self.dy, axis=1)
        
        return self.T.copy()
    
    def simulate(self, nt):
        """Simule nt pas de temps et retourne l'historique de température"""
        T_history = np.zeros((nt, self.nx, self.ny, self.nz), dtype=np.float32)
        
        print(f"\n⏳ Simulation Rayleigh-Bénard en cours ({nt} pas de temps)...")
        for t in range(nt):
            T_history[t] = self.step()
            
            if (t + 1) % 50 == 0:
                print(f"   Pas {t+1}/{nt} - T_min: {self.T.min():.2f}°C, "
                      f"T_max: {self.T.max():.2f}°C, "
                      f"Vitesse max: {np.abs(self.vz).max():.3f} m/s")
        
        print("✓ Simulation terminée!")
        return T_history


# ============================================================
# 2. MODÈLES DE PRÉDICTION
# ============================================================

class LSTMForecaster(nn.Module):
    """Modèle LSTM pour prédiction de séries temporelles"""
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Prendre la dernière sortie
        return out


def train_lstm(timeseries, horizon, epochs=50, batch_size=32, device='cpu'):
    """Entraîne un LSTM sur les séries temporelles"""
    N, T = timeseries.shape
    
    # Préparer les données d'entraînement
    X_train = torch.FloatTensor(timeseries[:, :-horizon]).unsqueeze(-1).to(device)
    
    model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n🧠 Entraînement LSTM ({epochs} epochs)...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, N, batch_size):
            batch = X_train[i:i+batch_size]
            
            # Prédire le prochain pas
            optimizer.zero_grad()
            pred = model(batch)
            
            # Target: prochain pas de temps
            target = torch.FloatTensor(timeseries[i:i+batch_size, -horizon]).unsqueeze(-1).to(device)
            
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs}, Loss: {total_loss/N:.6f}")
    
    return model


def forecast_lstm(model, timeseries, horizon, device='cpu'):
    """Prédiction avec LSTM"""
    model.eval()
    N, T = timeseries.shape
    predictions = np.zeros((N, horizon), dtype=np.float32)
    
    with torch.no_grad():
        for i in range(N):
            context = timeseries[i:i+1, :]
            pred_series = []
            
            for h in range(horizon):
                x = torch.FloatTensor(context).unsqueeze(-1).to(device)
                pred = model(x)
                pred_val = pred.cpu().numpy()[0, 0]
                pred_series.append(pred_val)
                
                # Mettre à jour le contexte (autorégressif)
                context = np.append(context[0, 1:], pred_val).reshape(1, -1)
            
            predictions[i] = np.array(pred_series)
    
    return predictions


def forecast_arima(timeseries, horizon):
    """Prédiction avec ARIMA (baseline classique)"""
    N, T = timeseries.shape
    predictions = np.zeros((N, horizon), dtype=np.float32)
    
    print(f"\n📊 Prédiction ARIMA en cours ({N} séries)...")
    
    for i in range(N):
        try:
            # Modèle ARIMA(1,0,0) - plus simple et plus stable
            model = ARIMA(timeseries[i], order=(1, 0, 0))
            fitted = model.fit(method='yule_walker')  # Méthode plus robuste
            forecast = fitted.forecast(steps=horizon)
            predictions[i] = forecast
        except Exception as e:
            # Si échec, utiliser une approche plus simple
            try:
                # Essayer un modèle AR simple
                model = ARIMA(timeseries[i], order=(1, 0, 0))
                fitted = model.fit(method='yule_walker')
                forecast = fitted.forecast(steps=horizon)
                predictions[i] = forecast
            except:
                # En dernier recours, prédire la dernière valeur
                predictions[i] = timeseries[i, -1]
        
        if (i + 1) % 1000 == 0:
            print(f"   [{i+1}/{N}] séries traitées")
    
    print("✓ ARIMA terminé!")
    return predictions


# ============================================================
# 3. CONVERSION ET FORECASTING
# ============================================================

def volume_to_timeseries(volume):
    """Convertit (T, X, Y, Z) -> (N, T)"""
    T, X, Y, Z = volume.shape
    return volume.reshape(T, X * Y * Z).T


def timeseries_to_volume(timeseries, spatial_shape):
    """Convertit (N, T) -> (T, X, Y, Z)"""
    N, T = timeseries.shape
    X, Y, Z = spatial_shape
    return timeseries.T.reshape(T, X, Y, Z)


class TimesFMForecaster:
    def __init__(self, max_context=512, max_horizon=128):
        torch.set_float32_matmul_precision("high")
        
        print("\n🤖 Chargement du modèle TimesFM...")
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
        print("✓ TimesFM chargé et compilé")
    
    def forecast(self, timeseries, horizon, batch_size=64):
        N = len(timeseries)
        point_all = []
        
        print(f"\n🔮 Prédiction TimesFM ({N} séries)...")
        for i in range(0, N, batch_size):
            batch = timeseries[i:i + batch_size]
            inputs = [series.astype(np.float32) for series in batch]
            
            points, _ = self.model.forecast(horizon=horizon, inputs=inputs)
            point_all.append(points)
            
            if (i + batch_size) % 512 == 0:
                print(f"   [{min(i + batch_size, N)}/{N}] séries traitées")
        
        print("✓ TimesFM terminé!")
        return np.vstack(point_all)


# ============================================================
# 4. VISUALISATION 3D
# ============================================================

def visualize_3d_field(volume, timestep, title="", filename=None, vmin=None, vmax=None):
    """Visualise un champ 3D avec coupes et isosurfaces"""
    fig = plt.figure(figsize=(18, 5))
    
    data = volume[timestep]
    nx, ny, nz = data.shape
    
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    
    # Coupe verticale centrale (plan X-Z)
    ax1 = fig.add_subplot(131)
    mid_y = ny // 2
    im1 = ax1.imshow(data[:, mid_y, :].T, cmap='RdBu_r', origin='lower', 
                     aspect='auto', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Coupe X-Z (y={mid_y})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z (hauteur)')
    plt.colorbar(im1, ax=ax1, label='Température (°C)')
    
    # Coupe horizontale (plan X-Y)
    ax2 = fig.add_subplot(132)
    mid_z = nz // 2
    im2 = ax2.imshow(data[:, :, mid_z].T, cmap='RdBu_r', origin='lower',
                     aspect='auto', vmin=vmin, vmax=vmax)
    ax2.set_title(f'Coupe X-Y (z={mid_z})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2, label='Température (°C)')
    
    # Vue 3D avec points colorés
    ax3 = fig.add_subplot(133, projection='3d')
    
    stride = max(1, nx // 12)
    x, y, z = np.meshgrid(
        np.arange(0, nx, stride),
        np.arange(0, ny, stride),
        np.arange(0, nz, stride),
        indexing='ij'
    )
    
    temps = data[::stride, ::stride, ::stride].flatten()
    norm = Normalize(vmin=vmin, vmax=vmax)
    colors = cm.RdBu_r(norm(temps))
    
    ax3.scatter(x.flatten(), y.flatten(), z.flatten(),
               c=colors, marker='o', s=15, alpha=0.6)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z (hauteur)')
    ax3.set_title('Vue 3D')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if filename:
        path = os.path.join(RESULTS_DIR, f"{filename}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Sauvegardé: {filename}.png")
    else:
        plt.show()


def compare_predictions(true_vol, pred_timesfm, pred_arima, pred_lstm, 
                       timestep, filename="comparison"):
    """Compare les 4 méthodes"""
    fig = plt.figure(figsize=(20, 5))
    
    mid_x = true_vol.shape[1] // 2
    
    vmin = min(true_vol[timestep].min(), pred_timesfm[timestep].min(),
               pred_arima[timestep].min(), pred_lstm[timestep].min())
    vmax = max(true_vol[timestep].max(), pred_timesfm[timestep].max(),
               pred_arima[timestep].max(), pred_lstm[timestep].max())
    
    titles = ['Ground Truth', 'TimesFM', 'ARIMA', 'LSTM']
    volumes = [true_vol, pred_timesfm, pred_arima, pred_lstm]
    
    for i, (vol, title) in enumerate(zip(volumes, titles)):
        ax = fig.add_subplot(1, 4, i+1)
        im = ax.imshow(vol[timestep, mid_x, :, :].T, cmap='RdBu_r', 
                      origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        plt.colorbar(im, ax=ax, label='°C')
    
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{filename}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Sauvegardé: {filename}.png")


def plot_errors_comparison(errors_dict, filename="errors_comparison"):
    """Graphique de comparaison des erreurs"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = list(errors_dict.keys())
    mae_vals = [errors_dict[m]['mae'] for m in models]
    rmse_vals = [errors_dict[m]['rmse'] for m in models]
    times = [errors_dict[m]['time'] for m in models]
    
    # MAE
    axes[0, 0].bar(models, mae_vals, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    axes[0, 0].set_ylabel('MAE (°C)', fontsize=12)
    axes[0, 0].set_title('Mean Absolute Error', fontsize=13, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # RMSE
    axes[0, 1].bar(models, rmse_vals, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    axes[0, 1].set_ylabel('RMSE (°C)', fontsize=12)
    axes[0, 1].set_title('Root Mean Square Error', fontsize=13, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Temps d'exécution
    axes[1, 0].bar(models, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    axes[1, 0].set_ylabel('Temps (s)', fontsize=12)
    axes[1, 0].set_title('Temps d\'exécution', fontsize=13, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Tableau récapitulatif
    axes[1, 1].axis('off')
    table_data = []
    for model in models:
        table_data.append([
            model,
            f"{errors_dict[model]['mae']:.4f}",
            f"{errors_dict[model]['rmse']:.4f}",
            f"{errors_dict[model]['time']:.2f}s"
        ])
    
    table = axes[1, 1].table(cellText=table_data,
                            colLabels=['Modèle', 'MAE', 'RMSE', 'Temps'],
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style du tableau
    for i in range(len(table_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor(['#FFE5E5', '#E5F9F9', '#E5F2F9', '#FFE9DD'][i-1])
    
    plt.suptitle('Comparaison des performances des modèles', 
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    path = os.path.join(RESULTS_DIR, f"{filename}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Sauvegardé: {filename}.png")


# ============================================================
# 5. SCRIPT PRINCIPAL
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Simulation Rayleigh-Bénard avec comparaison de modèles de prédiction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--nx", type=int, default=24,
                       help="Points de grille en x")
    parser.add_argument("--ny", type=int, default=24,
                       help="Points de grille en y")
    parser.add_argument("--nz", type=int, default=24,
                       help="Points de grille en z (hauteur)")
    parser.add_argument("--total", type=int, default=300,
                       help="Nombre total de pas de temps")
    parser.add_argument("--horizon", type=int, default=50,
                       help="Horizon de prédiction")
    parser.add_argument("--Ra", type=float, default=1000,
                       help="Nombre de Rayleigh")
    parser.add_argument("--Pr", type=float, default=0.71,
                       help="Nombre de Prandtl")
    parser.add_argument("--skip-arima", action="store_true",
                       help="Sauter ARIMA (très lent)")
    parser.add_argument("--lstm-epochs", type=int, default=30,
                       help="Nombre d'epochs pour LSTM")
    
    args = parser.parse_args()
    
    if args.horizon >= args.total:
        raise ValueError(f"L'horizon ({args.horizon}) doit être < total ({args.total})")
    
    context_len = args.total - args.horizon
    
    print("=" * 70)
    print("SIMULATION RAYLEIGH-BÉNARD 3D")
    print("Comparaison: TimesFM vs ARIMA vs LSTM vs Exact")
    print("=" * 70)
    print(f"\nParamètres:")
    print(f"  Grille: {args.nx}×{args.ny}×{args.nz}")
    print(f"  Total: {args.total} pas, Contexte: {context_len}, Horizon: {args.horizon}")
    print(f"  Ra = {args.Ra}, Pr = {args.Pr}")
    
    # ============================================================
    # 1. SIMULATION EXACTE
    # ============================================================
    print("\n" + "=" * 70)
    print("SIMULATION PHYSIQUE (Solution exacte)")
    print("=" * 70)
    
    simulator = RayleighBenardSimulator(
        nx=args.nx, ny=args.ny, nz=args.nz,
        dx=0.1, dy=0.1, dz=0.1, dt=0.001,
        Ra=args.Ra, Pr=args.Pr,
        T_hot=100, T_cold=20
    )
    
    start_sim = time.time()
    volume_full = simulator.simulate(args.total)
    time_simulation = time.time() - start_sim
    
    print(f"\n✓ Simulation terminée en {time_simulation:.2f}s")
    print(f"  Shape: {volume_full.shape}")
    
    # Visualiser quelques pas de temps
    print("\n📊 Génération des visualisations de la simulation...")
    for t_idx, t_name in [(0, "initial"), (context_len//2, "mid_context"), 
                           (context_len-1, "last_context")]:
        visualize_3d_field(volume_full, t_idx,
                          title=f"Rayleigh-Bénard à t={t_idx}",
                          filename=f"simulation_{t_name}")
    
    # Séparer contexte/futur
    volume_context = volume_full[:context_len]
    volume_future = volume_full[context_len:context_len + args.horizon]
    
    # Conversion en séries temporelles
    timeseries_context = volume_to_timeseries(volume_context)
    print(f"\n📈 Séries temporelles: {timeseries_context.shape[0]} séries de longueur {timeseries_context.shape[1]}")
    
    # ============================================================
    # 2. PRÉDICTIONS AVEC LES DIFFÉRENTS MODÈLES
    # ============================================================
    errors_dict = {}
    spatial_shape = (args.nx, args.ny, args.nz)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    
    # --- TimesFM ---
    print("\n" + "=" * 70)
    print("PRÉDICTION TIMESFM")
    print("=" * 70)
    
    forecaster_timesfm = TimesFMForecaster(max_context=context_len, max_horizon=args.horizon)
    start_timesfm = time.time()
    pred_timesfm = forecaster_timesfm.forecast(timeseries_context, args.horizon, batch_size=32)
    time_timesfm = time.time() - start_timesfm
    
    vol_pred_timesfm = timeseries_to_volume(pred_timesfm, spatial_shape)
    
    mae_timesfm = np.mean(np.abs(volume_future - vol_pred_timesfm))
    rmse_timesfm = np.sqrt(np.mean((volume_future - vol_pred_timesfm)**2))
    
    errors_dict['TimesFM'] = {
        'mae': mae_timesfm,
        'rmse': rmse_timesfm,
        'time': time_timesfm
    }
    
    print(f"\n✓ TimesFM: MAE={mae_timesfm:.4f}°C, RMSE={rmse_timesfm:.4f}°C, Temps={time_timesfm:.2f}s")
    
    # --- ARIMA ---
    if not args.skip_arima:
        print("\n" + "=" * 70)
        print("PRÉDICTION ARIMA (Baseline)")
        print("=" * 70)
        
        start_arima = time.time()
        pred_arima = forecast_arima(timeseries_context, args.horizon)
        time_arima = time.time() - start_arima
        
        vol_pred_arima = timeseries_to_volume(pred_arima, spatial_shape)
        
        mae_arima = np.mean(np.abs(volume_future - vol_pred_arima))
        rmse_arima = np.sqrt(np.mean((volume_future - vol_pred_arima)**2))
        
        errors_dict['ARIMA'] = {
            'mae': mae_arima,
            'rmse': rmse_arima,
            'time': time_arima
        }
        
        print(f"\n✓ ARIMA: MAE={mae_arima:.4f}°C, RMSE={rmse_arima:.4f}°C, Temps={time_arima:.2f}s")
    else:
        print("\n⚠️  ARIMA ignoré (--skip-arima)")
        vol_pred_arima = None
    
    # --- LSTM ---
    print("\n" + "=" * 70)
    print("PRÉDICTION LSTM")
    print("=" * 70)
    
    start_lstm_train = time.time()
    lstm_model = train_lstm(timeseries_context, args.horizon, 
                           epochs=args.lstm_epochs, batch_size=64, device=device)
    time_lstm_train = time.time() - start_lstm_train
    
    print(f"\n✓ Entraînement LSTM terminé en {time_lstm_train:.2f}s")
    
    start_lstm_pred = time.time()
    pred_lstm = forecast_lstm(lstm_model, timeseries_context, args.horizon, device=device)
    time_lstm_pred = time.time() - start_lstm_pred
    time_lstm_total = time_lstm_train + time_lstm_pred
    
    vol_pred_lstm = timeseries_to_volume(pred_lstm, spatial_shape)
    
    mae_lstm = np.mean(np.abs(volume_future - vol_pred_lstm))
    rmse_lstm = np.sqrt(np.mean((volume_future - vol_pred_lstm)**2))
    
    errors_dict['LSTM'] = {
        'mae': mae_lstm,
        'rmse': rmse_lstm,
        'time': time_lstm_total
    }
    
    print(f"\n✓ LSTM: MAE={mae_lstm:.4f}°C, RMSE={rmse_lstm:.4f}°C, "
          f"Temps={time_lstm_total:.2f}s (train: {time_lstm_train:.2f}s)")
    
    # ============================================================
    # 3. VISUALISATION DES RÉSULTATS
    # ============================================================
    print("\n" + "=" * 70)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("=" * 70)
    
    # Visualiser quelques pas de temps de chaque prédiction
    t_vis = min(10, args.horizon - 1)
    
    print("\n📊 Visualisation des prédictions...")
    visualize_3d_field(volume_future, t_vis,
                      title=f"Ground Truth (t={context_len + t_vis})",
                      filename="ground_truth")
    
    visualize_3d_field(vol_pred_timesfm, t_vis,
                      title=f"TimesFM Prediction (t={context_len + t_vis})",
                      filename="pred_timesfm")
    
    if vol_pred_arima is not None:
        visualize_3d_field(vol_pred_arima, t_vis,
                          title=f"ARIMA Prediction (t={context_len + t_vis})",
                          filename="pred_arima")
    
    visualize_3d_field(vol_pred_lstm, t_vis,
                      title=f"LSTM Prediction (t={context_len + t_vis})",
                      filename="pred_lstm")
    
    # Comparaison côte à côte
    print("\n📊 Comparaison visuelle des modèles...")
    if vol_pred_arima is not None:
        compare_predictions(volume_future, vol_pred_timesfm, 
                          vol_pred_arima, vol_pred_lstm,
                          timestep=t_vis, filename="comparison_all")
    else:
        # Version sans ARIMA
        fig = plt.figure(figsize=(15, 5))
        mid_x = volume_future.shape[1] // 2
        
        vmin = min(volume_future[t_vis].min(), vol_pred_timesfm[t_vis].min(), 
                   vol_pred_lstm[t_vis].min())
        vmax = max(volume_future[t_vis].max(), vol_pred_timesfm[t_vis].max(),
                   vol_pred_lstm[t_vis].max())
        
        for i, (vol, title) in enumerate([(volume_future, 'Ground Truth'),
                                           (vol_pred_timesfm, 'TimesFM'),
                                           (vol_pred_lstm, 'LSTM')]):
            ax = fig.add_subplot(1, 3, i+1)
            im = ax.imshow(vol[t_vis, mid_x, :, :].T, cmap='RdBu_r',
                          origin='lower', vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
            plt.colorbar(im, ax=ax, label='°C')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "comparison_all.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("   Sauvegardé: comparison_all.png")
    
    # Graphiques de comparaison des erreurs
    print("\n📊 Graphiques de performance...")
    plot_errors_comparison(errors_dict, filename="performance_comparison")
    
    # Évolution temporelle en un point
    print("\n📊 Évolution temporelle en points clés...")
    point_center = (args.nx // 2, args.ny // 2, args.nz // 2)
    point_bottom = (args.nx // 2, args.ny // 2, 2)
    point_top = (args.nx // 2, args.ny // 2, args.nz - 3)
    
    for point, label in [(point_center, "center"), 
                         (point_bottom, "near_hot"),
                         (point_top, "near_cold")]:
        x, y, z = point
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Température vraie
        true_series = volume_full[:, x, y, z]
        ax.plot(range(len(true_series)), true_series, 
               'k-', linewidth=2, label='Ground Truth', alpha=0.8)
        
        # Prédictions
        pred_timesfm_series = vol_pred_timesfm[:, x, y, z]
        ax.plot(range(context_len, context_len + args.horizon), 
               pred_timesfm_series, 
               'r--', linewidth=2, label='TimesFM', alpha=0.8)
        
        if vol_pred_arima is not None:
            pred_arima_series = vol_pred_arima[:, x, y, z]
            ax.plot(range(context_len, context_len + args.horizon),
                   pred_arima_series,
                   'g--', linewidth=2, label='ARIMA', alpha=0.8)
        
        pred_lstm_series = vol_pred_lstm[:, x, y, z]
        ax.plot(range(context_len, context_len + args.horizon),
               pred_lstm_series,
               'b--', linewidth=2, label='LSTM', alpha=0.8)
        
        # Ligne de séparation contexte/prédiction
        ax.axvline(x=context_len, color='gray', linestyle=':',
                  linewidth=1.5, label='Forecast Start')
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title(f'Temperature Evolution at Point ({x}, {y}, {z}) - {label}',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"evolution_{label}.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Sauvegardé: evolution_{label}.png")
    
    # ============================================================
    # 4. RAPPORT FINAL
    # ============================================================
    print("\n" + "=" * 70)
    print("RAPPORT FINAL")
    print("=" * 70)
    
    print(f"\n📊 Performances des modèles:")
    print(f"\n{'Modèle':<15} {'MAE (°C)':<12} {'RMSE (°C)':<12} {'Temps (s)':<12}")
    print("-" * 55)
    
    for model_name in errors_dict:
        mae = errors_dict[model_name]['mae']
        rmse = errors_dict[model_name]['rmse']
        t = errors_dict[model_name]['time']
        print(f"{model_name:<15} {mae:<12.4f} {rmse:<12.4f} {t:<12.2f}")
    
    # Calculer le speedup par rapport à la simulation exacte
    print(f"\n⚡ Speedup par rapport à la simulation exacte ({time_simulation:.2f}s):")
    for model_name in errors_dict:
        speedup = time_simulation / errors_dict[model_name]['time']
        print(f"   {model_name}: {speedup:.2f}x plus rapide")
    
    # Trouver le meilleur modèle
    best_mae_model = min(errors_dict, key=lambda x: errors_dict[x]['mae'])
    best_speed_model = min(errors_dict, key=lambda x: errors_dict[x]['time'])
    
    print(f"\n🏆 Meilleur modèle (précision): {best_mae_model} "
          f"(MAE={errors_dict[best_mae_model]['mae']:.4f}°C)")
    print(f"⚡ Modèle le plus rapide: {best_speed_model} "
          f"({errors_dict[best_speed_model]['time']:.2f}s)")
    
    print("\n" + "=" * 70)
    print(f"✓ Tous les résultats sauvegardés dans ./{RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()