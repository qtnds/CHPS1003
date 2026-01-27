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
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

# ============================================================
# 0. CONFIGURATION
# ============================================================

RESULTS_DIR = "results_turbulence"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 1. SIMULATION RAYLEIGH-BÉNARD 3D (CORRIGÉE)
# ============================================================

class RayleighBenardSimulator:
    """
    Simulateur de convection de Rayleigh-Bénard 3D avec physique correcte
    
    Équations résolues:
    - Navier-Stokes incompressible: ∂v/∂t + (v·∇)v = -∇p + ν∇²v + βg(T-T₀)ẑ
    - Incompressibilité: ∇·v = 0
    - Transport thermique: ∂T/∂t + v·∇T = α∇²T
    
    Méthode: Projection (correction de pression)
    """
    
    def __init__(self, nx=32, ny=32, nz=32, dx=0.1, dy=0.1, dz=0.1, dt=0.0005,
                 Ra=50000, Pr=0.71, T_hot=100, T_cold=20):
        """
        Parameters:
        -----------
        nx, ny, nz : dimensions de la grille
        dx, dy, dz : pas spatiaux [m]
        dt : pas de temps [s]
        Ra : nombre de Rayleigh (>10000 pour turbulence)
        Pr : nombre de Prandtl
        T_hot : température plaque inférieure [°C]
        T_cold : température plaque supérieure [°C]
        """
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.dt = dt
        self.Ra = Ra
        self.Pr = Pr
        self.T_hot = T_hot
        self.T_cold = T_cold
        self.dT = T_hot - T_cold
        
        # Paramètres physiques adimensionnels
        self.alpha = 1.0 / (Ra * Pr)  # Diffusivité thermique
        self.nu = Pr / Ra              # Viscosité cinématique
        self.g = 1.0                   # Gravité (normalisée)
        
        # Champs sur grille décalée (staggered grid)
        self.T = np.zeros((nx, ny, nz), dtype=np.float32)
        self.vx = np.zeros((nx+1, ny, nz), dtype=np.float32)  # Décalé en x
        self.vy = np.zeros((nx, ny+1, nz), dtype=np.float32)  # Décalé en y
        self.vz = np.zeros((nx, ny, nz+1), dtype=np.float32)  # Décalé en z
        self.p = np.zeros((nx, ny, nz), dtype=np.float32)     # Pression
        
        # Initialisation température: gradient + perturbations
        for k in range(nz):
            self.T[:, :, k] = T_hot - (T_hot - T_cold) * k / (nz - 1)
        
        # Perturbations gaussiennes pour déclencher l'instabilité
        noise = np.random.randn(nx, ny, nz) * self.dT * 0.05
        self.T += gaussian_filter(noise, sigma=2.0)
        
        # Conditions aux limites température
        self.T[:, :, 0] = T_hot
        self.T[:, :, -1] = T_cold
        
        # Préparer la matrice de Poisson pour la pression
        self._prepare_poisson_solver()
        
        print(f"\n🌊 Simulateur Rayleigh-Bénard (PHYSIQUE CORRIGÉE):")
        print(f"   Grille: {nx}×{ny}×{nz}")
        print(f"   Rayleigh: Ra = {Ra:.0f} {'(TURBULENT!)' if Ra > 10000 else ''}")
        print(f"   Prandtl: Pr = {Pr:.2f}")
        print(f"   ΔT = {self.dT}°C, ν = {self.nu:.6f}, α = {self.alpha:.6f}")
        print(f"   Pas de temps: dt = {dt:.6f}s")
    
    def _prepare_poisson_solver(self):
        """Prépare la matrice pour résoudre l'équation de Poisson 3D"""
        nx, ny, nz = self.nx, self.ny, self.nz
        N = nx * ny * nz
        
        # Coefficients du Laplacien discret
        cx = 1.0 / (self.dx ** 2)
        cy = 1.0 / (self.dy ** 2)
        cz = 1.0 / (self.dz ** 2)
        
        # Diagonale principale
        main_diag = -2 * (cx + cy + cz) * np.ones(N)
        
        # Diagonales pour les dérivées
        diag_x = cx * np.ones(N)
        diag_y = cy * np.ones(N)
        diag_z = cz * np.ones(N)
        
        # Masquer les connections aux bords périodiques
        for i in range(N):
            ix = i % nx
            iy = (i // nx) % ny
            
            if ix == 0:
                diag_x[i-1] = 0
            if ix == nx-1:
                diag_x[i] = 0
            if iy == 0:
                diag_y[i-nx] = 0
            if iy == ny-1:
                diag_y[i] = 0
        
        # Construire la matrice sparse
        offsets = [0, 1, -1, nx, -nx, nx*ny, -nx*ny]
        data = [main_diag, diag_x[:-1], diag_x[1:], 
                diag_y[:-nx], diag_y[nx:],
                diag_z[:-nx*ny], diag_z[nx*ny:]]
        
        self.poisson_matrix = diags(data, offsets, shape=(N, N), format='csr')
        
        print(f"   ✓ Solveur de Poisson prêt ({N} inconnues)")
    
    def _advect_upwind(self, field, vx, vy, vz):
        """Advection avec schéma upwind (stable pour grands Ra)"""
        nx, ny, nz = field.shape
        advection = np.zeros_like(field)
        
        # Interpoler les vitesses sur la grille centrale
        vx_c = 0.5 * (vx[:-1, :, :] + vx[1:, :, :])
        vy_c = 0.5 * (vy[:, :-1, :] + vy[:, 1:, :])
        vz_c = 0.5 * (vz[:, :, :-1] + vz[:, :, 1:])
        
        # Upwind en x
        for i in range(1, nx-1):
            if vx_c[i, :, :].mean() > 0:
                advection[i, :, :] += vx_c[i, :, :] * (field[i, :, :] - field[i-1, :, :]) / self.dx
            else:
                advection[i, :, :] += vx_c[i, :, :] * (field[i+1, :, :] - field[i, :, :]) / self.dx
        
        # Upwind en y
        for j in range(1, ny-1):
            if vy_c[:, j, :].mean() > 0:
                advection[:, j, :] += vy_c[:, j, :] * (field[:, j, :] - field[:, j-1, :]) / self.dy
            else:
                advection[:, j, :] += vy_c[:, j, :] * (field[:, j+1, :] - field[:, j, :]) / self.dy
        
        # Upwind en z
        for k in range(1, nz-1):
            if vz_c[:, :, k].mean() > 0:
                advection[:, :, k] += vz_c[:, :, k] * (field[:, :, k] - field[:, :, k-1]) / self.dz
            else:
                advection[:, :, k] += vz_c[:, :, k] * (field[:, :, k+1] - field[:, :, k]) / self.dz
        
        return -advection
    
    def _laplacian(self, field):
        """Calcul du Laplacien 3D"""
        laplacian = np.zeros_like(field)
        nx, ny, nz = field.shape
        
        # Dérivées secondes avec conditions aux limites
        for i in range(1, nx-1):
            laplacian[i, :, :] += (field[i+1, :, :] - 2*field[i, :, :] + field[i-1, :, :]) / (self.dx**2)
        
        for j in range(1, ny-1):
            laplacian[:, j, :] += (field[:, j+1, :] - 2*field[:, j, :] + field[:, j-1, :]) / (self.dy**2)
        
        for k in range(1, nz-1):
            laplacian[:, :, k] += (field[:, :, k+1] - 2*field[:, :, k] + field[:, :, k-1]) / (self.dz**2)
        
        return laplacian
    
    def _project_velocity(self):
        """Correction de pression pour imposer ∇·v = 0"""
        nx, ny, nz = self.nx, self.ny, self.nz
        
        # 1. Calculer la divergence de la vitesse
        div = np.zeros((nx, ny, nz), dtype=np.float32)
        
        div += (self.vx[1:, :, :] - self.vx[:-1, :, :]) / self.dx
        div += (self.vy[:, 1:, :] - self.vy[:, :-1, :]) / self.dy
        div += (self.vz[:, :, 1:] - self.vz[:, :, :-1]) / self.dz
        
        # 2. Résoudre l'équation de Poisson: ∇²p = ρ∇·v
        rhs = div.flatten()
        p_correction = spsolve(self.poisson_matrix, rhs)
        p_correction = p_correction.reshape((nx, ny, nz))
        
        # 3. Corriger les vitesses: v_new = v_old - ∇p
        dp_dx = np.gradient(p_correction, self.dx, axis=0)
        dp_dy = np.gradient(p_correction, self.dy, axis=1)
        dp_dz = np.gradient(p_correction, self.dz, axis=2)
        
        # Interpoler sur grille décalée
        self.vx[1:-1, :, :] -= 0.5 * (dp_dx[:-1, :, :] + dp_dx[1:, :, :])
        self.vy[:, 1:-1, :] -= 0.5 * (dp_dy[:, :-1, :] + dp_dy[:, 1:, :])
        self.vz[:, :, 1:-1] -= 0.5 * (dp_dz[:, :, :-1] + dp_dz[:, :, 1:])
        
        self.p += p_correction
    
    def step(self):
        """Un pas de temps avec méthode de projection"""
        
        # Température moyenne pour la flottabilité
        T_mean = (self.T_hot + self.T_cold) / 2
        
        # ===== ÉTAPE 1: Advection + Diffusion de la vitesse =====
        
        # Interpoler les vitesses sur la grille centrale pour l'advection
        vx_c = 0.5 * (self.vx[:-1, :, :] + self.vx[1:, :, :])
        vy_c = 0.5 * (self.vy[:, :-1, :] + self.vy[:, 1:, :])
        vz_c = 0.5 * (self.vz[:, :, :-1] + self.vz[:, :, 1:])
        
        # Terme de flottabilité (force de volume)
        buoyancy = self.g * (self.T - T_mean) / self.dT
        
        # Calculer les termes advectifs (non-linéaires)
        advect_vx = self._advect_upwind(vx_c, self.vx, self.vy, self.vz)
        advect_vy = self._advect_upwind(vy_c, self.vx, self.vy, self.vz)
        advect_vz = self._advect_upwind(vz_c, self.vx, self.vy, self.vz)
        
        # Diffusion visqueuse
        diff_vx = self.nu * self._laplacian(vx_c)
        diff_vy = self.nu * self._laplacian(vy_c)
        diff_vz = self.nu * self._laplacian(vz_c)
        
        # Mise à jour prédictive (sans pression)
        vx_star = vx_c + self.dt * (advect_vx + diff_vx)
        vy_star = vy_c + self.dt * (advect_vy + diff_vy)
        vz_star = vz_c + self.dt * (advect_vz + diff_vz + buoyancy)
        
        # Réaffecter sur grille décalée
        self.vx[1:-1, :, :] = 0.5 * (vx_star[:-1, :, :] + vx_star[1:, :, :])
        self.vy[:, 1:-1, :] = 0.5 * (vy_star[:, :-1, :] + vy_star[:, 1:, :])
        self.vz[:, :, 1:-1] = 0.5 * (vz_star[:, :, :-1] + vz_star[:, :, 1:])

        
        # Conditions aux limites (pas de glissement)
        self.vx[0, :, :] = 0
        self.vx[-1, :, :] = 0
        self.vy[:, 0, :] = 0
        self.vy[:, -1, :] = 0
        self.vz[:, :, 0] = 0
        self.vz[:, :, -1] = 0
        
        # ===== ÉTAPE 2: Projection pour imposer ∇·v = 0 =====
        self._project_velocity()
        
        # ===== ÉTAPE 3: Transport de la température =====
        advect_T = self._advect_upwind(self.T, self.vx, self.vy, self.vz)
        diff_T = self.alpha * self._laplacian(self.T)
        
        self.T += self.dt * (advect_T + diff_T)
        
        # Conditions aux limites température (Dirichlet)
        self.T[:, :, 0] = self.T_hot
        self.T[:, :, -1] = self.T_cold
        
        return self.T.copy()
    
    def compute_nusselt(self):
        """Calcule le nombre de Nusselt (mesure de la convection)"""
        # Flux thermique à la paroi chaude
        dT_dz_bottom = (self.T[:, :, 1] - self.T[:, :, 0]) / self.dz
        Nu = 1 + np.abs(dT_dz_bottom.mean()) * (self.nz * self.dz) / self.dT
        return Nu
    
    def simulate(self, nt):
        """Simule nt pas de temps"""
        T_history = np.zeros((nt, self.nx, self.ny, self.nz), dtype=np.float32)
        
        print(f"\n⏳ Simulation Rayleigh-Bénard en cours ({nt} pas)...")
        
        for t in range(nt):
            T_history[t] = self.step()
            
            if (t + 1) % 50 == 0:
                Nu = self.compute_nusselt()
                vmax = np.max([np.abs(self.vx).max(), np.abs(self.vy).max(), 
                              np.abs(self.vz).max()])
                
                print(f"   t={t+1:4d}: T=[{self.T.min():.1f}, {self.T.max():.1f}]°C, "
                      f"v_max={vmax:.4f}, Nu={Nu:.2f}")
        
        print("✓ Simulation terminée!")
        return T_history


# ============================================================
# 2. MODÈLES DE PRÉDICTION (IDENTIQUE)
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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_lstm(timeseries, horizon, epochs=50, batch_size=32, device='cpu'):
    """Entraîne un LSTM"""
    N, T = timeseries.shape
    
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
            
            optimizer.zero_grad()
            pred = model(batch)
            
            target = torch.FloatTensor(timeseries[i:i+batch_size, -horizon]).unsqueeze(-1).to(device)
            
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs}, Loss: {total_loss/N:.6f}")
    
    return model


def forecast_lstm(model, timeseries, horizon, device='cpu'):
    """Prédiction LSTM"""
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
                
                context = np.append(context[0, 1:], pred_val).reshape(1, -1)
            
            predictions[i] = np.array(pred_series)
    
    return predictions


def forecast_arima(timeseries, horizon):
    """Prédiction ARIMA"""
    N, T = timeseries.shape
    predictions = np.zeros((N, horizon), dtype=np.float32)
    
    print(f"\n📊 Prédiction ARIMA en cours ({N} séries)...")
    
    for i in range(N):
        try:
            model = ARIMA(timeseries[i], order=(1, 0, 0))
            fitted = model.fit(method='yule_walker')
            forecast = fitted.forecast(steps=horizon)
            predictions[i] = forecast
        except:
            try:
                model = ARIMA(timeseries[i], order=(1, 0, 0))
                fitted = model.fit(method='yule_walker')
                forecast = fitted.forecast(steps=horizon)
                predictions[i] = forecast
            except:
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
# 4. VISUALISATION 3D (IDENTIQUE)
# ============================================================

def visualize_3d_field(volume, timestep, title="", filename=None, vmin=None, vmax=None):
    """Visualise un champ 3D"""
    fig = plt.figure(figsize=(18, 5))
    
    data = volume[timestep]
    nx, ny, nz = data.shape
    
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    
    ax1 = fig.add_subplot(131)
    mid_y = ny // 2
    im1 = ax1.imshow(data[:, mid_y, :].T, cmap='RdBu_r', origin='lower', 
                     aspect='auto', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Coupe X-Z (y={mid_y})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z (hauteur)')
    plt.colorbar(im1, ax=ax1, label='Température (°C)')
    
    ax2 = fig.add_subplot(132)
    mid_z = nz // 2
    im2 = ax2.imshow(data[:, :, mid_z].T, cmap='RdBu_r', origin='lower',
                     aspect='auto', vmin=vmin, vmax=vmax)
    ax2.set_title(f'Coupe X-Y (z={mid_z})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2, label='Température (°C)')
    
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
    
    axes[0, 0].bar(models, mae_vals, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    axes[0, 0].set_ylabel('MAE (°C)', fontsize=12)
    axes[0, 0].set_title('Mean Absolute Error', fontsize=13, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    axes[0, 1].bar(models, rmse_vals, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    axes[0, 1].set_ylabel('RMSE (°C)', fontsize=12)
    axes[0, 1].set_title('Root Mean Square Error', fontsize=13, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    axes[1, 0].bar(models, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    axes[1, 0].set_ylabel('Temps (s)', fontsize=12)
    axes[1, 0].set_title('Temps d\'exécution', fontsize=13, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
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
        description="Simulation Rayleigh-Bénard CORRIGÉE avec comparaison de modèles",
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
    parser.add_argument("--Ra", type=float, default=50000,
                       help="Nombre de Rayleigh (défaut: 50000 pour turbulence)")
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
    print("SIMULATION RAYLEIGH-BÉNARD 3D (PHYSIQUE CORRIGÉE)")
    print("Comparaison: TimesFM vs ARIMA vs LSTM vs Exact")
    print("=" * 70)
    print(f"\nParamètres:")
    print(f"  Grille: {args.nx}×{args.ny}×{args.nz}")
    print(f"  Total: {args.total} pas, Contexte: {context_len}, Horizon: {args.horizon}")
    print(f"  Ra = {args.Ra:.0f} {'← TURBULENT!' if args.Ra > 10000 else ''}")
    print(f"  Pr = {args.Pr}")
    
    # ============================================================
    # 1. SIMULATION EXACTE
    # ============================================================
    print("\n" + "=" * 70)
    print("SIMULATION PHYSIQUE (Solution exacte)")
    print("=" * 70)
    
    simulator = RayleighBenardSimulator(
        nx=args.nx, ny=args.ny, nz=args.nz,
        dx=0.1, dy=0.1, dz=0.1, dt=0.0005,
        Ra=args.Ra, Pr=args.Pr,
        T_hot=100, T_cold=20
    )
    
    start_sim = time.time()
    volume_full = simulator.simulate(args.total)
    time_simulation = time.time() - start_sim
    
    print(f"\n✓ Simulation terminée en {time_simulation:.2f}s")
    print(f"  Shape: {volume_full.shape}")
    
    print("\n📊 Génération des visualisations de la simulation...")
    for t_idx, t_name in [(0, "initial"), (context_len//2, "mid_context"), 
                           (context_len-1, "last_context")]:
        visualize_3d_field(volume_full, t_idx,
                          title=f"Rayleigh-Bénard à t={t_idx}",
                          filename=f"simulation_{t_name}")
    
    volume_context = volume_full[:context_len]
    volume_future = volume_full[context_len:context_len + args.horizon]
    
    timeseries_context = volume_to_timeseries(volume_context)
    print(f"\n📈 Séries temporelles: {timeseries_context.shape[0]} séries de longueur {timeseries_context.shape[1]}")
    
    # ============================================================
    # 2. PRÉDICTIONS
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
          f"Temps={time_lstm_total:.2f}s")
    
    # ============================================================
    # 3. VISUALISATION
    # ============================================================
    print("\n" + "=" * 70)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("=" * 70)
    
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
    
    print("\n📊 Comparaison visuelle des modèles...")
    if vol_pred_arima is not None:
        compare_predictions(volume_future, vol_pred_timesfm, 
                          vol_pred_arima, vol_pred_lstm,
                          timestep=t_vis, filename="comparison_all")
    
    print("\n📊 Graphiques de performance...")
    plot_errors_comparison(errors_dict, filename="performance_comparison")
    
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
    
    print(f"\n⚡ Speedup par rapport à la simulation exacte ({time_simulation:.2f}s):")
    for model_name in errors_dict:
        speedup = time_simulation / errors_dict[model_name]['time']
        print(f"   {model_name}: {speedup:.2f}x plus rapide")
    
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