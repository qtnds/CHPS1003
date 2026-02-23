import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, PillowWriter
import timesfm
import os
import argparse
import time
from statsmodels.tsa.arima.model import ARIMA

# ============================================================
# 0. CONFIGURATION
# ============================================================

RESULTS_DIR = "results_karman"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 1. SIMULATION ALLÉE DE KARMAN (LATTICE BOLTZMANN 2D)
# ============================================================

class KarmanVortexSimulator:
    """
    Simulateur d'allée de tourbillons de von Kármán avec Lattice Boltzmann Method (LBM)
    Modèle D2Q9 (2D, 9 vitesses)
    """
    
    def __init__(self, nx=600, ny=150, Re=150, U_inlet=0.08, obstacle_radius=15):
        """
        Parameters:
        -----------
        nx, ny : dimensions de la grille
        Re : nombre de Reynolds (contrôle la turbulence)
        U_inlet : vitesse d'entrée [m/s]
        obstacle_radius : rayon de l'obstacle cylindrique
        """
        self.nx, self.ny = nx, ny
        self.Re = Re
        self.U_inlet = U_inlet
        self.obstacle_radius = obstacle_radius
        
        # Paramètres LBM
        self.c = 1.0  # Vitesse du réseau
        self.cs = self.c / np.sqrt(3)  # Vitesse du son
        
        # Viscosité cinématique: ν = U * D / Re
        D = 2 * obstacle_radius
        self.nu = U_inlet * D / Re
        
        # Paramètre de relaxation: τ = 3ν + 0.5
        self.tau = 3 * self.nu + 0.5
        self.omega = 1.0 / self.tau  # Fréquence de collision
        
        print(f"   ω (omega) = {self.omega:.4f}")
        
        # Vecteurs de vitesse D2Q9
        self.c_vec = np.array([
            [0, 0],   # 0: repos
            [1, 0],   # 1: droite
            [0, 1],   # 2: haut
            [-1, 0],  # 3: gauche
            [0, -1],  # 4: bas
            [1, 1],   # 5: diagonale NE
            [-1, 1],  # 6: diagonale NW
            [-1, -1], # 7: diagonale SW
            [1, -1]   # 8: diagonale SE
        ])
        
        # Poids pour l'équilibre
        self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        
        # Fonctions de distribution (9 vitesses)
        self.f = np.zeros((9, nx, ny), dtype=np.float32)
        self.feq = np.zeros((9, nx, ny), dtype=np.float32)
        
        # Champs macroscopiques
        self.rho = np.ones((nx, ny), dtype=np.float32)
        self.ux = np.zeros((nx, ny), dtype=np.float32)
        self.uy = np.zeros((nx, ny), dtype=np.float32)
        
        # Créer l'obstacle (cylindre)
        self.obstacle = self._create_cylinder_obstacle()
        
        # Initialiser l'écoulement
        self._initialize_flow()
        
        print(f"\n🌊 Simulateur Allée de Karman (Lattice Boltzmann):")
        print(f"   Grille: {nx}×{ny}")
        print(f"   Reynolds: Re = {Re:.1f}")
        print(f"   Vitesse entrée: U = {U_inlet:.3f} m/s")
        print(f"   Viscosité: ν = {self.nu:.6f} m²/s")
        print(f"   Paramètre relaxation: τ = {self.tau:.3f}")
        print(f"   Obstacle: cylindre rayon {obstacle_radius} px")
    
    def _create_cylinder_obstacle(self):
        """Crée un masque booléen pour l'obstacle cylindrique"""
        cx, cy = self.nx // 5, self.ny // 2  # Plus en amont pour voir les tourbillons
        
        X, Y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')
        obstacle = (X - cx)**2 + (Y - cy)**2 <= self.obstacle_radius**2
        
        print(f"   Position obstacle: ({cx}, {cy})")
        return obstacle
    
    def _initialize_flow(self):
        """Initialise l'écoulement avec vitesse uniforme + perturbation"""
        # Vitesse uniforme horizontale
        self.ux[:] = self.U_inlet
        self.uy[:] = 0.0
        self.rho[:] = 1.0
        
        # IMPORTANT : Ajouter une petite perturbation pour déclencher l'instabilité de Karman
        # Perturbation sinusoïdale en aval de l'obstacle
        cx = self.nx // 5
        for i in range(cx + self.obstacle_radius + 5, self.nx):
            for j in range(self.ny):
                # Perturbation en vitesse verticale
                self.uy[i, j] += 0.01 * self.U_inlet * np.sin(2 * np.pi * j / self.ny * 4)
        
        # Initialiser les fonctions de distribution à l'équilibre
        for i in range(9):
            self.f[i] = self._equilibrium(i, self.rho, self.ux, self.uy)
    
    def _equilibrium(self, i, rho, ux, uy):
        """Calcule la fonction d'équilibre pour la vitesse i"""
        cu = self.c_vec[i, 0] * ux + self.c_vec[i, 1] * uy
        u2 = ux**2 + uy**2
        
        feq = self.w[i] * rho * (
            1 + 3 * cu / self.c**2 +
            4.5 * cu**2 / self.c**4 -
            1.5 * u2 / self.c**2
        )
        return feq
    
    def _streaming(self):
        """Étape de streaming (advection)"""
        for i in range(1, 9):  # Ignorer la vitesse de repos
            self.f[i] = np.roll(self.f[i], self.c_vec[i, 0], axis=0)
            self.f[i] = np.roll(self.f[i], self.c_vec[i, 1], axis=1)
    
    def _macroscopic(self):
        """Calcule les variables macroscopiques à partir de f"""
        self.rho = np.sum(self.f, axis=0)
        self.ux = np.sum(self.f * self.c_vec[:, 0, None, None], axis=0) / self.rho
        self.uy = np.sum(self.f * self.c_vec[:, 1, None, None], axis=0) / self.rho
    
    def _collision(self):
        """Étape de collision (BGK)"""
        for i in range(9):
            self.feq[i] = self._equilibrium(i, self.rho, self.ux, self.uy)
            self.f[i] += self.omega * (self.feq[i] - self.f[i])
    
    def _boundary_conditions(self):
        """Applique les conditions aux limites"""
        # Entrée (Zou-He): vitesse imposée
        self.ux[0, :] = self.U_inlet
        self.uy[0, :] = 0.0
        self.rho[0, :] = (
            self.f[0, 0, :] + self.f[2, 0, :] + self.f[4, 0, :] +
            2 * (self.f[3, 0, :] + self.f[6, 0, :] + self.f[7, 0, :])
        ) / (1 - self.ux[0, :])
        
        # Réinitialiser les distributions à l'entrée
        for i in range(9):
            self.f[i, 0, :] = self._equilibrium(i, self.rho[0, :], 
                                               self.ux[0, :], self.uy[0, :])
        
        # Sortie: gradient nul (extrapolation)
        self.f[:, -1, :] = self.f[:, -2, :]
        
        # Parois haut/bas: bounce-back
        self.f[2, :, -1] = self.f[4, :, -1]  # Haut
        self.f[5, :, -1] = self.f[7, :, -1]
        self.f[6, :, -1] = self.f[8, :, -1]
        
        self.f[4, :, 0] = self.f[2, :, 0]    # Bas
        self.f[7, :, 0] = self.f[5, :, 0]
        self.f[8, :, 0] = self.f[6, :, 0]
        
        # Obstacle: bounce-back total
        for i in range(9):
            # Vitesse opposée
            i_opp = [0, 3, 4, 1, 2, 7, 8, 5, 6][i]
            self.f[i_opp, self.obstacle] = self.f[i, self.obstacle]
    
    def step(self):
        """Un pas de temps LBM"""
        self._streaming()
        self._macroscopic()
        self._collision()
        self._boundary_conditions()
        
        # Retourner la vorticité (pour visualisation)
        return self.compute_vorticity()
    
    def compute_vorticity(self):
        """Calcule la vorticité ω = ∂uy/∂x - ∂ux/∂y"""
        duy_dx = np.gradient(self.uy, axis=0)
        dux_dy = np.gradient(self.ux, axis=1)
        vorticity = duy_dx - dux_dy
        return vorticity
    
    def simulate(self, nt):
        """Simule nt pas de temps"""
        vorticity_history = np.zeros((nt, self.nx, self.ny), dtype=np.float32)
        
        print(f"\n⏳ Simulation de l'allée de Karman ({nt} pas)...")
        
        for t in range(nt):
            vorticity_history[t] = self.step()
            
            if (t + 1) % 50 == 0:
                u_mag = np.sqrt(self.ux**2 + self.uy**2)
                print(f"   t={t+1:4d}: U_max={u_mag.max():.4f}, "
                      f"ω_max={np.abs(vorticity_history[t]).max():.4f}")
        
        print("✓ Simulation terminée!")
        return vorticity_history


# ============================================================
# 2. MODÈLES DE PRÉDICTION
# ============================================================

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def train_lstm(timeseries, horizon, epochs=30, batch_size=64, device='cpu'):
    N, T = timeseries.shape
    X_train = torch.FloatTensor(timeseries[:, :-horizon]).unsqueeze(-1).to(device)
    
    model = LSTMForecaster().to(device)
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
    model.eval()
    N, T = timeseries.shape
    predictions = np.zeros((N, horizon), dtype=np.float32)
    
    with torch.no_grad():
        for i in range(N):
            context = timeseries[i:i+1, :]
            pred_series = []
            
            for h in range(horizon):
                x = torch.FloatTensor(context).unsqueeze(-1).to(device)
                pred = model(x).cpu().numpy()[0, 0]
                pred_series.append(pred)
                context = np.append(context[0, 1:], pred).reshape(1, -1)
            
            predictions[i] = pred_series
    
    return predictions


def forecast_arima(timeseries, horizon):
    N, T = timeseries.shape
    predictions = np.zeros((N, horizon), dtype=np.float32)
    
    print(f"\n📊 Prédiction ARIMA ({N} séries)...")
    
    for i in range(N):
        try:
            model = ARIMA(timeseries[i], order=(1, 0, 0))
            fitted = model.fit(method='yule_walker')
            predictions[i] = fitted.forecast(steps=horizon)
        except:
            predictions[i] = timeseries[i, -1]
        
        if (i + 1) % 2000 == 0:
            print(f"   [{i+1}/{N}] séries traitées")
    
    print("✓ ARIMA terminé!")
    return predictions


def volume_to_timeseries(volume):
    T, X, Y = volume.shape
    return volume.reshape(T, X * Y).T


def timeseries_to_volume(timeseries, spatial_shape):
    N, T = timeseries.shape
    X, Y = spatial_shape
    return timeseries.T.reshape(T, X, Y)


class TimesFMForecaster:
    def __init__(self, max_context=512, max_horizon=128):
        torch.set_float32_matmul_precision("high")
        
        print("\n🤖 Chargement TimesFM...")
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
                infer_is_positive=False,  # Vorticité peut être négative
                fix_quantile_crossing=True,
            )
        )
        print("✓ TimesFM prêt")
    
    def forecast(self, timeseries, horizon, batch_size=64):
        N = len(timeseries)
        point_all = []
        
        print(f"\n🔮 Prédiction TimesFM ({N} séries)...")
        for i in range(0, N, batch_size):
            batch = timeseries[i:i + batch_size]
            inputs = [series.astype(np.float32) for series in batch]
            points, _ = self.model.forecast(horizon=horizon, inputs=inputs)
            point_all.append(points)
            
            if (i + batch_size) % 1000 == 0:
                print(f"   [{min(i + batch_size, N)}/{N}] séries")
        
        print("✓ TimesFM terminé!")
        return np.vstack(point_all)


# ============================================================
# 3. CRÉATION DE GIFS
# ============================================================

def create_animation_gif(volume, title, filename, fps=20, vmin=None, vmax=None, obstacle=None):
    """Crée un GIF animé de l'évolution temporelle avec fond noir et tourbillons en blanc"""
    print(f"\n🎬 Création du GIF: {filename}...")
    
    nt = volume.shape[0]
    
    # Calculer la valeur absolue de la vorticité
    vort_abs = np.abs(volume)
    
    # Seuil beaucoup plus agressif pour ne montrer que les forts tourbillons
    threshold = np.percentile(vort_abs, 85)  # 85e percentile au lieu de 90
    
    # Normalisation avec compression pour intensifier les blancs
    vmax_sym = np.percentile(vort_abs, 98)  # Saturer plus tôt = plus de blanc
    
    # Colormap personnalisée: NOIR -> BLANC avec transition rapide
    from matplotlib.colors import LinearSegmentedColormap
    # Transition rapide vers le blanc pour intensifier les tourbillons
    colors = ['#000000', '#0a0a0a', '#1a1a1a', '#404040', '#808080', '#d0d0d0', '#ffffff', '#ffffff']
    cmap = LinearSegmentedColormap.from_list('karman_bw', colors, N=256)
    
    # Tout en dessous du seuil = noir complet
    cmap.set_under('black')
    
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='black')
    ax.set_facecolor('black')
    
    def update(frame):
        ax.clear()
        ax.set_facecolor('black')
        
        # Valeur absolue de la vorticité
        vort_frame = np.abs(volume[frame])
        
        # Appliquer un gamma < 1 pour intensifier les hautes valeurs (plus de blanc)
        vort_enhanced = np.power(vort_frame / vmax_sym, 0.6) * vmax_sym
        vort_enhanced = np.clip(vort_enhanced, 0, vmax_sym)
        
        # Afficher avec seuillage agressif
        im = ax.imshow(vort_enhanced.T, cmap=cmap, origin='lower',
                      aspect='auto', vmin=threshold*0.2, vmax=vmax_sym,
                      interpolation='bilinear')
        
        # Superposer l'obstacle en blanc opaque
        if obstacle is not None:
            obstacle_mask = np.ma.masked_where(~obstacle, obstacle)
            ax.imshow(obstacle_mask.T, cmap='gray', origin='lower',
                     aspect='auto', alpha=1.0, vmin=0, vmax=1)
        
        ax.set_title(f'{title} - Frame {frame}/{nt-1}', 
                    fontsize=14, fontweight='bold', color='white')
        ax.set_xlabel('X (direction écoulement)', color='white')
        ax.set_ylabel('Y (largeur)', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        return [im]
    
    anim = FuncAnimation(fig, update, frames=nt, interval=1000//fps, blit=True)
    
    path = os.path.join(RESULTS_DIR, f"{filename}.gif")
    writer = PillowWriter(fps=fps)
    anim.save(path, writer=writer, savefig_kwargs={'facecolor': 'black'})
    plt.close()
    
    print(f"   ✓ Sauvegardé: {filename}.gif ({nt} frames)")


def create_comparison_gif(true_vol, pred_timesfm, pred_arima, pred_lstm, 
                         filename="comparison", fps=15, obstacle=None):
    """Crée un GIF comparant les 4 méthodes côte à côte avec fond noir et tourbillons blancs intensifiés"""
    print(f"\n🎬 Création du GIF comparatif...")
    
    nt = true_vol.shape[0]
    
    # Utiliser la valeur absolue et calculer le seuil
    vort_all = np.concatenate([np.abs(true_vol), np.abs(pred_timesfm), 
                               np.abs(pred_arima), np.abs(pred_lstm)])
    threshold = np.percentile(vort_all, 85)
    vmax_sym = np.percentile(vort_all, 98)
    
    # Colormap noir -> blanc avec transition rapide
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#000000', '#0a0a0a', '#1a1a1a', '#404040', '#808080', '#d0d0d0', '#ffffff', '#ffffff']
    cmap = LinearSegmentedColormap.from_list('karman_bw', colors, N=256)
    cmap.set_under('black')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), facecolor='black')
    fig.suptitle('Comparaison des prédictions - Allée de Karman', 
                fontsize=16, fontweight='bold', color='white')
    
    titles = ['Ground Truth', 'TimesFM', 'ARIMA', 'LSTM']
    
    def update(frame):
        for idx, (ax, vol, title) in enumerate(zip(axes.flat, 
                                                    [true_vol, pred_timesfm, pred_arima, pred_lstm],
                                                    titles)):
            ax.clear()
            ax.set_facecolor('black')
            
            # Valeur absolue avec intensification gamma
            vort_frame = np.abs(vol[frame])
            vort_enhanced = np.power(vort_frame / vmax_sym, 0.6) * vmax_sym
            vort_enhanced = np.clip(vort_enhanced, 0, vmax_sym)
            
            im = ax.imshow(vort_enhanced.T, cmap=cmap, origin='lower',
                          aspect='auto', vmin=threshold*0.2, vmax=vmax_sym,
                          interpolation='bilinear')
            
            # Superposer l'obstacle
            if obstacle is not None:
                obstacle_mask = np.ma.masked_where(~obstacle, obstacle)
                ax.imshow(obstacle_mask.T, cmap='gray', origin='lower',
                         aspect='auto', alpha=1.0, vmin=0, vmax=1)
            
            ax.set_title(f'{title} (t={frame})', fontsize=12, color='white')
            ax.set_xlabel('X', color='white')
            ax.set_ylabel('Y', color='white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
        
        return axes.flat
    
    anim = FuncAnimation(fig, update, frames=nt, interval=1000//fps, blit=False)
    
    path = os.path.join(RESULTS_DIR, f"{filename}.gif")
    writer = PillowWriter(fps=fps)
    anim.save(path, writer=writer, savefig_kwargs={'facecolor': 'black'})
    plt.close()
    
    print(f"   ✓ Sauvegardé: {filename}.gif")


def plot_errors_comparison(errors_dict, filename="errors"):
    """Graphique de comparaison des performances"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    models = list(errors_dict.keys())
    mae_vals = [errors_dict[m]['mae'] for m in models]
    rmse_vals = [errors_dict[m]['rmse'] for m in models]
    times = [errors_dict[m]['time'] for m in models]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    axes[0].bar(models, mae_vals, color=colors[:len(models)])
    axes[0].set_ylabel('MAE', fontsize=12)
    axes[0].set_title('Mean Absolute Error', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].bar(models, rmse_vals, color=colors[:len(models)])
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title('Root Mean Square Error', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    axes[2].bar(models, times, color=colors[:len(models)])
    axes[2].set_ylabel('Temps (s)', fontsize=12)
    axes[2].set_title('Temps d\'exécution', fontsize=13, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{filename}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Sauvegardé: {filename}.png")


# ============================================================
# 4. SCRIPT PRINCIPAL
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Simulation Allée de Karman avec comparaison TimesFM/ARIMA/LSTM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--nx", type=int, default=600,
                       help="Longueur de la grille (recommandé: 600)")
    parser.add_argument("--ny", type=int, default=150,
                       help="Largeur de la grille (recommandé: 150)")
    parser.add_argument("--total", type=int, default=1000,
                       help="Nombre total de pas de temps (min 800 pour Karman)")
    parser.add_argument("--horizon", type=int, default=200,
                       help="Horizon de prédiction")
    parser.add_argument("--Re", type=float, default=150,
                       help="Nombre de Reynolds (100-300 pour Karman, 150 optimal)")
    parser.add_argument("--obstacle-radius", type=int, default=15,
                       help="Rayon de l'obstacle")
    parser.add_argument("--skip-arima", action="store_true",
                       help="Sauter ARIMA")
    parser.add_argument("--lstm-epochs", type=int, default=20,
                       help="Epochs LSTM")
    parser.add_argument("--gif-fps", type=int, default=20,
                       help="FPS des GIFs")
    
    args = parser.parse_args()
    
    if args.horizon >= args.total:
        raise ValueError(f"horizon < total requis")
    
    context_len = args.total - args.horizon
    
    print("=" * 70)
    print("ALLÉE DE TOURBILLONS DE VON KÁRMÁN")
    print("Comparaison: TimesFM vs ARIMA vs LSTM")
    print("=" * 70)
    print(f"\nParamètres:")
    print(f"  Grille: {args.nx}×{args.ny}")
    print(f"  Total: {args.total}, Contexte: {context_len}, Horizon: {args.horizon}")
    print(f"  Reynolds: Re = {args.Re:.1f}")
    print(f"\n⚠️  NOTE: Les tourbillons de Karman se développent progressivement.")
    print(f"  Attendez ~300-500 pas de temps pour voir les structures tourbillonnaires!")
    print(f"  Re optimal: 100-300 (150 recommandé)")
    print(f"  Position obstacle: nx/5 pour laisser place aux tourbillons en aval")
    
    # ============================================================
    # 1. SIMULATION
    # ============================================================
    print("\n" + "=" * 70)
    print("SIMULATION LATTICE BOLTZMANN")
    print("=" * 70)
    
    simulator = KarmanVortexSimulator(
        nx=args.nx, ny=args.ny,
        Re=args.Re,
        U_inlet=0.1,
        obstacle_radius=args.obstacle_radius
    )
    
    start_sim = time.time()
    vorticity_full = simulator.simulate(args.total)
    time_sim = time.time() - start_sim
    
    print(f"\n✓ Simulation: {time_sim:.2f}s, shape={vorticity_full.shape}")
    
    # GIF simulation exacte
    create_animation_gif(vorticity_full, "Simulation Exacte - Allée de Karman", 
                        "simulation_exact", fps=args.gif_fps, obstacle=simulator.obstacle)
    
    # Séparer contexte/futur
    vort_context = vorticity_full[:context_len]
    vort_future = vorticity_full[context_len:context_len + args.horizon]
    
    timeseries_context = volume_to_timeseries(vort_context)
    print(f"\n📈 Séries: {timeseries_context.shape[0]} × {timeseries_context.shape[1]}")
    
    # ============================================================
    # 2. PRÉDICTIONS
    # ============================================================
    errors_dict = {}
    spatial_shape = (args.nx, args.ny)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    
    # --- TimesFM ---
    print("\n" + "=" * 70)
    print("TIMESFM")
    print("=" * 70)
    
    forecaster_timesfm = TimesFMForecaster(max_context=context_len, max_horizon=args.horizon)
    start = time.time()
    pred_timesfm = forecaster_timesfm.forecast(timeseries_context, args.horizon, batch_size=64)
    time_timesfm = time.time() - start
    
    vol_pred_timesfm = timeseries_to_volume(pred_timesfm, spatial_shape)
    
    mae_tf = np.mean(np.abs(vort_future - vol_pred_timesfm))
    rmse_tf = np.sqrt(np.mean((vort_future - vol_pred_timesfm)**2))
    errors_dict['TimesFM'] = {'mae': mae_tf, 'rmse': rmse_tf, 'time': time_timesfm}
    
    print(f"\n✓ TimesFM: MAE={mae_tf:.4f}, RMSE={rmse_tf:.4f}, Temps={time_timesfm:.2f}s")
    
    # GIF TimesFM
    create_animation_gif(vol_pred_timesfm, "Prédiction TimesFM", 
                        "prediction_timesfm", fps=args.gif_fps,
                        vmin=vorticity_full.min(), vmax=vorticity_full.max(),
                        obstacle=simulator.obstacle)
    
    # --- ARIMA ---
    if not args.skip_arima:
        print("\n" + "=" * 70)
        print("ARIMA")
        print("=" * 70)
        
        start = time.time()
        pred_arima = forecast_arima(timeseries_context, args.horizon)
        time_arima = time.time() - start
        
        vol_pred_arima = timeseries_to_volume(pred_arima, spatial_shape)
        
        mae_ar = np.mean(np.abs(vort_future - vol_pred_arima))
        rmse_ar = np.sqrt(np.mean((vort_future - vol_pred_arima)**2))
        errors_dict['ARIMA'] = {'mae': mae_ar, 'rmse': rmse_ar, 'time': time_arima}
        
        print(f"\n✓ ARIMA: MAE={mae_ar:.4f}, RMSE={rmse_ar:.4f}, Temps={time_arima:.2f}s")
        
        create_animation_gif(vol_pred_arima, "Prédiction ARIMA",
                            "prediction_arima", fps=args.gif_fps,
                            vmin=vorticity_full.min(), vmax=vorticity_full.max(),
                            obstacle=simulator.obstacle)
    else:
        print("\n⚠️  ARIMA ignoré")
        vol_pred_arima = np.zeros_like(vort_future)
    
    # --- LSTM ---
    print("\n" + "=" * 70)
    print("LSTM")
    print("=" * 70)
    
    start_train = time.time()
    lstm_model = train_lstm(timeseries_context, args.horizon, 
                           epochs=args.lstm_epochs, batch_size=64, device=device)
    time_train = time.time() - start_train
    
    start_pred = time.time()
    pred_lstm = forecast_lstm(lstm_model, timeseries_context, args.horizon, device=device)
    time_pred = time.time() - start_pred
    time_lstm = time_train + time_pred
    
    vol_pred_lstm = timeseries_to_volume(pred_lstm, spatial_shape)
    
    mae_lstm = np.mean(np.abs(vort_future - vol_pred_lstm))
    rmse_lstm = np.sqrt(np.mean((vort_future - vol_pred_lstm)**2))
    errors_dict['LSTM'] = {'mae': mae_lstm, 'rmse': rmse_lstm, 'time': time_lstm}
    
    print(f"\n✓ LSTM: MAE={mae_lstm:.4f}, RMSE={rmse_lstm:.4f}, Temps={time_lstm:.2f}s")
    
    create_animation_gif(vol_pred_lstm, "Prédiction LSTM",
                        "prediction_lstm", fps=args.gif_fps,
                        vmin=vorticity_full.min(), vmax=vorticity_full.max())

    # ============================================================
    # 3. GIF COMPARATIF
    # ============================================================
    print("\n" + "=" * 70)
    print("CRÉATION GIF COMPARATIF")
    print("=" * 70)

    if not args.skip_arima:
        create_comparison_gif(vort_future, vol_pred_timesfm, vol_pred_arima, vol_pred_lstm,
                            filename="comparison_all", fps=args.gif_fps)
    else:
        # Version sans ARIMA (3 modèles)
        print(f"\n🎬 Création du GIF comparatif (sans ARIMA)...")
        
        nt = vort_future.shape[0]
        vmin = min(vort_future.min(), vol_pred_timesfm.min(), vol_pred_lstm.min())
        vmax = max(vort_future.max(), vol_pred_timesfm.max(), vol_pred_lstm.max())
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Comparaison - Allée de Karman', fontsize=16, fontweight='bold')
        
        titles = ['Ground Truth', 'TimesFM', 'LSTM']
        volumes = [vort_future, vol_pred_timesfm, vol_pred_lstm]
        
        def update(frame):
            for ax, vol, title in zip(axes, volumes, titles):
                ax.clear()
                im = ax.imshow(vol[frame].T, cmap='RdBu_r', origin='lower',
                            aspect='auto', vmin=vmin, vmax=vmax)
                ax.set_title(f'{title} (t={frame})', fontsize=12)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
            return axes
        
        anim = FuncAnimation(fig, update, frames=nt, interval=1000//args.gif_fps, blit=False)
        
        path = os.path.join(RESULTS_DIR, "comparison_all.gif")
        writer = PillowWriter(fps=args.gif_fps)
        anim.save(path, writer=writer)
        plt.close()
        
        print(f"   ✓ Sauvegardé: comparison_all.gif")

    # ============================================================
    # 4. GRAPHIQUES DE PERFORMANCE
    # ============================================================
    print("\n" + "=" * 70)
    print("GRAPHIQUES DE PERFORMANCE")
    print("=" * 70)

    plot_errors_comparison(errors_dict, filename="performance_comparison")

    # Évolution temporelle en quelques points
    print("\n📊 Évolution temporelle en points clés...")

    points = [
        (args.nx // 2, args.ny // 2, "center"),
        (3 * args.nx // 4, args.ny // 2, "downstream"),
        (args.nx // 4 + 20, args.ny // 2 + 5, "near_obstacle")
    ]

    for x, y, label in points:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Série vraie
        true_series = vorticity_full[:, x, y]
        ax.plot(range(len(true_series)), true_series, 
            'k-', linewidth=2, label='Ground Truth', alpha=0.8)
        
        # Prédictions
        pred_timesfm_series = vol_pred_timesfm[:, x, y]
        ax.plot(range(context_len, context_len + args.horizon), 
            pred_timesfm_series, 
            'r--', linewidth=2, label='TimesFM', alpha=0.8)
        
        if not args.skip_arima:
            pred_arima_series = vol_pred_arima[:, x, y]
            ax.plot(range(context_len, context_len + args.horizon),
                pred_arima_series,
                'g--', linewidth=2, label='ARIMA', alpha=0.8)
        
        pred_lstm_series = vol_pred_lstm[:, x, y]
        ax.plot(range(context_len, context_len + args.horizon),
            pred_lstm_series,
            'b--', linewidth=2, label='LSTM', alpha=0.8)
        
        ax.axvline(x=context_len, color='gray', linestyle=':',
                linewidth=1.5, label='Forecast Start')
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Vorticity', fontsize=12)
        ax.set_title(f'Vorticity Evolution at ({x}, {y}) - {label}',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"evolution_{label}.png"),
                dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Sauvegardé: evolution_{label}.png")

    # ============================================================
    # 5. RAPPORT FINAL
    # ============================================================
    print("\n" + "=" * 70)
    print("RAPPORT FINAL")
    print("=" * 70)

    print(f"\n📊 Performances des modèles:")
    print(f"\n{'Modèle':<15} {'MAE':<12} {'RMSE':<12} {'Temps (s)':<12}")
    print("-" * 55)

    for model_name in errors_dict:
        mae = errors_dict[model_name]['mae']
        rmse = errors_dict[model_name]['rmse']
        t = errors_dict[model_name]['time']
        print(f"{model_name:<15} {mae:<12.4f} {rmse:<12.4f} {t:<12.2f}")

    print(f"\n⚡ Speedup par rapport à la simulation exacte ({time_sim:.2f}s):")
    for model_name in errors_dict:
        speedup = time_sim / errors_dict[model_name]['time']
        print(f"   {model_name}: {speedup:.2f}x plus rapide")

    best_mae_model = min(errors_dict, key=lambda x: errors_dict[x]['mae'])
    best_speed_model = min(errors_dict, key=lambda x: errors_dict[x]['time'])

    print(f"\n🏆 Meilleur modèle (précision): {best_mae_model} "
        f"(MAE={errors_dict[best_mae_model]['mae']:.4f})")
    print(f"⚡ Modèle le plus rapide: {best_speed_model} "
        f"({errors_dict[best_speed_model]['time']:.2f}s)")

    print("\n📁 Fichiers générés:")
    print(f"   • simulation_exact.gif")
    print(f"   • prediction_timesfm.gif")
    if not args.skip_arima:
        print(f"   • prediction_arima.gif")
    print(f"   • prediction_lstm.gif")
    print(f"   • comparison_all.gif")
    print(f"   • performance_comparison.png")
    print(f"   • evolution_*.png (3 points)")

    print("\n" + "=" * 70)
    print(f"✓ Tous les résultats sauvegardés dans ./{RESULTS_DIR}/")
    print("=" * 70)

if __name__ == "__main__":
    main()