import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, PillowWriter
import timesfm
from chronos import ChronosPipeline
import os
import argparse
import time
from statsmodels.tsa.arima.model import ARIMA

# ============================================================
# 0. CONFIGURATION
# ============================================================

RESULTS_DIR = "results_karman_comparaison"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 1. SIMULATION ALLÉE DE KARMAN (LATTICE BOLTZMANN 2D)
# ============================================================

class KarmanVortexSimulator:
    """
    Simulateur d'allée de tourbillons de von Kármán avec Lattice Boltzmann Method (LBM)
    Modèle D2Q9 (2D, 9 vitesses)

    Stratégie "boîte large + fenêtre" :
    - La simulation tourne sur une grille PHYSIQUE de taille (nx_phys × ny_phys).
    - nx_phys = nx_win  + 2*pad_x  (marges gauche/droite)
    - ny_phys = ny_win  + 2*pad_y  (marges haut/bas)
    - Seule la fenêtre centrale (nx_win × ny_win) est extraite pour les GIFs et les
      prédictions → toute perturbation de bord reste hors fenêtre.
    """

    def __init__(self,
                 nx_win=400, ny_win=100,   # taille de la fenêtre d'analyse
                 pad_x=100,  pad_y=50,     # marges supplémentaires de chaque côté
                 Re=150, U_inlet=0.08,
                 obstacle_radius=15):
        """
        Parameters
        ----------
        nx_win, ny_win : dimensions de la fenêtre d'analyse (GIFs / forecast)
        pad_x, pad_y   : marges ajoutées de chaque côté (absorbent les effets de bord)
        Re             : nombre de Reynolds
        U_inlet        : vitesse d'entrée [lu/ts]
        obstacle_radius: rayon du cylindre [lu]
        """
        self.nx_win  = nx_win
        self.ny_win  = ny_win
        self.pad_x   = pad_x
        self.pad_y   = pad_y

        # Grille physique totale
        self.nx = nx_win + 2 * pad_x
        self.ny = ny_win + 2 * pad_y

        self.Re = Re
        self.U_inlet = U_inlet
        self.obstacle_radius = obstacle_radius

        # Paramètres LBM
        self.c  = 1.0
        self.cs = self.c / np.sqrt(3)

        D       = 2 * obstacle_radius
        self.nu = U_inlet * D / Re
        self.tau   = 3 * self.nu + 0.5
        self.omega = 1.0 / self.tau

        print(f"   τ = {self.tau:.4f},  ω = {self.omega:.4f}")

        # Vecteurs de vitesse D2Q9
        self.c_vec = np.array([
            [ 0,  0],  # 0
            [ 1,  0],  # 1
            [ 0,  1],  # 2
            [-1,  0],  # 3
            [ 0, -1],  # 4
            [ 1,  1],  # 5
            [-1,  1],  # 6
            [-1, -1],  # 7
            [ 1, -1],  # 8
        ])
        self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9,
                           1/36, 1/36, 1/36, 1/36])

        # Fonctions de distribution
        self.f    = np.zeros((9, self.nx, self.ny), dtype=np.float32)
        self.feq  = np.zeros((9, self.nx, self.ny), dtype=np.float32)

        # Champs macroscopiques
        self.rho = np.ones( (self.nx, self.ny), dtype=np.float32)
        self.ux  = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.uy  = np.zeros((self.nx, self.ny), dtype=np.float32)

        # Obstacle (dans la grille physique)
        self.obstacle = self._create_cylinder_obstacle()

        self._initialize_flow()

        # Fenêtre d'extraction (indices dans la grille physique)
        self.x0 = pad_x
        self.x1 = pad_x + nx_win
        self.y0 = pad_y
        self.y1 = pad_y + ny_win

        # Masque obstacle restreint à la fenêtre
        self.obstacle_win = self.obstacle[self.x0:self.x1, self.y0:self.y1]

        print(f"\n🌊 Simulateur Allée de Karman (Lattice Boltzmann):")
        print(f"   Grille physique : {self.nx}×{self.ny}")
        print(f"   Fenêtre analyse : {nx_win}×{ny_win}  "
              f"(marges x={pad_x}, y={pad_y})")
        print(f"   Reynolds : Re = {Re:.1f}")
        print(f"   U_inlet  = {U_inlet:.3f}  ν = {self.nu:.6f}")
        print(f"   τ = {self.tau:.3f}  ω = {self.omega:.4f}")
        print(f"   Obstacle : cylindre r={obstacle_radius} px")

    # ----------------------------------------------------------
    def _create_cylinder_obstacle(self):
        """Cylindre centré dans la grille physique, un peu en amont."""
        # On le place dans le premier tiers de la fenêtre (pas de la grille totale)
        cx = self.pad_x + self.nx_win // 4
        cy = self.ny // 2
        X, Y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')
        obstacle = (X - cx)**2 + (Y - cy)**2 <= self.obstacle_radius**2
        print(f"   Position obstacle (grille physique): ({cx}, {cy})")
        return obstacle

    def _initialize_flow(self):
        """Vitesse uniforme + petite perturbation sinusoïdale."""
        self.ux[:] = self.U_inlet
        self.uy[:] = 0.0
        self.rho[:] = 1.0

        cx = self.pad_x + self.nx_win // 4
        for i in range(cx + self.obstacle_radius + 5, self.nx):
            for j in range(self.ny):
                self.uy[i, j] += 0.01 * self.U_inlet * np.sin(
                    2 * np.pi * j / self.ny * 4)

        for i in range(9):
            self.f[i] = self._equilibrium(i, self.rho, self.ux, self.uy)

    # ----------------------------------------------------------
    def _equilibrium(self, i, rho, ux, uy):
        cu = self.c_vec[i, 0] * ux + self.c_vec[i, 1] * uy
        u2 = ux**2 + uy**2
        return self.w[i] * rho * (
            1 + 3*cu/self.c**2 + 4.5*cu**2/self.c**4 - 1.5*u2/self.c**2)

    def _streaming(self):
        for i in range(1, 9):
            self.f[i] = np.roll(self.f[i], self.c_vec[i, 0], axis=0)
            self.f[i] = np.roll(self.f[i], self.c_vec[i, 1], axis=1)

    def _macroscopic(self):
        self.rho = np.sum(self.f, axis=0)
        self.ux  = np.sum(self.f * self.c_vec[:, 0, None, None], axis=0) / self.rho
        self.uy  = np.sum(self.f * self.c_vec[:, 1, None, None], axis=0) / self.rho

    def _collision(self):
        for i in range(9):
            self.feq[i] = self._equilibrium(i, self.rho, self.ux, self.uy)
            self.f[i]  += self.omega * (self.feq[i] - self.f[i])

    def _boundary_conditions(self):
        """Conditions aux limites sur la GRILLE PHYSIQUE.

        Entrée (x=0) : vitesse imposée Zou-He.
        Sortie (x=-1): gradient nul (outflow).
        Parois (y=0, y=-1) : bounce-back — ces parois sont dans les marges,
            donc hors de la fenêtre d'analyse → leur artefact n'est pas vu.
        Obstacle : bounce-back total.
        """
        # --- Entrée (hors fenêtre grâce au pad_x) ---
        self.ux[0, :] = self.U_inlet
        self.uy[0, :] = 0.0
        self.rho[0, :] = (
            self.f[0, 0, :] + self.f[2, 0, :] + self.f[4, 0, :] +
            2*(self.f[3, 0, :] + self.f[6, 0, :] + self.f[7, 0, :])
        ) / (1 - self.ux[0, :])
        for i in range(9):
            self.f[i, 0, :] = self._equilibrium(
                i, self.rho[0, :], self.ux[0, :], self.uy[0, :])

        # --- Sortie (hors fenêtre) ---
        self.f[:, -1, :] = self.f[:, -2, :]

        # --- Parois haut/bas (hors fenêtre grâce au pad_y) ---
        # Paroi y_max
        self.f[2, :, -1] = self.f[4, :, -1]
        self.f[5, :, -1] = self.f[7, :, -1]
        self.f[6, :, -1] = self.f[8, :, -1]
        # Paroi y_min
        self.f[4, :,  0] = self.f[2, :,  0]
        self.f[7, :,  0] = self.f[5, :,  0]
        self.f[8, :,  0] = self.f[6, :,  0]

        # --- Obstacle ---
        opp = [0, 3, 4, 1, 2, 7, 8, 5, 6]
        for i in range(9):
            self.f[opp[i], self.obstacle] = self.f[i, self.obstacle]

    # ----------------------------------------------------------
    def step(self):
        """Un pas de temps LBM ; retourne la vorticité sur la FENÊTRE."""
        self._streaming()
        self._macroscopic()
        self._collision()
        self._boundary_conditions()
        return self._vorticity_window()

    def _vorticity_window(self):
        """Vorticité ω = ∂uy/∂x − ∂ux/∂y extraite sur la fenêtre centrale."""
        ux_w = self.ux[self.x0:self.x1, self.y0:self.y1]
        uy_w = self.uy[self.x0:self.x1, self.y0:self.y1]
        duy_dx = np.gradient(uy_w, axis=0)
        dux_dy = np.gradient(ux_w, axis=1)
        return duy_dx - dux_dy

    def compute_vorticity(self):
        return self._vorticity_window()

    # ----------------------------------------------------------
    def simulate(self, nt):
        """Simule nt pas ; retourne l'historique de vorticité sur la fenêtre."""
        vorticity_history = np.zeros(
            (nt, self.nx_win, self.ny_win), dtype=np.float32)

        print(f"\n⏳ Simulation Allée de Karman ({nt} pas) …")
        for t in range(nt):
            vorticity_history[t] = self.step()
            if (t + 1) % 50 == 0:
                u_mag = np.sqrt(self.ux**2 + self.uy**2)
                print(f"   t={t+1:4d}: U_max={u_mag.max():.4f}, "
                      f"ω_max={np.abs(vorticity_history[t]).max():.4f}")

        print("✓ Simulation terminée !")
        return vorticity_history


# ============================================================
# 2. MODÈLES DE PRÉDICTION
# ============================================================

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def train_lstm(timeseries, horizon, epochs=30, batch_size=64, device='cpu'):
    N, T = timeseries.shape
    X_train = torch.FloatTensor(timeseries[:, :-horizon]).unsqueeze(-1).to(device)

    model     = LSTMForecaster().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\n🧠 Entraînement LSTM ({epochs} epochs) …")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, N, batch_size):
            batch  = X_train[i:i+batch_size]
            target = torch.FloatTensor(
                timeseries[i:i+batch_size, -horizon]).unsqueeze(-1).to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs}  loss={total_loss/N:.6f}")
    return model


def forecast_lstm(model, timeseries, horizon, device='cpu'):
    model.eval()
    N = timeseries.shape[0]
    predictions = np.zeros((N, horizon), dtype=np.float32)
    with torch.no_grad():
        for i in range(N):
            context    = timeseries[i:i+1, :]
            pred_list  = []
            for _ in range(horizon):
                x    = torch.FloatTensor(context).unsqueeze(-1).to(device)
                pred = model(x).cpu().numpy()[0, 0]
                pred_list.append(pred)
                context = np.append(context[0, 1:], pred).reshape(1, -1)
            predictions[i] = pred_list
    return predictions


def forecast_arima(timeseries, horizon):
    N = timeseries.shape[0]
    predictions = np.zeros((N, horizon), dtype=np.float32)
    print(f"\n📊 Prédiction ARIMA ({N} séries) …")
    for i in range(N):
        try:
            fitted = ARIMA(timeseries[i], order=(1, 0, 0)).fit(method='yule_walker')
            predictions[i] = fitted.forecast(steps=horizon)
        except Exception:
            predictions[i] = timeseries[i, -1]
        if (i + 1) % 2000 == 0:
            print(f"   [{i+1}/{N}]")
    print("✓ ARIMA terminé !")
    return predictions


def volume_to_timeseries(volume):
    T, X, Y = volume.shape
    return volume.reshape(T, X * Y).T


def timeseries_to_volume(timeseries, spatial_shape):
    X, Y = spatial_shape
    return timeseries.T.reshape(-1, X, Y)


class TimesFMForecaster:
    def __init__(self, max_context=512, max_horizon=128):
        torch.set_float32_matmul_precision("high")
        print("\n🤖 Chargement TimesFM …")
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch")
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=max_context,
                max_horizon=max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=False,
                fix_quantile_crossing=True,
            ))
        print("✓ TimesFM prêt")

    def forecast(self, timeseries, horizon, batch_size=64):
        N = len(timeseries)
        point_all = []
        print(f"\n🔮 Prédiction TimesFM ({N} séries) …")
        for i in range(0, N, batch_size):
            batch  = [s.astype(np.float32) for s in timeseries[i:i+batch_size]]
            pts, _ = self.model.forecast(horizon=horizon, inputs=batch)
            point_all.append(pts)
            if (i + batch_size) % 1000 == 0:
                print(f"   [{min(i+batch_size, N)}/{N}]")
        print("✓ TimesFM terminé !")
        return np.vstack(point_all)


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

            if batch_end % 1000 == 0:
                print(f"   [{batch_end}/{N}]")

        print("✓ Chronos terminé !")
        return predictions


# ============================================================
# 3. CRÉATION DE GIFs  (données déjà dans la fenêtre)
# ============================================================

def _bw_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#000000','#0a0a0a','#1a1a1a','#404040',
              '#808080','#d0d0d0','#ffffff','#ffffff']
    cmap = LinearSegmentedColormap.from_list('karman_bw', colors, N=256)
    cmap.set_under('black')
    return cmap


def create_animation_gif(volume, title, filename,
                         fps=20, obstacle_win=None):
    """GIF sur les données déjà extraites (fenêtre centrale uniquement)."""
    print(f"\n🎬 Création du GIF : {filename} …")
    nt = volume.shape[0]

    vort_abs  = np.abs(volume)
    threshold = np.percentile(vort_abs, 85)
    vmax_sym  = np.percentile(vort_abs, 98)
    cmap      = _bw_cmap()

    fig, ax = plt.subplots(figsize=(12, 4), facecolor='black')
    ax.set_facecolor('black')

    def update(frame):
        ax.clear(); ax.set_facecolor('black')
        vf = np.abs(volume[frame])
        ve = np.clip(np.power(vf / vmax_sym, 0.6) * vmax_sym, 0, vmax_sym)
        ax.imshow(ve.T, cmap=cmap, origin='lower', aspect='auto',
                  vmin=threshold*0.2, vmax=vmax_sym, interpolation='bilinear')
        if obstacle_win is not None:
            mask = np.ma.masked_where(~obstacle_win, obstacle_win)
            ax.imshow(mask.T, cmap='gray', origin='lower',
                      aspect='auto', alpha=1.0, vmin=0, vmax=1)
        ax.set_title(f'{title} — t={frame}/{nt-1}',
                     fontsize=13, fontweight='bold', color='white')
        ax.set_xlabel('X (écoulement)', color='white')
        ax.set_ylabel('Y', color='white')
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_edgecolor('white')

    anim = FuncAnimation(fig, update, frames=nt,
                         interval=1000//fps, blit=False)
    path = os.path.join(RESULTS_DIR, f"{filename}.gif")
    anim.save(path, writer=PillowWriter(fps=fps),
              savefig_kwargs={'facecolor': 'black'})
    plt.close()
    print(f"   ✓ {filename}.gif  ({nt} frames)")


def create_comparison_gif(true_vol, pred_timesfm, pred_chronos, pred_arima, pred_lstm,
                          filename="comparison", fps=15, obstacle_win=None):
    """GIF comparatif 2×3, données déjà dans la fenêtre."""
    print(f"\n🎬 Création du GIF comparatif …")
    nt = true_vol.shape[0]

    vort_all  = np.concatenate([np.abs(v) for v in
                                 [true_vol, pred_timesfm, pred_chronos,
                                  pred_arima, pred_lstm]])
    threshold = np.percentile(vort_all, 85)
    vmax_sym  = np.percentile(vort_all, 98)
    cmap      = _bw_cmap()

    fig, axes = plt.subplots(2, 3, figsize=(20, 8), facecolor='black')
    fig.suptitle('Comparaison prédictions — Allée de Kármán',
                 fontsize=15, fontweight='bold', color='white')

    titles  = ['Ground Truth', 'TimesFM', 'Chronos', 'ARIMA', 'LSTM']
    volumes = [true_vol, pred_timesfm, pred_chronos, pred_arima, pred_lstm]

    # Masquer la 6e case (2×3 = 6, mais seulement 5 panneaux)
    axes.flat[5].set_visible(False)

    def update(frame):
        for ax, vol, title in zip(list(axes.flat)[:5], volumes, titles):
            ax.clear(); ax.set_facecolor('black')
            vf = np.abs(vol[frame])
            ve = np.clip(np.power(vf / vmax_sym, 0.6) * vmax_sym, 0, vmax_sym)
            ax.imshow(ve.T, cmap=cmap, origin='lower', aspect='auto',
                      vmin=threshold*0.2, vmax=vmax_sym, interpolation='bilinear')
            if obstacle_win is not None:
                mask = np.ma.masked_where(~obstacle_win, obstacle_win)
                ax.imshow(mask.T, cmap='gray', origin='lower',
                          aspect='auto', alpha=1.0, vmin=0, vmax=1)
            ax.set_title(f'{title}  (t={frame})', fontsize=11, color='white')
            ax.set_xlabel('X', color='white'); ax.set_ylabel('Y', color='white')
            ax.tick_params(colors='white')
            for sp in ax.spines.values(): sp.set_edgecolor('white')

    anim = FuncAnimation(fig, update, frames=nt,
                         interval=1000//fps, blit=False)
    path = os.path.join(RESULTS_DIR, f"{filename}.gif")
    anim.save(path, writer=PillowWriter(fps=fps),
              savefig_kwargs={'facecolor': 'black'})
    plt.close()
    print(f"   ✓ {filename}.gif")


def plot_errors_comparison(errors_dict, filename="errors"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    models    = list(errors_dict.keys())
    mae_vals  = [errors_dict[m]['mae']  for m in models]
    rmse_vals = [errors_dict[m]['rmse'] for m in models]
    times     = [errors_dict[m]['time'] for m in models]
    colors    = ['#FF6B6B', '#4ECDC4', '#9B59B6', '#45B7D1', '#FFA07A']

    for ax, vals, label in zip(axes,
                                [mae_vals, rmse_vals, times],
                                ['MAE', 'RMSE', "Temps (s)"]):
        ax.bar(models, vals, color=colors[:len(models)])
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{filename}.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Sauvegardé : {filename}.png")


# ============================================================
# 4. SCRIPT PRINCIPAL
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Allée de Kármán — comparaison TimesFM / Chronos / ARIMA / LSTM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- fenêtre d'analyse (ce que l'on affiche / prédit) ---
    parser.add_argument("--nx-win",  type=int, default=400,
                        help="Largeur fenêtre analyse")
    parser.add_argument("--ny-win",  type=int, default=100,
                        help="Hauteur fenêtre analyse")
    # --- marges pour éloigner les bords ---
    parser.add_argument("--pad-x",   type=int, default=100,
                        help="Marge gauche/droite (grille physique = nx-win + 2*pad-x)")
    parser.add_argument("--pad-y",   type=int, default=50,
                        help="Marge haut/bas    (grille physique = ny-win + 2*pad-y)")

    parser.add_argument("--total",   type=int, default=1000,
                        help="Pas de temps totaux")
    parser.add_argument("--horizon", type=int, default=200,
                        help="Horizon de prédiction")
    parser.add_argument("--Re",      type=float, default=150,
                        help="Nombre de Reynolds")
    parser.add_argument("--obstacle-radius", type=int, default=15)
    parser.add_argument("--skip-arima",      action="store_true")
    parser.add_argument("--skip-chronos",    action="store_true",
                        help="Sauter Chronos")
    parser.add_argument("--chronos-model",   type=str,
                        default="amazon/chronos-t5-small",
                        choices=["amazon/chronos-t5-tiny",
                                 "amazon/chronos-t5-small",
                                 "amazon/chronos-t5-base"],
                        help="Modèle Chronos à utiliser")
    parser.add_argument("--chronos-batch",   type=int, default=8,
                        help="Taille de batch pour Chronos")
    parser.add_argument("--chronos-samples", type=int, default=20,
                        help="Nombre d'échantillons Monte-Carlo pour Chronos")
    parser.add_argument("--lstm-epochs",     type=int, default=20)
    parser.add_argument("--gif-fps",         type=int, default=20)

    args = parser.parse_args()

    if args.horizon >= args.total:
        raise ValueError("horizon doit être < total")

    context_len = args.total - args.horizon

    print("=" * 70)
    print("ALLÉE DE TOURBILLONS DE VON KÁRMÁN")
    print("Comparaison : TimesFM vs Chronos vs ARIMA vs LSTM")
    print("=" * 70)
    print(f"\n  Fenêtre affichée  : {args.nx_win}×{args.ny_win}")
    print(f"  Marges (pad)      : x={args.pad_x}, y={args.pad_y}")
    print(f"  Grille physique   : "
          f"{args.nx_win+2*args.pad_x}×{args.ny_win+2*args.pad_y}")
    print(f"  Total / contexte / horizon : "
          f"{args.total} / {context_len} / {args.horizon}")
    print(f"  Re = {args.Re}")

    # --------------------------------------------------------
    # 1. SIMULATION
    # --------------------------------------------------------
    sim = KarmanVortexSimulator(
        nx_win=args.nx_win, ny_win=args.ny_win,
        pad_x=args.pad_x,   pad_y=args.pad_y,
        Re=args.Re,
        U_inlet=0.1,
        obstacle_radius=args.obstacle_radius,
    )

    t0 = time.time()
    vorticity_full = sim.simulate(args.total)   # shape (total, nx_win, ny_win)
    time_sim = time.time() - t0
    print(f"\n✓ Simulation : {time_sim:.2f}s  —  shape={vorticity_full.shape}")

    create_animation_gif(vorticity_full,
                         "Simulation exacte — Allée de Kármán",
                         "simulation_exact",
                         fps=args.gif_fps,
                         obstacle_win=sim.obstacle_win)

    # Séparer contexte / futur
    vort_context = vorticity_full[:context_len]
    vort_future  = vorticity_full[context_len:]

    ts_context   = volume_to_timeseries(vort_context)   # (N_pixels, context_len)
    spatial_shape = (args.nx_win, args.ny_win)
    print(f"\n📈 Séries temporelles : {ts_context.shape[0]} × {ts_context.shape[1]}")

    # --------------------------------------------------------
    # 2. PRÉDICTIONS
    # --------------------------------------------------------
    errors_dict = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device : {device}")

    # --- TimesFM ---
    print("\n" + "="*70)
    forecaster_tfm = TimesFMForecaster(
        max_context=context_len, max_horizon=args.horizon)
    t0 = time.time()
    pred_tfm = forecaster_tfm.forecast(ts_context, args.horizon)
    time_tfm = time.time() - t0
    vol_tfm  = timeseries_to_volume(pred_tfm, spatial_shape)
    mae_tfm  = float(np.mean(np.abs(vort_future - vol_tfm)))
    rmse_tfm = float(np.sqrt(np.mean((vort_future - vol_tfm)**2)))
    errors_dict['TimesFM'] = dict(mae=mae_tfm, rmse=rmse_tfm, time=time_tfm)
    print(f"✓ TimesFM : MAE={mae_tfm:.4f}  RMSE={rmse_tfm:.4f}  t={time_tfm:.2f}s")
    create_animation_gif(vol_tfm, "Prédiction TimesFM", "prediction_timesfm",
                         fps=args.gif_fps, obstacle_win=sim.obstacle_win)

    # --- Chronos ---
    if not args.skip_chronos:
        print("\n" + "="*70)
        forecaster_chronos = ChronosForecaster(
            model_name=args.chronos_model, device=device)
        t0 = time.time()
        pred_chronos = forecaster_chronos.forecast(
            ts_context, args.horizon,
            batch_size=args.chronos_batch,
            num_samples=args.chronos_samples,
        )
        time_chronos = time.time() - t0
        vol_chronos  = timeseries_to_volume(pred_chronos, spatial_shape)
        mae_chronos  = float(np.mean(np.abs(vort_future - vol_chronos)))
        rmse_chronos = float(np.sqrt(np.mean((vort_future - vol_chronos)**2)))
        errors_dict['Chronos'] = dict(mae=mae_chronos, rmse=rmse_chronos, time=time_chronos)
        print(f"✓ Chronos : MAE={mae_chronos:.4f}  RMSE={rmse_chronos:.4f}  t={time_chronos:.2f}s")
        create_animation_gif(vol_chronos, "Prédiction Chronos", "prediction_chronos",
                             fps=args.gif_fps, obstacle_win=sim.obstacle_win)
    else:
        print("\n⚠️  Chronos ignoré")
        vol_chronos = np.zeros_like(vort_future)
        errors_dict['Chronos'] = dict(mae=0., rmse=0., time=0.)

    # --- ARIMA ---
    if not args.skip_arima:
        print("\n" + "="*70)
        t0 = time.time()
        pred_ar = forecast_arima(ts_context, args.horizon)
        time_ar = time.time() - t0
        vol_ar  = timeseries_to_volume(pred_ar, spatial_shape)
        mae_ar  = float(np.mean(np.abs(vort_future - vol_ar)))
        rmse_ar = float(np.sqrt(np.mean((vort_future - vol_ar)**2)))
        errors_dict['ARIMA'] = dict(mae=mae_ar, rmse=rmse_ar, time=time_ar)
        print(f"✓ ARIMA : MAE={mae_ar:.4f}  RMSE={rmse_ar:.4f}  t={time_ar:.2f}s")
        create_animation_gif(vol_ar, "Prédiction ARIMA", "prediction_arima",
                             fps=args.gif_fps, obstacle_win=sim.obstacle_win)
    else:
        print("\n⚠️  ARIMA ignoré")
        vol_ar = np.zeros_like(vort_future)
        errors_dict['ARIMA'] = dict(mae=0., rmse=0., time=0.)

    # --- LSTM ---
    print("\n" + "="*70)
    t0 = time.time()
    lstm_model = train_lstm(ts_context, args.horizon,
                            epochs=args.lstm_epochs, batch_size=64, device=device)
    pred_lstm  = forecast_lstm(lstm_model, ts_context, args.horizon, device=device)
    time_lstm  = time.time() - t0
    vol_lstm   = timeseries_to_volume(pred_lstm, spatial_shape)
    mae_lstm   = float(np.mean(np.abs(vort_future - vol_lstm)))
    rmse_lstm  = float(np.sqrt(np.mean((vort_future - vol_lstm)**2)))
    errors_dict['LSTM'] = dict(mae=mae_lstm, rmse=rmse_lstm, time=time_lstm)
    print(f"✓ LSTM : MAE={mae_lstm:.4f}  RMSE={rmse_lstm:.4f}  t={time_lstm:.2f}s")
    create_animation_gif(vol_lstm, "Prédiction LSTM", "prediction_lstm",
                         fps=args.gif_fps, obstacle_win=sim.obstacle_win)

    # --------------------------------------------------------
    # 3. GIF COMPARATIF
    # --------------------------------------------------------
    create_comparison_gif(vort_future, vol_tfm, vol_chronos, vol_ar, vol_lstm,
                          filename="comparison_all",
                          fps=args.gif_fps,
                          obstacle_win=sim.obstacle_win)

    # --------------------------------------------------------
    # 4. GRAPHIQUES DE PERFORMANCE
    # --------------------------------------------------------
    plot_errors_comparison(errors_dict, "performance_comparison")

    # Évolution temporelle en 3 points de la fenêtre
    pts = [
        (args.nx_win // 2,       args.ny_win // 2,       "center"),
        (3 * args.nx_win // 4,   args.ny_win // 2,       "downstream"),
        (args.nx_win // 4 + 20,  args.ny_win // 2 + 5,   "near_obstacle"),
    ]
    for x, y, label in pts:
        fig, ax = plt.subplots(figsize=(12, 5))
        true_s = vorticity_full[:, x, y]
        ax.plot(true_s, 'k-', lw=2, label='Ground Truth', alpha=0.8)
        plot_specs = [
            ('TimesFM', vol_tfm,      'r--'),
            ('Chronos', vol_chronos,  'm-.'),
            ('ARIMA',   vol_ar,       'g--'),
            ('LSTM',    vol_lstm,     'b--'),
        ]
        for name, vol, style in plot_specs:
            if name == 'ARIMA'   and args.skip_arima:   continue
            if name == 'Chronos' and args.skip_chronos: continue
            ax.plot(range(context_len, context_len + args.horizon),
                    vol[:, x, y], style, lw=2, label=name, alpha=0.8)
        ax.axvline(context_len, color='gray', ls=':', lw=1.5, label='Forecast start')
        ax.set_xlabel('Time step', fontsize=12)
        ax.set_ylabel('Vorticity', fontsize=12)
        ax.set_title(f'Vorticity @ ({x},{y}) — {label}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"evolution_{label}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Sauvegardé : evolution_{label}.png")

    # --------------------------------------------------------
    # 5. RAPPORT FINAL
    # --------------------------------------------------------
    print("\n" + "="*70)
    print(f"{'Modèle':<12} {'MAE':>10} {'RMSE':>10} {'Temps (s)':>12}")
    print("-"*46)
    for m, d in errors_dict.items():
        print(f"{m:<12} {d['mae']:>10.4f} {d['rmse']:>10.4f} {d['time']:>12.2f}")

    best = min(errors_dict, key=lambda k: errors_dict[k]['mae'])
    fast = min(errors_dict, key=lambda k: errors_dict[k]['time'])
    print(f"\n🏆 Meilleur (MAE) : {best}  ({errors_dict[best]['mae']:.4f})")
    print(f"⚡ Plus rapide    : {fast}  ({errors_dict[fast]['time']:.2f}s)")
    print(f"\n✓ Résultats dans ./{RESULTS_DIR}/")
    print("="*70)


if __name__ == "__main__":
    main()