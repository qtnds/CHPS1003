"""
Allée de Tourbillons de Von Kármán — 3D
========================================
Simulation  : Lattice Boltzmann D3Q19
Obstacle    : cylindre d'axe Z dans écoulement X
Visualisation : Vue 3D volumétrique avec rendu de volume
Forecast    : TimesFM

Paramètres critiques pour l'allée de Kármán :
- Reynolds : 100 < Re < 300 (régime instable périodique)
- Nombre de Strouhal : St = f·D/U ≈ 0.2 pour Re~150
- Temps développement : ~500-1000 pas
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
import os, time, argparse

RESULTS_DIR = "results_karman3d"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ================================================================
# 1.  LBM  D3Q19
# ================================================================

C19 = np.array([
    [ 0, 0, 0],
    [ 1, 0, 0], [-1, 0, 0],
    [ 0, 1, 0], [ 0,-1, 0],
    [ 0, 0, 1], [ 0, 0,-1],
    [ 1, 1, 0], [-1,-1, 0],
    [ 1,-1, 0], [-1, 1, 0],
    [ 1, 0, 1], [-1, 0,-1],
    [ 1, 0,-1], [-1, 0, 1],
    [ 0, 1, 1], [ 0,-1,-1],
    [ 0, 1,-1], [ 0,-1, 1],
], dtype=np.int32)

W19 = np.array([
    1/3,
    1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
    1/36, 1/36, 1/36, 1/36,
    1/36, 1/36, 1/36, 1/36,
    1/36, 1/36, 1/36, 1/36,
], dtype=np.float64)

OPP19 = np.array([
    0,
    2, 1, 4, 3, 6, 5,
    8, 7, 10, 9,
    12, 11, 14, 13,
    16, 15, 18, 17,
], dtype=np.int32)

IDX_CX0 = [i for i in range(19) if C19[i, 0] == 0]
IDX_CXP = [i for i in range(19) if C19[i, 0] >  0]
IDX_CXN = [i for i in range(19) if C19[i, 0] <  0]


# ================================================================
# 2.  Équilibre LBM
# ================================================================

def _feq_vectorized(rho, ux, uy, uz):
    """Distributions d'équilibre D3Q19."""
    UMAX = 0.3
    ux_c = np.clip(ux, -UMAX, UMAX)
    uy_c = np.clip(uy, -UMAX, UMAX)
    uz_c = np.clip(uz, -UMAX, UMAX)
    u2   = ux_c**2 + uy_c**2 + uz_c**2

    feq = np.empty((19,) + rho.shape, dtype=np.float64)
    for i in range(19):
        eu     = C19[i,0]*ux_c + C19[i,1]*uy_c + C19[i,2]*uz_c
        feq[i] = W19[i] * rho * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*u2)
    return feq


# ================================================================
# 3.  Simulateur
# ================================================================

class KarmanVortex3D:
    """
    LBM D3Q19 - Allée de Kármán 3D.
    
    Pour observer l'allée :
    - Re entre 100-300
    - Simulation longue (>500 pas après warm-up)
    - U_inlet faible (~0.04-0.08)
    """

    def __init__(self,
                 nx_win=200, ny_win=80, nz_win=60,
                 pad_x=50,  pad_y=30,  pad_z=20,
                 Re=180,    U_inlet=0.05,
                 cyl_radius=10):

        self.nx_win, self.ny_win, self.nz_win = nx_win, ny_win, nz_win
        self.pad_x,  self.pad_y,  self.pad_z  = pad_x, pad_y, pad_z
        self.nx = nx_win + 2*pad_x
        self.ny = ny_win + 2*pad_y
        self.nz = nz_win + 2*pad_z
        self.Re      = Re
        self.U_inlet = float(U_inlet)
        self.cyl_r   = cyl_radius

        # Relaxation
        D       = 2.0 * cyl_radius
        self.nu = U_inlet * D / Re
        tau_raw = 3.0 * self.nu + 0.5
        self.tau   = float(np.clip(tau_raw, 0.55, 2.0))
        self.omega = 1.0 / self.tau

        # Strouhal (fréquence attendue)
        self.St_expected = 0.198 * (1 - 19.7/Re)  # Formule empirique
        self.period_expected = int(D / (self.St_expected * U_inlet))

        Ma = U_inlet * np.sqrt(3.0)
        print(f"\n{'='*70}")
        print(f"  LBM D3Q19 — Allée de Kármán 3D")
        print(f"{'='*70}")
        print(f"  Grille : {self.nx} × {self.ny} × {self.nz}")
        print(f"  Fenêtre : {nx_win} × {ny_win} × {nz_win}")
        print(f"  Re = {Re:.1f},  U = {U_inlet:.4f},  D = {D:.1f}")
        print(f"  ν = {self.nu:.6f},  τ = {self.tau:.4f},  ω = {self.omega:.4f}")
        print(f"  Ma = {Ma:.4f}  {'✓' if Ma < 0.15 else '⚠ TROP ÉLEVÉ'}")
        print(f"  Strouhal attendu : St ≈ {self.St_expected:.3f}")
        print(f"  Période attendue : ~{self.period_expected} pas")
        print(f"  → Warm-up recommandé : {max(500, self.period_expected*3)} pas")

        if tau_raw < 0.55:
            print(f"  ⚠ ATTENTION : τ clampé à {self.tau:.3f} (instabilité)")

        # Tableaux
        shape = (self.nx, self.ny, self.nz)
        self.f   = np.zeros((19,) + shape, dtype=np.float64)
        self.rho = np.ones(shape,  dtype=np.float64)
        self.ux  = np.zeros(shape, dtype=np.float64)
        self.uy  = np.zeros(shape, dtype=np.float64)
        self.uz  = np.zeros(shape, dtype=np.float64)

        # Cylindre
        self.obstacle, self.cyl_cx, self.cyl_cy = self._make_cylinder()
        self._init_flow()

        # Fenêtre
        self.x0, self.x1 = pad_x, pad_x + nx_win
        self.y0, self.y1 = pad_y, pad_y + ny_win
        self.z0, self.z1 = pad_z, pad_z + nz_win
        self.obstacle_win = self.obstacle[
            self.x0:self.x1, self.y0:self.y1, self.z0:self.z1]

        print(f"  Cylindre : centre fenêtre ({self.cyl_cx - self.x0}, "
              f"{self.cyl_cy - self.y0})")
        print(f"{'='*70}\n")

    def _make_cylinder(self):
        """Cylindre vertical (axe Z)."""
        cx = self.pad_x + self.nx_win // 5  # Plus en amont
        cy = self.ny // 2

        X = np.arange(self.nx)[:, None, None]
        Y = np.arange(self.ny)[None, :, None]

        mask = np.broadcast_to(
            (X - cx)**2 + (Y - cy)**2 <= self.cyl_r**2,
            (self.nx, self.ny, self.nz)
        ).copy()

        return mask, int(cx), int(cy)

    def _init_flow(self):
        """
        Initialisation avec perturbation asymétrique.
        CRUCIAL : perturbation asymétrique pour déclencher l'instabilité.
        """
        self.ux[:] = self.U_inlet
        self.uy[:] = 0.0
        self.uz[:] = 0.0
        self.rho[:] = 1.0

        # Perturbation ASYMÉTRIQUE en aval
        x_start = self.cyl_cx + self.cyl_r + 3
        if x_start < self.nx:
            # Perturbation sinusoïdale ASYMÉTRIQUE
            for i in range(x_start, self.nx):
                for j in range(self.ny):
                    # Amplitude différente haut/bas
                    amp = 0.04 * self.U_inlet if j < self.ny//2 else 0.05 * self.U_inlet
                    self.uy[i, j, :] += amp * np.sin(2 * np.pi * j / self.ny * 3)
                    
                    # Composante Z aussi
                    for k in range(self.nz):
                        self.uz[i, j, k] += 0.02 * self.U_inlet * np.sin(
                            2 * np.pi * k / self.nz * 2)

        self.f[:] = _feq_vectorized(self.rho, self.ux, self.uy, self.uz)

    def _stream(self):
        """Streaming avec copie pour éviter aliasing."""
        f_new = np.zeros_like(self.f)
        for i in range(19):
            f_new[i] = np.roll(
                np.roll(
                    np.roll(self.f[i], int(C19[i, 0]), axis=0),
                    int(C19[i, 1]), axis=1),
                int(C19[i, 2]), axis=2)
        self.f = f_new

    def _macro(self):
        """Calcul macroscopique."""
        self.rho = self.f.sum(axis=0)
        rho_safe = np.maximum(self.rho, 1e-12)
        self.ux  = (self.f * C19[:, 0, None, None, None]).sum(0) / rho_safe
        self.uy  = (self.f * C19[:, 1, None, None, None]).sum(0) / rho_safe
        self.uz  = (self.f * C19[:, 2, None, None, None]).sum(0) / rho_safe

    def _collision(self):
        """Collision BGK."""
        feq = _feq_vectorized(self.rho, self.ux, self.uy, self.uz)
        self.f += self.omega * (feq - self.f)

    def _bc(self):
        """
        Conditions aux limites CORRIGÉES.
        Le bounce-back doit se faire AVANT le streaming sur les obstacles.
        """
        # ---- Entrée Zou-He ----
        s_cx0 = sum(self.f[i, 0] for i in IDX_CX0)
        s_cxp = sum(self.f[i, 0] for i in IDX_CXP)
        rho_in = (s_cx0 + 2.0 * s_cxp) / (1.0 + self.U_inlet)

        self.rho[0] = rho_in
        self.ux[0]  = self.U_inlet
        self.uy[0]  = 0.0
        self.uz[0]  = 0.0
        feq_in = _feq_vectorized(self.rho, self.ux, self.uy, self.uz)
        self.f[:, 0, :, :] = feq_in[:, 0, :, :]

        # ---- Sortie : gradient nul ----
        self.f[:, -1, :, :] = self.f[:, -2, :, :]

        # ---- Parois Y : bounce-back ----
        for i in range(19):
            if C19[i, 1] > 0:
                self.f[OPP19[i], :, -1, :] = self.f[i, :, -1, :]
            elif C19[i, 1] < 0:
                self.f[OPP19[i], :,  0, :] = self.f[i, :,  0, :]

        # ---- Parois Z : bounce-back ----
        for i in range(19):
            if C19[i, 2] > 0:
                self.f[OPP19[i], :, :, -1] = self.f[i, :, :, -1]
            elif C19[i, 2] < 0:
                self.f[OPP19[i], :, :,  0] = self.f[i, :, :,  0]

        # ---- Cylindre : bounce-back AMÉLIORÉ ----
        # On stocke temporairement les valeurs
        f_temp = np.zeros((19,) + self.obstacle.shape, dtype=np.float64)
        for i in range(19):
            f_temp[i] = self.f[i].copy()
        
        # Bounce-back : inversion des directions
        for i in range(19):
            self.f[i][self.obstacle] = f_temp[OPP19[i]][self.obstacle]

    def _clamp_nans(self):
        """Reset cellules divergées."""
        bad = ~np.isfinite(self.f).all(axis=0)
        n_bad = int(bad.sum())
        if n_bad > 0:
            for i in range(19):
                self.f[i][bad] = W19[i]
            self.rho[bad] = 1.0
            self.ux[bad]  = self.U_inlet
            self.uy[bad]  = 0.0
            self.uz[bad]  = 0.0
        return n_bad

    def step(self):
        """Un pas LBM."""
        self._collision()
        self._bc()
        self._stream()
        self._macro()
        n_bad = self._clamp_nans()
        if n_bad > 10:
            print(f"      ⚠ {n_bad} cellules instables")

    def get_vorticity_window(self):
        """Vorticité magnitude sur fenêtre."""
        ux = self.ux[self.x0:self.x1, self.y0:self.y1, self.z0:self.z1]
        uy = self.uy[self.x0:self.x1, self.y0:self.y1, self.z0:self.z1]
        uz = self.uz[self.x0:self.x1, self.y0:self.y1, self.z0:self.z1]

        wx = np.gradient(uz, axis=1) - np.gradient(uy, axis=2)
        wy = np.gradient(ux, axis=2) - np.gradient(uz, axis=0)
        wz = np.gradient(uy, axis=0) - np.gradient(ux, axis=1)

        return np.sqrt(wx**2 + wy**2 + wz**2).astype(np.float32)

    def simulate(self, nt, record_every=1):
        """Simulation avec enregistrement."""
        n_frames = nt // record_every
        hist = np.zeros((n_frames, self.nx_win, self.ny_win, self.nz_win),
                       dtype=np.float32)
        
        print(f"\n⏳ Simulation ({nt} pas, {n_frames} frames)...")
        frame = 0
        for t in range(nt):
            self.step()
            
            if t % record_every == 0:
                hist[frame] = self.get_vorticity_window()
                frame += 1
            
            if (t + 1) % 100 == 0:
                umag = np.sqrt(self.ux**2 + self.uy**2 + self.uz**2)
                vort = self.get_vorticity_window()
                print(f"   t={t+1:5d}  |u|_max={umag.max():.5f}  "
                      f"|ω|_max={vort.max():.5f}  |ω|_mean={vort.mean():.5f}")
        
        print("✓ Simulation terminée !")
        return hist


# ================================================================
# 4.  Visualisation 3D volumétrique
# ================================================================

def _bw_cmap():
    """Colormap gris."""
    colors = ['#000000', '#101010', '#202020', '#404040',
              '#707070', '#a0a0a0', '#d0d0d0', '#ffffff']
    return LinearSegmentedColormap.from_list('bw', colors, N=256)


def _smooth(vol, sigma=1.5):
    """Lissage gaussien."""
    return gaussian_filter(vol.astype(np.float64), sigma=sigma).astype(np.float32)


def _isosurface(vol, level):
    """Marching cubes."""
    try:
        verts, faces, _, _ = marching_cubes(
            vol, level=level, spacing=(1, 1, 1), allow_degenerate=False)
        return verts, faces
    except:
        return None


def _draw_cylinder(ax, cx, cy, radius, nz, n_theta=40):
    """Cylindre blanc."""
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    xs = cx + radius * np.cos(theta)
    ys = cy + radius * np.sin(theta)

    # Latéral
    lateral = []
    for k in range(n_theta):
        k1 = (k + 1) % n_theta
        lateral.append([
            [xs[k],  ys[k],  0],
            [xs[k1], ys[k1], 0],
            [xs[k1], ys[k1], nz],
            [xs[k],  ys[k],  nz],
        ])
    ax.add_collection3d(Poly3DCollection(
        lateral, facecolor='white', edgecolor='none', alpha=0.95, zorder=100))

    # Disques
    for z in (0, nz):
        disk = [[xs[k], ys[k], z] for k in range(n_theta)]
        ax.add_collection3d(Poly3DCollection(
            [disk], facecolor='white', edgecolor='none', alpha=0.95, zorder=100))


def _setup_ax3d(ax, NX, NY, NZ, elev=20, azim=45):
    """Configuration axes 3D."""
    ax.set_facecolor('#0a0a0a')
    ax.set_xlim(0, NX)
    ax.set_ylim(0, NY)
    ax.set_zlim(0, NZ)
    ax.set_xlabel('X (écoulement)', color='white', fontsize=9)
    ax.set_ylabel('Y', color='white', fontsize=9)
    ax.set_zlabel('Z', color='white', fontsize=9)
    ax.tick_params(colors='#666666', labelsize=7)
    
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#1a1a1a')
    ax.grid(False)
    ax.view_init(elev=elev, azim=azim)


def _render_vorticity_3d(ax, vol_raw, iso_levels, cyl_cx, cyl_cy, cyl_r,
                         smooth_sigma=1.5, n_scatter=1000):
    """
    Rendu volumétrique 3D de la vorticité.
    
    Stratégie multi-couches :
    1. Scatter nuage diffus
    2. Isosurface basse (enveloppe)
    3. Isosurface haute (cœur)
    4. Cylindre blanc
    """
    NX, NY, NZ = vol_raw.shape
    vol = _smooth(vol_raw, sigma=smooth_sigma)
    
    cmap = _bw_cmap()
    iso_lo, iso_hi = iso_levels

    # 1. Scatter nuage
    mask = vol > iso_lo * 0.5
    pts = np.argwhere(mask)
    if len(pts) > n_scatter:
        idx = np.random.choice(len(pts), n_scatter, replace=False)
        pts = pts[idx]
    
    if len(pts) > 0:
        vals = vol[pts[:, 0], pts[:, 1], pts[:, 2]]
        norm = np.clip((vals - iso_lo*0.5) / (iso_hi - iso_lo*0.5 + 1e-9), 0, 1)
        colors = cmap(norm)
        colors[:, 3] = norm * 0.2
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                  c=colors, s=1.5, linewidths=0, depthshade=True, zorder=1)

    # 2. Isosurface basse
    res = _isosurface(vol, iso_lo)
    if res is not None:
        v, f = res
        ax.add_collection3d(Poly3DCollection(
            v[f], facecolor='#555555', edgecolor='none',
            alpha=0.12, zorder=2))

    # 3. Isosurface haute
    if iso_hi < vol.max():
        res = _isosurface(vol, iso_hi)
        if res is not None:
            v, f = res
            # Gradient selon X
            cx_f = v[f][:, :, 0].mean(axis=1)
            norm = (cx_f - cx_f.min()) / (cx_f.max() - cx_f.min() + 1e-9)
            fc = cmap(norm * 0.6 + 0.4)
            fc[:, 3] = 0.75
            ax.add_collection3d(Poly3DCollection(
                v[f], facecolor=fc, edgecolor='none',
                alpha=0.75, zorder=3))

    # 4. Cylindre
    _draw_cylinder(ax, cyl_cx, cyl_cy, cyl_r, NZ)


def create_3d_animation(vort_hist, sim, filename, fps=12,
                       elev=20, azim=45, iso_pcts=(55, 80)):
    """GIF 3D volumétrique."""
    print(f"\n🎬 Création GIF 3D : {filename}...")
    nt, NX, NY, NZ = vort_hist.shape

    # Calibration isosurfaces
    ref = vort_hist[nt//2:].ravel()
    iso_lo = float(np.percentile(ref, iso_pcts[0]))
    iso_hi = float(np.percentile(ref, iso_pcts[1]))
    print(f"   Isosurfaces : {iso_lo:.5f} / {iso_hi:.5f}")

    cyl_cx = sim.cyl_cx - sim.x0
    cyl_cy = sim.cyl_cy - sim.y0

    fig = plt.figure(figsize=(14, 9), facecolor='#0a0a0a')
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    def update(frame):
        ax.cla()
        _setup_ax3d(ax, NX, NY, NZ, elev, azim)
        _render_vorticity_3d(ax, vort_hist[frame], (iso_lo, iso_hi),
                            cyl_cx, cyl_cy, sim.cyl_r)
        ax.set_title(f'Allée de Kármán 3D  —  t = {frame}',
                    color='white', fontsize=12, fontweight='bold', pad=10)

    anim = FuncAnimation(fig, update, frames=nt, interval=1000//fps, blit=False)
    path = os.path.join(RESULTS_DIR, f"{filename}.gif")
    anim.save(path, writer=PillowWriter(fps=fps),
             savefig_kwargs={'facecolor': '#0a0a0a'})
    plt.close()
    print(f"   ✓ {filename}.gif ({nt} frames)")
    return path


def create_comparison_3d(true_hist, pred_hist, sim, filename, fps=10):
    """GIF comparatif 3D."""
    print(f"\n🎬 Création comparaison 3D : {filename}...")
    nt, NX, NY, NZ = true_hist.shape

    ref = true_hist[nt//2:].ravel()
    iso_lo = float(np.percentile(ref, 55))
    iso_hi = float(np.percentile(ref, 80))

    cyl_cx = sim.cyl_cx - sim.x0
    cyl_cy = sim.cyl_cy - sim.y0

    fig = plt.figure(figsize=(20, 9), facecolor='#0a0a0a')
    ax1 = fig.add_subplot(121, projection='3d', computed_zorder=False)
    ax2 = fig.add_subplot(122, projection='3d', computed_zorder=False)

    def update(frame):
        ax1.cla()
        ax2.cla()
        _setup_ax3d(ax1, NX, NY, NZ)
        _setup_ax3d(ax2, NX, NY, NZ)
        
        _render_vorticity_3d(ax1, true_hist[frame], (iso_lo, iso_hi),
                            cyl_cx, cyl_cy, sim.cyl_r)
        _render_vorticity_3d(ax2, pred_hist[frame], (iso_lo, iso_hi),
                            cyl_cx, cyl_cy, sim.cyl_r)
        
        ax1.set_title(f'Ground Truth (t={frame})',
                     color='white', fontsize=11, fontweight='bold')
        ax2.set_title(f'TimesFM (t={frame})',
                     color='#ff6b6b', fontsize=11, fontweight='bold')
        fig.suptitle('Comparaison Allée de Kármán 3D',
                    color='white', fontsize=14, y=0.96)

    anim = FuncAnimation(fig, update, frames=nt, interval=1000//fps, blit=False)
    path = os.path.join(RESULTS_DIR, f"{filename}.gif")
    anim.save(path, writer=PillowWriter(fps=fps),
             savefig_kwargs={'facecolor': '#0a0a0a'})
    plt.close()
    print(f"   ✓ {filename}.gif")
    return path


# ================================================================
# 5.  TimesFM
# ================================================================

def volume3d_to_ts(volume):
    T = volume.shape[0]
    return volume.reshape(T, -1).T


def ts_to_volume3d(ts, shape3d):
    X, Y, Z = shape3d
    return ts.T.reshape((-1, X, Y, Z))


def forecast_timesfm(ts, horizon, context_len, batch_size=64):
    import torch
    import timesfm

    torch.set_float32_matmul_precision("high")
    print("\n🤖 TimesFM...")

    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch")
    tfm.compile(timesfm.ForecastConfig(
        max_context=context_len,
        max_horizon=horizon,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    ))

    N = ts.shape[0]
    out = []
    print(f"🔮 Prédiction ({N} séries)...")
    for i in range(0, N, batch_size):
        batch = [ts[j].astype(np.float32) 
                for j in range(i, min(i + batch_size, N))]
        pts, _ = tfm.forecast(horizon=horizon, inputs=batch)
        out.append(pts)
        if (i + batch_size) % 5000 < batch_size:
            print(f"   [{min(i+batch_size, N)}/{N}]")

    return np.vstack(out)


# ================================================================
# 6.  MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx-win", type=int, default=200)
    parser.add_argument("--ny-win", type=int, default=80)
    parser.add_argument("--nz-win", type=int, default=60)
    parser.add_argument("--pad-x", type=int, default=50)
    parser.add_argument("--pad-y", type=int, default=30)
    parser.add_argument("--pad-z", type=int, default=20)
    parser.add_argument("--Re", type=float, default=180,
                       help="Reynolds (100-300 pour allée)")
    parser.add_argument("--U-inlet", type=float, default=0.05)
    parser.add_argument("--cyl-radius", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=800,
                       help="Warm-up CRUCIAL pour développer l'allée")
    parser.add_argument("--total", type=int, default=600)
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--record-every", type=int, default=2,
                       help="Enregistrer 1 frame tous les N pas")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--skip-timesfm", action="store_true")

    args = parser.parse_args()
    context_len = args.total - args.horizon

    print("=" * 70)
    print("  ALLÉE DE KÁRMÁN 3D — Vue volumétrique")
    print("=" * 70)

    # Simulation
    sim = KarmanVortex3D(
        nx_win=args.nx_win, ny_win=args.ny_win, nz_win=args.nz_win,
        pad_x=args.pad_x, pad_y=args.pad_y, pad_z=args.pad_z,
        Re=args.Re, U_inlet=args.U_inlet, cyl_radius=args.cyl_radius)

    # Warm-up ESSENTIEL
    print(f"\n🔥 Warm-up ({args.warmup} pas) — développement tourbillons...")
    for t in range(args.warmup):
        sim.step()
        if (t + 1) % 200 == 0:
            vort = sim.get_vorticity_window()
            print(f"   {t+1}/{args.warmup}  |ω|_max={vort.max():.5f}")
    print("✓ Warm-up terminé")

    # Enregistrement
    t0 = time.time()
    vort_full = sim.simulate(args.total, record_every=args.record_every)
    time_sim = time.time() - t0
    print(f"\n✓ Enregistrement : {time_sim:.1f}s  shape={vort_full.shape}")

    create_3d_animation(vort_full, sim, "karman3d_exact", fps=args.fps)

    # TimesFM
    if not args.skip_timesfm:
        n_ctx = context_len // args.record_every
        vort_ctx = vort_full[:n_ctx]
        vort_future = vort_full[n_ctx:]
        
        ts_ctx = volume3d_to_ts(vort_ctx)
        print(f"\n📊 {ts_ctx.shape[0]} séries × {ts_ctx.shape[1]} pas")

        t0 = time.time()
        pred_ts = forecast_timesfm(ts_ctx, len(vort_future), len(vort_ctx))
        time_tfm = time.time() - t0

        vol_tfm = ts_to_volume3d(pred_ts, 
                                 (args.nx_win, args.ny_win, args.nz_win))
        
        mae = np.mean(np.abs(vort_future - vol_tfm))
        rmse = np.sqrt(np.mean((vort_future - vol_tfm)**2))
        print(f"\n📈 MAE={mae:.5f}  RMSE={rmse:.5f}  t={time_tfm:.1f}s")

        create_3d_animation(vol_tfm, sim, "karman3d_timesfm", fps=args.fps)
        create_comparison_3d(vort_future, vol_tfm, sim, "comparison3d", fps=args.fps)

    print("\n" + "=" * 70)
    print(f"  ✓ Fichiers dans ./{RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()