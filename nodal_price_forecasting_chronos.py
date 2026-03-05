"""
=============================================================================
Prédiction de Prix Nodaux — TimesFM vs Chronos
Optimal Power Flow — Japon 2024-2025
=============================================================================
5 scénarios de prédiction :
  1/ Journalier    — 2 pas de temps futurs (1h)
  2/ Hebdomadaire  — 7 jours (336 points)
  3/ Mensuel       — Sept 2025 à partir de Sept 2024
  4/ Multi-site    — Tokyo avec contexte multi-zones
  5/ Pic           — Pic de prix du 23/09/2024 à Chubu

Deux forecasters comparés :
  - TimesFM 2.5-200M (Google)
  - Chronos T5-small  (Amazon)

Sorties : graphiques comparatifs + dashboard de performance → results_opf_compare/
=============================================================================
"""

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time
import traceback
import warnings
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List

warnings.filterwarnings('ignore')

# ─── Dépendances optionnelles ─────────────────────────────────────────────────
try:
    import timesfm
    if hasattr(timesfm, 'TimesFM_2p5_200M_torch'):
        TIMESFM_VERSION = 2
    elif hasattr(timesfm, 'TimesFm'):
        TIMESFM_VERSION = 1
    else:
        TIMESFM_VERSION = 0
    HAS_TIMESFM = TIMESFM_VERSION > 0
    print(f"[OK] TimesFM détecté (v{TIMESFM_VERSION}).")
except ImportError:
    HAS_TIMESFM = False
    TIMESFM_VERSION = 0
    print("[WARN] TimesFM non disponible → substitut linéaire.")

try:
    from chronos import ChronosPipeline
    HAS_CHRONOS = True
    print("[OK] Chronos détecté.")
except ImportError:
    HAS_CHRONOS = False
    print("[WARN] Chronos non disponible → substitut linéaire.")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ─── Configuration ────────────────────────────────────────────────────────────
RESULTS_DIR = "results_opf_compare"
os.makedirs(RESULTS_DIR, exist_ok=True)

ZONES = ['Hokkaido', 'Tohoku', 'Tokyo', 'Chubu', 'Hokuriku',
         'Kansai', 'Chugoku', 'Shikoku', 'Kyushu']

MODEL_COLORS = {
    "TimesFM": "#E07B39",
    "Chronos":  "#3A7DC9",
    "Baseline": "#888888",
}
MODEL_MARKERS = {"TimesFM": "o", "Chronos": "s", "Baseline": "^"}


# =============================================================================
# 1. MESURE DE RESSOURCES
# =============================================================================

class ResourceMonitor:
    def __init__(self, label: str):
        self.label = label
        self.t0 = None
        self.t1 = None
        self.ram_before = None
        self.ram_after = None
        self.gpu_peak = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        if HAS_PSUTIL:
            self.ram_before = psutil.Process().memory_info().rss / 1024**2
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *_):
        self.t1 = time.perf_counter()
        if HAS_PSUTIL:
            self.ram_after = psutil.Process().memory_info().rss / 1024**2
        if torch.cuda.is_available():
            self.gpu_peak = torch.cuda.max_memory_allocated() / 1024**2

    @property
    def elapsed(self) -> float:
        return (self.t1 or time.perf_counter()) - self.t0

    @property
    def ram_delta(self) -> float:
        if self.ram_before is None or self.ram_after is None:
            return 0.0
        return self.ram_after - self.ram_before

    def summary(self) -> Dict:
        return {
            "elapsed_s": round(self.elapsed, 3),
            "ram_delta_mb": round(self.ram_delta, 1),
            "gpu_peak_mb": round(self.gpu_peak, 1),
        }


# =============================================================================
# 2. CHARGEMENT DES DONNÉES
# =============================================================================

def load_data(file_2024: str, file_2025: str) -> pd.DataFrame:
    print("\n[DATA] Chargement des données...")
    df_2024 = pd.read_csv(file_2024)
    df_2025 = pd.read_csv(file_2025)
    df = pd.concat([df_2024, df_2025], ignore_index=True)
    df.columns = df.columns.str.strip()
    df['datetime'] = (pd.to_datetime(df['Date'])
                      + pd.to_timedelta((df['n° tranche horaire'] - 1) * 30, unit='m'))
    df = df.sort_values('datetime').reset_index(drop=True)
    print(f"   [OK] {len(df)} lignes ({df['datetime'].min()} → {df['datetime'].max()})")
    return df


def _zone_col(df: pd.DataFrame, zone: str) -> str:
    for suffix in ['  (JPY/kWh)', ' (JPY/kWh)']:
        col = zone + suffix
        if col in df.columns:
            return col
    raise KeyError(f"Colonne introuvable pour la zone '{zone}'. "
                   f"Colonnes disponibles : {list(df.columns)}")


def get_zone_series(df, zone, start_date=None, end_date=None
                    ) -> Tuple[np.ndarray, np.ndarray]:
    col = _zone_col(df, zone)
    if start_date and end_date:
        mask = (df['datetime'] >= start_date) & (df['datetime'] < end_date)
        return df.loc[mask, col].values.astype(np.float32), df.loc[mask, 'datetime'].values
    return df[col].values.astype(np.float32), df['datetime'].values


# =============================================================================
# 3. FORECASTERS
# =============================================================================

class BaselineForecaster:
    """Régression linéaire locale → substitut si modèles absents."""
    name = "Baseline"

    def forecast(self, context: np.ndarray, horizon: int
                 ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        T = len(context)
        t = np.arange(T)
        coeffs = np.polyfit(t[-min(T, 48):], context[-min(T, 48):], deg=1)
        future_t = np.arange(T, T + horizon)
        pred = np.polyval(coeffs, future_t)
        mu = context[-48:].mean()
        alpha = np.linspace(0, 0.6, horizon)
        pred = pred * (1 - alpha) + mu * alpha
        return np.clip(pred, 0, None).astype(np.float32), None


class TimesFMForecaster:
    name = "TimesFM"

    def __init__(self, max_context: int = 8192, max_horizon: int = 1024):
        if not HAS_TIMESFM:
            raise RuntimeError("TimesFM non disponible")
        torch.set_float32_matmul_precision("high")
        print("\n[INFO] Chargement TimesFM...")
        if TIMESFM_VERSION == 2:
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                "google/timesfm-2.5-200m-pytorch"
            )
            self.model.compile(
                timesfm.ForecastConfig(
                    max_context=max_context,
                    max_horizon=max_horizon,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=False,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
        else:
            self.model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu" if torch.cuda.is_available() else "cpu",
                    per_core_batch_size=32,
                    horizon_len=max_horizon,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
                ),
            )
        print("[OK] TimesFM prêt.")

    def forecast(self, context: np.ndarray, horizon: int
                 ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        inputs = [context.astype(np.float32)]
        if TIMESFM_VERSION == 2:
            points, quantiles = self.model.forecast(horizon=horizon, inputs=inputs)
            return np.array(points[0], dtype=np.float32), np.array(quantiles[0], dtype=np.float32)
        else:
            _, preds = self.model.forecast([context.tolist()], freq=[0])
            return np.array(preds[0], dtype=np.float32), None


class ChronosForecaster:
    name = "Chronos"

    def __init__(self, model_name: str = "amazon/chronos-t5-small",
                 num_samples: int = 20):
        if not HAS_CHRONOS:
            raise RuntimeError("Chronos non disponible")
        self.num_samples = num_samples
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[INFO] Chargement Chronos ({model_name}) sur {device}...")
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        print("[OK] Chronos prêt.")

    def forecast(self, context: np.ndarray, horizon: int
                 ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        ctx_tensor = [torch.tensor(context, dtype=torch.float32)]
        with torch.no_grad():
            fc = self.pipeline.predict(
                ctx_tensor,
                prediction_length=horizon,
                num_samples=self.num_samples,
            )
        fc_np = fc.numpy()  # (1, num_samples, horizon)
        median = np.median(fc_np[0], axis=0).astype(np.float32)
        # Construire pseudo-quantiles [Q10, Q25, Q50, Q75, Q90]
        q10 = np.percentile(fc_np[0], 10, axis=0).astype(np.float32)
        q25 = np.percentile(fc_np[0], 25, axis=0).astype(np.float32)
        q75 = np.percentile(fc_np[0], 75, axis=0).astype(np.float32)
        q90 = np.percentile(fc_np[0], 90, axis=0).astype(np.float32)
        quantiles = np.stack([q10, q25, median, q75, q90], axis=1)
        return median, quantiles


def build_forecasters(max_context: int = 8192,
                      max_horizon: int = 1024,
                      chronos_model: str = "amazon/chronos-t5-small") -> Dict:
    forecasters = {}

    # TimesFM
    if HAS_TIMESFM:
        try:
            forecasters["TimesFM"] = TimesFMForecaster(max_context, max_horizon)
        except Exception as e:
            print(f"[WARN] TimesFM échoué : {e} → Baseline")
            forecasters["TimesFM"] = BaselineForecaster()
    else:
        forecasters["TimesFM"] = BaselineForecaster()

    # Chronos
    if HAS_CHRONOS:
        try:
            forecasters["Chronos"] = ChronosForecaster(chronos_model)
        except Exception as e:
            print(f"[WARN] Chronos échoué : {e} → Baseline")
            forecasters["Chronos"] = BaselineForecaster()
    else:
        forecasters["Chronos"] = BaselineForecaster()

    return forecasters


# =============================================================================
# 4. MÉTRIQUES
# =============================================================================

def compute_metrics(true: np.ndarray, pred: np.ndarray) -> Dict:
    true = np.asarray(true, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    mae  = float(np.mean(np.abs(true - pred)))
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    mape = float(np.mean(np.abs((true - pred) / (true + 1e-8))) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}


# =============================================================================
# 5. VISUALISATION — COURBE FORECAST (avec raccordement au dernier point hist.)
# =============================================================================

def plot_forecast_comparison(
    context: np.ndarray,
    true_future: np.ndarray,
    forecasts: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
    datetimes_context: np.ndarray,
    datetimes_future: np.ndarray,
    title: str,
    filename: str,
    zone_name: str = "",
    show_context_points: int = 96,   # nb de points d'historique affichés
):
    """
    Trace le contexte, la vérité terrain et toutes les prédictions.
    Les courbes prédites sont raccordées au DERNIER POINT de l'historique
    pour éviter tout saut visuel.
    """
    # Restreindre le contexte affiché aux N derniers points
    ctx_disp  = context[-show_context_points:]
    dt_ctx    = datetimes_context[-show_context_points:]

    # Dernier point de l'historique (point d'ancrage)
    anchor_val = float(context[-1])
    anchor_dt  = datetimes_context[-1]

    n_models = len(forecasts)
    fig_h = 4 + 1.2 * n_models
    fig, ax = plt.subplots(figsize=(15, fig_h))

    # ── Historique ────────────────────────────────────────────────────────────
    ax.plot(dt_ctx, ctx_disp, color='#1a1a2e', lw=2, label='Historique', zorder=6)

    # ── Vérité terrain ────────────────────────────────────────────────────────
    # Raccorder au dernier point historique
    dt_real  = np.concatenate([[anchor_dt], datetimes_future])
    val_real = np.concatenate([[anchor_val], true_future])
    ax.plot(dt_real, val_real, color='#2ecc71', lw=2.2,
            label='Réel (futur)', zorder=7, linestyle='-')

    # ── Prédictions ───────────────────────────────────────────────────────────
    for model_name, (pred_mean, pred_quantiles) in forecasts.items():
        color = MODEL_COLORS.get(model_name, "gray")

        # Raccordement : préfixer avec le dernier point historique
        dt_pred  = np.concatenate([[anchor_dt], datetimes_future])
        val_pred = np.concatenate([[anchor_val], pred_mean])

        # Intervalles de confiance (raccordés aussi)
        if pred_quantiles is not None and pred_quantiles.shape[1] >= 5:
            q10 = np.concatenate([[anchor_val], pred_quantiles[:, 0]])
            q90 = np.concatenate([[anchor_val], pred_quantiles[:, 4]])
            q25 = np.concatenate([[anchor_val], pred_quantiles[:, 1]])
            q75 = np.concatenate([[anchor_val], pred_quantiles[:, 3]])
            ax.fill_between(dt_pred, q10, q90, color=color, alpha=0.12,
                            label=f'{model_name} IC 80%')
            ax.fill_between(dt_pred, q25, q75, color=color, alpha=0.22,
                            label=f'{model_name} IC 50%')

        ax.plot(dt_pred, val_pred, color=color, lw=2, linestyle='--',
                marker=MODEL_MARKERS.get(model_name, 'o'),
                markevery=max(1, len(val_pred) // 12),
                markersize=5, label=f'{model_name}', zorder=8)

    # ── Ligne de coupure ──────────────────────────────────────────────────────
    ax.axvline(x=anchor_dt, color='#666', linestyle=':', lw=1.5,
               label='Début forecast', zorder=5)

    ax.set_xlabel('Date / Heure', fontsize=11)
    ax.set_ylabel('Prix nodal (JPY/kWh)', fontsize=11)
    ax.set_title(f'{title}\n{zone_name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.25)
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, f"{filename}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   [OK] {filename}.png")


# =============================================================================
# 6. DASHBOARD DE COMPARAISON
# =============================================================================

def plot_scenario_dashboard(scenario_results: Dict):
    """
    Dashboard récapitulatif par scénario :
    MAE | RMSE | MAPE | Temps | RAM | Radar synthétique
    """
    # Aplatir les résultats en DataFrame
    rows = []
    for sc_name, sc_data in scenario_results.items():
        for model_name, m in sc_data.items():
            rows.append({
                "scenario": sc_name,
                "model": model_name,
                **m,
            })
    df = pd.DataFrame(rows)

    models  = df["model"].unique().tolist()
    scenarios = df["scenario"].unique().tolist()
    colors_m = [MODEL_COLORS.get(m, "gray") for m in models]

    fig = plt.figure(figsize=(22, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── MAE par scénario ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    x  = np.arange(len(scenarios))
    w  = 0.35
    for i, (model, color) in enumerate(zip(models, colors_m)):
        vals = [df[(df.scenario == s) & (df.model == model)]["mae"].values
                for s in scenarios]
        vals = [v[0] if len(v) else np.nan for v in vals]
        offset = (i - (len(models)-1)/2) * w
        bars = ax.bar(x + offset, vals, w, label=model, color=color,
                      edgecolor='white', alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=8)
    ax.set_title("MAE par scénario (JPY/kWh)  ↓", fontsize=10, fontweight='bold')
    ax.set_ylabel("MAE"); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # ── RMSE par scénario ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for i, (model, color) in enumerate(zip(models, colors_m)):
        vals = [df[(df.scenario == s) & (df.model == model)]["rmse"].values
                for s in scenarios]
        vals = [v[0] if len(v) else np.nan for v in vals]
        offset = (i - (len(models)-1)/2) * w
        ax.bar(x + offset, vals, w, label=model, color=color,
               edgecolor='white', alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=8)
    ax.set_title("RMSE par scénario (JPY/kWh)  ↓", fontsize=10, fontweight='bold')
    ax.set_ylabel("RMSE"); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # ── MAPE par scénario ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    for i, (model, color) in enumerate(zip(models, colors_m)):
        vals = [df[(df.scenario == s) & (df.model == model)]["mape"].values
                for s in scenarios]
        vals = [v[0] if len(v) else np.nan for v in vals]
        offset = (i - (len(models)-1)/2) * w
        ax.bar(x + offset, vals, w, label=model, color=color,
               edgecolor='white', alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=8)
    ax.set_title("MAPE par scénario (%)  ↓", fontsize=10, fontweight='bold')
    ax.set_ylabel("MAPE (%)"); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # ── Temps de calcul ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    for i, (model, color) in enumerate(zip(models, colors_m)):
        vals = [df[(df.scenario == s) & (df.model == model)]["elapsed_s"].values
                for s in scenarios]
        vals = [v[0] if len(v) else np.nan for v in vals]
        offset = (i - (len(models)-1)/2) * w
        ax.bar(x + offset, vals, w, label=model, color=color,
               edgecolor='white', alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=8)
    ax.set_title("Temps de forecast (s)  ↓", fontsize=10, fontweight='bold')
    ax.set_ylabel("Secondes"); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # ── Mémoire RAM ───────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    for i, (model, color) in enumerate(zip(models, colors_m)):
        vals = [df[(df.scenario == s) & (df.model == model)]["ram_delta_mb"].values
                for s in scenarios]
        vals = [v[0] if len(v) else np.nan for v in vals]
        offset = (i - (len(models)-1)/2) * w
        ax.bar(x + offset, vals, w, label=model, color=color,
               edgecolor='white', alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=8)
    ax.set_title("ΔRAM (MB)  ↓", fontsize=10, fontweight='bold')
    ax.set_ylabel("MB"); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # ── Radar synthétique (moyennes) ──────────────────────────────────────────
    ax_r = fig.add_subplot(gs[1, 2], polar=True)
    criteria  = ["MAE\n(inv)", "RMSE\n(inv)", "MAPE\n(inv)", "Vitesse\n(inv)", "RAM\n(inv)"]
    keys_r    = ["mae", "rmse", "mape", "elapsed_s", "ram_delta_mb"]
    N_crit    = len(criteria)
    angles    = np.linspace(0, 2 * np.pi, N_crit, endpoint=False).tolist()
    angles   += angles[:1]

    def norm_inv(key, df_in, model_list):
        """Normalise et inverse (1 = meilleur)."""
        vals = {}
        for m in model_list:
            sub = df_in[df_in.model == m][key].dropna()
            vals[m] = sub.mean() if len(sub) else 0.0
        arr  = np.array(list(vals.values()))
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            return {m: 0.5 for m in model_list}
        return {m: 1 - (vals[m] - mn) / (mx - mn) for m in model_list}

    radar_scores = {k: norm_inv(k, df, models) for k in keys_r}

    for model, color in zip(models, colors_m):
        vals_r = [radar_scores[k][model] for k in keys_r]
        vals_r += vals_r[:1]
        ax_r.plot(angles, vals_r, 'o-', color=color, lw=2, label=model)
        ax_r.fill(angles, vals_r, alpha=0.12, color=color)

    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(criteria, fontsize=9)
    ax_r.set_ylim(0, 1)
    ax_r.set_title("Synthèse (plus loin = meilleur)", fontsize=10,
                   fontweight='bold', pad=20)
    ax_r.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.35, 1.1))

    # ── Tableau récapitulatif ─────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[2, :])
    ax_t.axis('off')

    col_labels = ["Scénario", "Modèle", "MAE", "RMSE", "MAPE (%)", "Temps (s)", "ΔRAM (MB)"]
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row["scenario"], row["model"],
            f"{row['mae']:.4f}", f"{row['rmse']:.4f}",
            f"{row['mape']:.2f}", f"{row['elapsed_s']:.2f}",
            f"{row['ram_delta_mb']:.0f}",
        ])

    tbl = ax_t.table(cellText=table_data, colLabels=col_labels,
                     cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)

    # Colorier les lignes par modèle
    row_colors = {"TimesFM": "#FDE9D9", "Chronos": "#DAE8FC", "Baseline": "#EEEEEE"}
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2C3E50'); cell.set_text_props(color='white', weight='bold')
        elif r <= len(table_data):
            model_val = table_data[r-1][1]
            cell.set_facecolor(row_colors.get(model_val, '#FFFFFF'))

    fig.suptitle("Benchmark OPF — TimesFM vs Chronos — Prix Nodaux Japon 2024-2025",
                 fontsize=14, fontweight='bold', y=1.01)

    path = os.path.join(RESULTS_DIR, "dashboard_benchmark.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n   [OK] Dashboard : {path}")


def plot_mae_evolution(scenario_results: Dict):
    """Courbe MAE en fonction de l'horizon (scénarios hebdo/mensuel)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, sc_key, sc_title in zip(
        axes,
        ["S2_Hebdomadaire", "S3_Mensuel"],
        ["Scénario 2 — Hebdomadaire (7j)", "Scénario 3 — Mensuel (30j)"]
    ):
        if sc_key not in scenario_results:
            ax.set_visible(False)
            continue
        sc = scenario_results[sc_key]
        plotted = False
        for model_name, m in sc.items():
            if "mae_per_step" in m:
                color = MODEL_COLORS.get(model_name, "gray")
                steps = np.arange(1, len(m["mae_per_step"]) + 1)
                ax.plot(steps, m["mae_per_step"], color=color, lw=2,
                        marker=MODEL_MARKERS.get(model_name, 'o'),
                        markevery=max(1, len(steps) // 10),
                        markersize=5, label=model_name)
                plotted = True
        if plotted:
            ax.set_title(sc_title, fontsize=11, fontweight='bold')
            ax.set_xlabel("Pas de forecast"); ax.set_ylabel("MAE (JPY/kWh)")
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle("Évolution de la MAE en fonction de l'horizon de prédiction",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "mae_evolution_horizon.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   [OK] MAE evolution : {path}")


# =============================================================================
# 7. SCÉNARIOS
# =============================================================================

def _run_forecast_for_all_models(
    forecasters: Dict,
    context: np.ndarray,
    horizon: int,
    label: str = "",
) -> Tuple[Dict, Dict]:
    """Lance les forecasters et retourne {model: (pred, quant)} et {model: resource_stats}."""
    results  = {}
    res_stats = {}
    for model_name, fc in forecasters.items():
        try:
            with ResourceMonitor(model_name) as mon:
                pred, quant = fc.forecast(context, horizon)
            results[model_name]   = (pred, quant)
            res_stats[model_name] = mon.summary()
        except Exception as e:
            print(f"   [WARN] {model_name} échoué ({label}) : {e}")
            traceback.print_exc()
    return results, res_stats


# ─────────────────────────────────────────────────────────────────────────────

def scenario_1_daily(df, forecasters) -> Dict:
    """Scénario 1 — Prédiction journalière, 2 pas de temps."""
    print("\n" + "=" * 70)
    print("SCÉNARIO 1 : PRÉDICTION JOURNALIÈRE (horizon = 2 pas / 1h)")
    print("=" * 70)

    test_cases = [
        ('2024-04-15', 'Tokyo',    20),
        ('2024-07-20', 'Hokkaido', 30),
        ('2024-10-10', 'Kansai',   25),
        ('2025-01-05', 'Chubu',    35),
        ('2025-02-14', 'Kyushu',   28),
    ]

    all_metrics: Dict[str, Dict] = {m: {"mae": [], "rmse": [], "mape": [],
                                        "elapsed_s": [], "ram_delta_mb": []}
                                    for m in forecasters}

    for date_str, zone, ctx_h in test_cases:
        print(f"\n   📅 {date_str} — {zone} (contexte {ctx_h}h)")
        date = pd.to_datetime(date_str)
        start_ctx = date - timedelta(hours=ctx_h)
        series, datetimes = get_zone_series(df, zone, start_ctx,
                                            date + timedelta(days=1))
        n_ctx = ctx_h * 2
        context = series[:n_ctx]
        true_fut = series[n_ctx:n_ctx + 2]
        dt_ctx = datetimes[:n_ctx]
        dt_fut = datetimes[n_ctx:n_ctx + 2]

        forecasts, res_stats = _run_forecast_for_all_models(
            forecasters, context, 2, label=f"S1-{date_str}-{zone}"
        )

        for model_name, (pred, quant) in forecasts.items():
            m = compute_metrics(true_fut, pred)
            for k in ["mae", "rmse", "mape"]:
                all_metrics[model_name][k].append(m[k])
            for k in ["elapsed_s", "ram_delta_mb"]:
                all_metrics[model_name][k].append(res_stats[model_name].get(k, 0))
            print(f"   {model_name:<10} MAE={m['mae']:.4f}  MAPE={m['mape']:.2f}%"
                  f"  t={res_stats[model_name]['elapsed_s']:.2f}s")

        # Graphique comparatif pour ce sous-cas
        plot_forecast_comparison(
            context, true_fut, forecasts,
            dt_ctx, dt_fut,
            f"Prédiction journalière — {date_str}",
            f"s1_daily_{date_str}_{zone}",
            zone_name=zone,
            show_context_points=48,
        )

    # Agréger
    out: Dict[str, Dict] = {}
    for model_name, m in all_metrics.items():
        out[model_name] = {
            "mae":          float(np.mean(m["mae"])),
            "rmse":         float(np.mean(m["rmse"])),
            "mape":         float(np.mean(m["mape"])),
            "elapsed_s":    float(np.mean(m["elapsed_s"])),
            "ram_delta_mb": float(np.mean(m["ram_delta_mb"])),
            "gpu_peak_mb":  0.0,
        }
        print(f"\n   Résumé {model_name} — MAE moy={out[model_name]['mae']:.4f}"
              f"  MAPE={out[model_name]['mape']:.2f}%")
    return out


def scenario_2_weekly(df, forecasters) -> Dict:
    """Scénario 2 — Prédiction hebdomadaire (7j)."""
    print("\n" + "=" * 70)
    print("SCÉNARIO 2 : PRÉDICTION HEBDOMADAIRE (7 jours)")
    print("=" * 70)

    test_start = pd.to_datetime('2025-03-03')
    test_end   = test_start + timedelta(days=7)
    ctx_start  = test_start - timedelta(days=21)
    n_ctx = 21 * 48
    n_hor = 7  * 48

    out: Dict[str, Dict] = {}

    for zone in ZONES[:3]:
        print(f"\n   📍 Zone : {zone}")
        series, datetimes = get_zone_series(df, zone, ctx_start, test_end)
        context  = series[:n_ctx]
        true_fut = series[n_ctx:n_ctx + n_hor]
        dt_ctx   = datetimes[:n_ctx]
        dt_fut   = datetimes[n_ctx:n_ctx + n_hor]

        forecasts, res_stats = _run_forecast_for_all_models(
            forecasters, context, n_hor, label=f"S2-{zone}"
        )

        for model_name, (pred, quant) in forecasts.items():
            m = compute_metrics(true_fut, pred)
            mae_step = np.abs(true_fut - pred)
            print(f"   {model_name:<10} MAE={m['mae']:.4f}  MAPE={m['mape']:.2f}%"
                  f"  t={res_stats[model_name]['elapsed_s']:.2f}s")

            if model_name not in out:
                out[model_name] = {
                    "mae": 0.0, "rmse": 0.0, "mape": 0.0,
                    "elapsed_s": 0.0, "ram_delta_mb": 0.0, "gpu_peak_mb": 0.0,
                    "mae_per_step": mae_step.tolist(),
                    "_count": 0,
                }
            c = out[model_name]["_count"] + 1
            for k in ["mae", "rmse", "mape"]:
                out[model_name][k] = (out[model_name][k] * out[model_name]["_count"] + m[k]) / c
            for k in ["elapsed_s", "ram_delta_mb"]:
                out[model_name][k] = (out[model_name][k] * out[model_name]["_count"]
                                      + res_stats[model_name].get(k, 0)) / c
            out[model_name]["_count"] = c

        # Graphique pour cette zone
        plot_forecast_comparison(
            context, true_fut, forecasts, dt_ctx, dt_fut,
            f"Prédiction hebdomadaire — {test_start.date()}",
            f"s2_weekly_{zone}",
            zone_name=zone,
            show_context_points=7*48,
        )

    for m in out:
        out[m].pop("_count", None)
    return out


def scenario_3_monthly(df, forecasters) -> Dict:
    """Scénario 3 — Mensuel : Sept 2025 à partir de Sept 2024."""
    print("\n" + "=" * 70)
    print("SCÉNARIO 3 : PRÉDICTION MENSUELLE (Sept 2025)")
    print("=" * 70)

    zone = 'Tokyo'
    ctx_start  = pd.to_datetime('2024-09-11')   # 20 jours de contexte fin sept 2024
    ctx_end    = pd.to_datetime('2024-10-01')
    fut_start  = pd.to_datetime('2025-09-11')
    fut_end    = pd.to_datetime('2025-10-01')

    series_ctx, dt_ctx = get_zone_series(df, zone, ctx_start, ctx_end)
    series_fut, dt_fut = get_zone_series(df, zone, fut_start, fut_end)
    n_hor = len(series_fut)

    print(f"   Contexte : {len(series_ctx)} pts  |  Horizon : {n_hor} pts "
          f"({n_hor/48:.1f}j)")

    forecasts, res_stats = _run_forecast_for_all_models(
        forecasters, series_ctx, n_hor, label="S3-mensuel"
    )

    out: Dict[str, Dict] = {}
    for model_name, (pred, quant) in forecasts.items():
        m = compute_metrics(series_fut, pred)
        mae_step = np.abs(series_fut - pred).tolist()
        print(f"   {model_name:<10} MAE={m['mae']:.4f}  MAPE={m['mape']:.2f}%"
              f"  t={res_stats[model_name]['elapsed_s']:.2f}s")
        out[model_name] = {**m, **res_stats[model_name], "mae_per_step": mae_step}

    plot_forecast_comparison(
        series_ctx, series_fut, forecasts, dt_ctx, dt_fut,
        "Prédiction mensuelle — Sept 2025 (contexte fin Sept 2024)",
        "s3_monthly_sept2025_Tokyo",
        zone_name=zone,
        show_context_points=len(series_ctx),
    )
    return out


def scenario_4_multisite(df, forecasters) -> Dict:
    """Scénario 4 — Multi-site : Tokyo + 4 zones de contexte."""
    print("\n" + "=" * 70)
    print("SCÉNARIO 4 : PRÉDICTION MULTI-SITE (Tokyo + 4 zones)")
    print("=" * 70)

    target_zone  = 'Tokyo'
    other_zones  = [z for z in ZONES if z != target_zone][:4]
    test_start   = pd.to_datetime('2025-02-01')
    test_end     = test_start + timedelta(days=7)
    tokyo_ctx_s  = test_start - timedelta(days=28)
    other_ctx_s  = test_start - timedelta(days=35)

    tokyo_s, dt_tokyo = get_zone_series(df, target_zone, tokyo_ctx_s, test_start)
    context_combined = tokyo_s.copy()
    for oz in other_zones:
        oz_s, _ = get_zone_series(df, oz, other_ctx_s, test_start)
        context_combined = np.concatenate([context_combined, oz_s])

    true_fut, dt_fut = get_zone_series(df, target_zone, test_start, test_end)
    n_hor = len(true_fut)
    print(f"   Contexte combiné : {len(context_combined)} pts  |  Horizon : {n_hor} pts")

    forecasts, res_stats = _run_forecast_for_all_models(
        forecasters, context_combined, n_hor, label="S4-multisite"
    )

    out: Dict[str, Dict] = {}
    for model_name, (pred, quant) in forecasts.items():
        m = compute_metrics(true_fut, pred)
        print(f"   {model_name:<10} MAE={m['mae']:.4f}  MAPE={m['mape']:.2f}%"
              f"  t={res_stats[model_name]['elapsed_s']:.2f}s")
        out[model_name] = {**m, **res_stats[model_name]}

    plot_forecast_comparison(
        tokyo_s, true_fut, forecasts, dt_tokyo, dt_fut,
        f"Prédiction multi-site — {target_zone}",
        f"s4_multisite_{target_zone}",
        zone_name=f"{target_zone} (contexte : {', '.join(other_zones)})",
        show_context_points=7*48,
    )
    return out


def scenario_5_peak(df, forecasters) -> Dict:
    """Scénario 5 — Prédiction de pic : 23/09/2024 Chubu."""
    print("\n" + "=" * 70)
    print("SCÉNARIO 5 : PRÉDICTION DE PIC (23/09/2024 — Chubu)")
    print("=" * 70)

    zone = 'Chubu'
    peak_date = pd.to_datetime('2024-09-23')
    ctx_start = peak_date - timedelta(days=14)
    pred_s    = peak_date - timedelta(days=1)
    pred_e    = peak_date + timedelta(days=2)

    ctx_series, dt_ctx = get_zone_series(df, zone, ctx_start, pred_s)
    fut_series, dt_fut = get_zone_series(df, zone, pred_s, pred_e)
    n_hor = len(fut_series)

    peak_val = float(fut_series.max())
    peak_idx = int(fut_series.argmax())
    print(f"   Pic réel : {peak_val:.2f} JPY/kWh  @ t+{peak_idx}")

    forecasts, res_stats = _run_forecast_for_all_models(
        forecasters, ctx_series, n_hor, label="S5-peak"
    )

    out: Dict[str, Dict] = {}
    for model_name, (pred, quant) in forecasts.items():
        m = compute_metrics(fut_series, pred)
        peak_pred = float(pred[peak_idx])
        peak_err  = abs(peak_val - peak_pred)
        print(f"   {model_name:<10} MAE={m['mae']:.4f}  MAPE={m['mape']:.2f}%"
              f"  Pic prédit={peak_pred:.2f} (err={peak_err:.2f}, {100*peak_err/peak_val:.1f}%)"
              f"  t={res_stats[model_name]['elapsed_s']:.2f}s")
        out[model_name] = {
            **m, **res_stats[model_name],
            "peak_value": peak_val, "peak_pred": peak_pred,
            "peak_error": peak_err, "peak_error_pct": 100 * peak_err / peak_val,
        }

    plot_forecast_comparison(
        ctx_series, fut_series, forecasts, dt_ctx, dt_fut,
        f"Prédiction de pic — {peak_date.date()}",
        f"s5_peak_{zone}",
        zone_name=f"{zone}  (pic réel : {peak_val:.2f} JPY/kWh)",
        show_context_points=7*48,
    )
    return out


# =============================================================================
# 8. RAPPORT TEXTE
# =============================================================================

def save_text_report(scenario_results: Dict):
    path = os.path.join(RESULTS_DIR, "rapport_benchmark_opf.txt")
    sep  = "=" * 72
    with open(path, 'w', encoding='utf-8') as f:
        f.write(sep + "\n")
        f.write("BENCHMARK OPF — TIMESFM vs CHRONOS — PRIX NODAUX JAPON\n")
        f.write(sep + "\n\n")
        header = f"{'Scénario':<25} {'Modèle':<12} {'MAE':>8} {'RMSE':>8} {'MAPE(%)':>9} {'t(s)':>8} {'RAM(MB)':>9}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")
        for sc, models in scenario_results.items():
            for model, m in models.items():
                f.write(f"{sc:<25} {model:<12} "
                        f"{m['mae']:>8.4f} {m['rmse']:>8.4f} {m['mape']:>9.2f} "
                        f"{m.get('elapsed_s', 0):>8.2f} {m.get('ram_delta_mb', 0):>9.0f}\n")
        f.write("\n")

        # Meilleur par critère
        f.write("MEILLEURS PAR CRITÈRE (moyenne toutes zones/cas) :\n")
        from collections import defaultdict
        totals: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for sc, models in scenario_results.items():
            for model, m in models.items():
                for k in ["mae", "rmse", "mape", "elapsed_s"]:
                    totals[model][k].append(m.get(k, np.nan))
        means = {m: {k: np.nanmean(v) for k, v in totals[m].items()} for m in totals}
        for crit, label in [("mae", "MAE ↓"), ("mape", "MAPE ↓"), ("elapsed_s", "Vitesse ↓")]:
            best = min(means, key=lambda m: means[m].get(crit, 1e9))
            f.write(f"  {label:<15}: {best}  (moy={means[best].get(crit, 0):.4f})\n")

    print(f"   [OK] Rapport texte : {path}")


# =============================================================================
# 9. PIPELINE PRINCIPALE
# =============================================================================

def run_pipeline(
    file_2024: str = "spot_summary_2024.csv",
    file_2025: str = "spot_summary_2025.csv",
    chronos_model: str = "amazon/chronos-t5-small",
    max_context: int = 8192,
    max_horizon: int = 1024,
    skip_scenarios: Optional[List[int]] = None,
):
    print("=" * 70)
    print("  BENCHMARK OPF — TimesFM vs Chronos — Prix Nodaux Japon 2024-2025")
    print("=" * 70)
    print(f"  TimesFM   : {HAS_TIMESFM} (v{TIMESFM_VERSION})")
    print(f"  Chronos   : {HAS_CHRONOS} ({chronos_model})")
    print(f"  Device    : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  Output    : {os.path.abspath(RESULTS_DIR)}")
    print("=" * 70)

    skip = set(skip_scenarios or [])

    # Données
    df = load_data(file_2024, file_2025)

    # Forecasters
    print("\n[STEP 1] Initialisation des forecasters...")
    forecasters = build_forecasters(max_context, max_horizon, chronos_model)

    # Scénarios
    scenario_results: Dict[str, Dict] = {}

    if 1 not in skip:
        scenario_results["S1_Journalier"]   = scenario_1_daily(df, forecasters)
    if 2 not in skip:
        scenario_results["S2_Hebdomadaire"] = scenario_2_weekly(df, forecasters)
    if 3 not in skip:
        scenario_results["S3_Mensuel"]      = scenario_3_monthly(df, forecasters)
    if 4 not in skip:
        scenario_results["S4_MultiSite"]    = scenario_4_multisite(df, forecasters)
    if 5 not in skip:
        scenario_results["S5_Pic"]          = scenario_5_peak(df, forecasters)

    # Visualisations globales
    print("\n[STEP 2] Génération des visualisations globales...")
    if scenario_results:
        plot_scenario_dashboard(scenario_results)
        plot_mae_evolution(scenario_results)
        save_text_report(scenario_results)

    # Résumé console
    print("\n" + "=" * 70)
    print("RÉSUMÉ FINAL")
    print("=" * 70)
    print(f"{'Scénario':<25} {'Modèle':<12} {'MAE':>8} {'MAPE%':>7} {'t(s)':>8}")
    print("-" * 65)
    for sc, models in scenario_results.items():
        for model, m in models.items():
            print(f"{sc:<25} {model:<12} {m['mae']:>8.4f} {m['mape']:>7.2f} "
                  f"{m.get('elapsed_s', 0):>8.2f}")

    print("\n" + "=" * 70)
    print(f"  Résultats : {os.path.abspath(RESULTS_DIR)}/")
    print("=" * 70)
    for f in sorted(os.listdir(RESULTS_DIR)):
        sz = os.path.getsize(os.path.join(RESULTS_DIR, f)) // 1024
        print(f"   {f:<50}  {sz:>5} Ko")
    print("\n[DONE]")


# =============================================================================
# 10. POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Benchmark OPF — TimesFM vs Chronos — Prix Nodaux Japon"
    )
    parser.add_argument('--file-2024', default='spot_summary_2024.csv')
    parser.add_argument('--file-2025', default='spot_summary_2025.csv')
    parser.add_argument('--chronos-model', default='amazon/chronos-t5-small',
                        choices=['amazon/chronos-t5-tiny',
                                 'amazon/chronos-t5-small',
                                 'amazon/chronos-t5-base',
                                 'amazon/chronos-t5-large'])
    parser.add_argument('--max-context', type=int, default=8192)
    parser.add_argument('--max-horizon', type=int, default=1024)
    parser.add_argument('--skip', type=int, nargs='*', default=[],
                        help='Numéros de scénarios à ignorer (ex: --skip 3 5)')
    args = parser.parse_args()

    run_pipeline(
        file_2024=args.file_2024,
        file_2025=args.file_2025,
        chronos_model=args.chronos_model,
        max_context=args.max_context,
        max_horizon=args.max_horizon,
        skip_scenarios=args.skip,
    )