import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import timesfm
from chronos import ChronosPipeline
import os
import argparse
import time
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. CONFIGURATION
# ============================================================

RESULTS_DIR = "results_comparison"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# PROPRIÉTÉS PHYSIQUES DES MATÉRIAUX
# ============================================================

MATERIALS = {
    "steel": {
        "name": "Acier (Steel)",
        "thermal_conductivity": 50.0,
        "density": 7850.0,
        "specific_heat": 490.0,
        "thermal_diffusivity": None
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
    k = material_dict["thermal_conductivity"]
    rho = material_dict["density"]
    cp = material_dict["specific_heat"]
    alpha = k / (rho * cp)
    material_dict["thermal_diffusivity"] = alpha
    return alpha

for material in MATERIALS.values():
    calculate_thermal_diffusivity(material)

# ============================================================
# 1. SIMULATION PHYSIQUE 3D
# ============================================================

def simulate_heat_diffusion_3d(nx=32, ny=32, nz=32, nt=200, 
                              dx=0.01, dy=0.01, dz=0.01, dt=0.1, 
                              material="steel", disruption=None):
    if material not in MATERIALS:
        raise ValueError(f"Matériau '{material}' inconnu")
    
    alpha = MATERIALS[material]["thermal_diffusivity"]
    material_name = MATERIALS[material]["name"]
    
    print(f"\nMatériau: {material_name}")
    print(f"Diffusivité thermique α = {alpha:.2e} m²/s")
    print(f"Dimensions: {nx*dx:.3f}m × {ny*dy:.3f}m × {nz*dz:.3f}m")
    
    dt_max = 1.0 / (2 * alpha * (1/dx**2 + 1/dy**2 + 1/dz**2))
    if dt > dt_max:
        print(f"⚠️  ATTENTION: dt={dt:.3e}s > dt_max={dt_max:.3e}s")
    else:
        print(f"✓ Stabilité: dt={dt:.3e}s < dt_max={dt_max:.3e}s")
    
    T = np.zeros((nt, nx, ny, nz), dtype=np.float32)
    T[0, -1, :, :] = 100.0
    
    for t in range(nt - 1):
        T_current = T[t].copy()
        laplacian = np.zeros_like(T_current)
        
        if disruption is not None and t == disruption['instant']:
            T_current[disruption['x'], disruption['y'], disruption['z']] = disruption['temp']
            print(f"\n⚡ Perturbation à t={t}: ({disruption['x']}, {disruption['y']}, {disruption['z']}) → {disruption['temp']}°C")
        
        # Laplacien (intérieur)
        laplacian[1:-1, 1:-1, 1:-1] = (
            (T_current[2:, 1:-1, 1:-1] - 2*T_current[1:-1, 1:-1, 1:-1] + T_current[:-2, 1:-1, 1:-1]) / (dx**2) +
            (T_current[1:-1, 2:, 1:-1] - 2*T_current[1:-1, 1:-1, 1:-1] + T_current[1:-1, :-2, 1:-1]) / (dy**2) +
            (T_current[1:-1, 1:-1, 2:] - 2*T_current[1:-1, 1:-1, 1:-1] + T_current[1:-1, 1:-1, :-2]) / (dz**2)
        )
        
        # Conditions aux limites (simplifiées)
        laplacian[0, 1:-1, 1:-1] = (T_current[1, 1:-1, 1:-1] - T_current[0, 1:-1, 1:-1]) / (dx**2)
        laplacian[-1, 1:-1, 1:-1] = (T_current[-2, 1:-1, 1:-1] - T_current[-1, 1:-1, 1:-1]) / (dx**2)
        laplacian[1:-1, 0, 1:-1] = (T_current[1:-1, 1, 1:-1] - T_current[1:-1, 0, 1:-1]) / (dy**2)
        laplacian[1:-1, -1, 1:-1] = (T_current[1:-1, -2, 1:-1] - T_current[1:-1, -1, 1:-1]) / (dy**2)
        laplacian[1:-1, 1:-1, 0] = (T_current[1:-1, 1:-1, 1] - T_current[1:-1, 1:-1, 0]) / (dz**2)
        laplacian[1:-1, 1:-1, -1] = (T_current[1:-1, 1:-1, -2] - T_current[1:-1, 1:-1, -1]) / (dz**2)
        
        T[t+1] = T[t] + alpha * dt * laplacian
        T[t+1, -1, :, :] = 100.0  # Source chaude
    
    return T


# ============================================================
# 2. CONVERSION VOLUME <-> SÉRIES TEMPORELLES
# ============================================================

def volume_to_timeseries(volume):
    T, X, Y, Z = volume.shape
    return volume.reshape(T, X * Y * Z).T

def timeseries_to_volume(timeseries, spatial_shape):
    N, T = timeseries.shape
    X, Y, Z = spatial_shape
    return timeseries.T.reshape(T, X, Y, Z)


# ============================================================
# 3. MODÈLES DE PRÉDICTION
# ============================================================

class TimesFMForecaster:
    def __init__(self, max_context=1024, max_horizon=256):
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
                infer_is_positive=True,
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
            
            if (i + batch_size) % 500 == 0:
                print(f"  [{min(i + batch_size, N)}/{N}] séries", end='\r')
        
        print(f"\n✓ TimesFM terminé!")
        return np.vstack(point_all)


class ChronosForecaster:
    def __init__(self, model_name="amazon/chronos-t5-small", device="cuda"):
        print(f"\n🕰️  Chargement Chronos ({model_name})...")
        self.device = device
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        print("✓ Chronos prêt")
    
    def forecast(self, timeseries, horizon, batch_size=8, num_samples=20):  # Batch réduit !
        N = len(timeseries)
        predictions = np.zeros((N, horizon), dtype=np.float32)
        
        print(f"\n🔮 Prédiction Chronos ({N} séries, batch={batch_size})...")
        
        for i in range(0, N, batch_size):
            batch_end = min(i + batch_size, N)
            
            # Convertir en tensors
            batch = [torch.tensor(series, dtype=torch.float32) for series in timeseries[i:batch_end]]
            
            try:
                # Prédiction avec gestion mémoire
                with torch.no_grad():  # Important !
                    forecast = self.pipeline.predict(
                        batch,
                        prediction_length=horizon,
                        num_samples=num_samples,
                    )
                
                # Extraire résultats
                forecast_np = forecast.numpy()
                predictions[i:batch_end] = np.median(forecast_np, axis=1)
                
                # Libérer mémoire GPU
                del forecast
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                print(f"\n⚠️  OOM à batch {i}, réduction batch_size...")
                # Fallback: traiter 1 par 1
                for j in range(i, batch_end):
                    single = [torch.tensor(timeseries[j], dtype=torch.float32)]
                    with torch.no_grad():
                        fc = self.pipeline.predict(single, prediction_length=horizon, num_samples=num_samples)
                    predictions[j] = np.median(fc.numpy(), axis=1)[0]
                    del fc
                    torch.cuda.empty_cache()
            
            if (batch_end) % 200 == 0:
                print(f"  [{batch_end}/{N}] séries", end='\r')
        
        print(f"\n✓ Chronos terminé!")
        return predictions


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
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/N:.6f}")
    
    return model


def forecast_lstm(model, timeseries, horizon, device='cpu'):
    model.eval()
    N, T = timeseries.shape
    predictions = np.zeros((N, horizon), dtype=np.float32)
    
    print(f"\n🔮 Prédiction LSTM ({N} séries)...")
    
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
            
            if (i + 1) % 1000 == 0:
                print(f"  [{i+1}/{N}] séries", end='\r')
    
    print(f"\n✓ LSTM terminé!")
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
        
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{N}] séries", end='\r')
    
    print("✓ ARIMA terminé!")
    return predictions


# ============================================================
# 4. VISUALISATION
# ============================================================

def show_3d_volume(volume, timestep, title="", filename=None):
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131)
    mid_x = volume.shape[1] // 2
    im1 = ax1.imshow(volume[timestep, mid_x, :, :].T, cmap="inferno", origin="lower", aspect="auto")
    ax1.set_title(f"Coupe Y-Z (x={mid_x})")
    ax1.set_xlabel("Y")
    ax1.set_ylabel("Z")
    plt.colorbar(im1, ax=ax1, label="°C")
    
    ax2 = fig.add_subplot(132)
    mid_z = volume.shape[3] // 2
    im2 = ax2.imshow(volume[timestep, :, :, mid_z].T, cmap="inferno", origin="lower", aspect="auto")
    ax2.set_title(f"Coupe X-Y (z={mid_z})")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    plt.colorbar(im2, ax=ax2, label="°C")
    
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
    print(f"  Sauvegardé: {filename}.png")


def compare_all_models(true_vol, pred_timesfm, pred_chronos, pred_arima, pred_lstm,
                       timestep, filename="comparison_all"):
    fig = plt.figure(figsize=(20, 8))
    
    mid_x = true_vol.shape[1] // 2
    
    vmin = min(true_vol[timestep].min(), pred_timesfm[timestep].min(),
               pred_chronos[timestep].min(), pred_arima[timestep].min(), 
               pred_lstm[timestep].min())
    vmax = max(true_vol[timestep].max(), pred_timesfm[timestep].max(),
               pred_chronos[timestep].max(), pred_arima[timestep].max(),
               pred_lstm[timestep].max())
    
    titles = ['Ground Truth', 'TimesFM', 'Chronos', 'ARIMA', 'LSTM']
    volumes = [true_vol, pred_timesfm, pred_chronos, pred_arima, pred_lstm]
    
    for i, (vol, title) in enumerate(zip(volumes, titles)):
        ax = fig.add_subplot(2, 3, i+1)
        im = ax.imshow(vol[timestep, mid_x, :, :].T, cmap='inferno',
                      origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        plt.colorbar(im, ax=ax, label='°C')
    
    # Ajouter un tableau de métriques
    ax_table = fig.add_subplot(2, 3, 6)
    ax_table.axis('off')
    
    plt.suptitle('Comparaison des modèles de prédiction', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(RESULTS_DIR, f"{filename}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Sauvegardé: {filename}.png")


def plot_errors_comparison(errors_dict, filename="performance"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = list(errors_dict.keys())
    mae_vals = [errors_dict[m]['mae'] for m in models]
    rmse_vals = [errors_dict[m]['rmse'] for m in models]
    times = [errors_dict[m]['time'] for m in models]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#96CEB4']
    
    # MAE
    axes[0, 0].bar(models, mae_vals, color=colors[:len(models)])
    axes[0, 0].set_ylabel('MAE (°C)', fontsize=12)
    axes[0, 0].set_title('Mean Absolute Error', fontsize=13, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # RMSE
    axes[0, 1].bar(models, rmse_vals, color=colors[:len(models)])
    axes[0, 1].set_ylabel('RMSE (°C)', fontsize=12)
    axes[0, 1].set_title('Root Mean Square Error', fontsize=13, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Temps
    axes[1, 0].bar(models, times, color=colors[:len(models)])
    axes[1, 0].set_ylabel('Temps (s)', fontsize=12)
    axes[1, 0].set_title('Temps d\'exécution', fontsize=13, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
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
    
    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=['Modèle', 'MAE', 'RMSE', 'Temps'],
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.25, 0.25, 0.25]
    )
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
                cell.set_facecolor(colors[i-1] + '40')
    
    plt.suptitle('Comparaison des performances', 
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    path = os.path.join(RESULTS_DIR, f"{filename}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Sauvegardé: {filename}.png")


def plot_temperature_at_point(true_vol, predictions_dict, context_len, point,
                              title="Temperature Evolution", filename="temp_evo"):
    x, y, z = point
    
    plt.figure(figsize=(14, 6))
    
    true_series = true_vol[:, x, y, z]
    T_true = len(true_series)
    
    # Contexte
    plt.plot(range(context_len), true_series[:context_len],
            'b-', linewidth=2.5, label="Contexte historique", alpha=0.8)
    
    # Vérité terrain future
    plt.plot(range(context_len - 1, T_true), true_series[context_len - 1:],
            'g-', linewidth=2.5, label="Vérité terrain", alpha=0.9)
    
    # Prédictions des différents modèles
    colors = {'TimesFM': 'r', 'Chronos': 'purple', 'ARIMA': 'orange', 'LSTM': 'cyan'}
    styles = {'TimesFM': '--', 'Chronos': '-.', 'ARIMA': ':', 'LSTM': '--'}
    
    for model_name, pred_vol in predictions_dict.items():
        pred_series = pred_vol[:, x, y, z]
        T_pred = len(pred_series)
        
        plt.plot(range(context_len, context_len + T_pred), pred_series,
                color=colors.get(model_name, 'gray'),
                linestyle=styles.get(model_name, '-'),
                linewidth=2,
                label=f"Prédiction {model_name}",
                alpha=0.8)
    
    plt.axvline(x=context_len, color='gray', linestyle=':',
               linewidth=1.5, label="Début prédiction", alpha=0.6)
    
    plt.xlabel("Pas de temps", fontsize=12)
    plt.ylabel("Température (°C)", fontsize=12)
    plt.title(f"{title} au point ({x}, {y}, {z})", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{filename}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Sauvegardé: {filename}.png")


# ============================================================
# 5. SCRIPT PRINCIPAL
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comparaison TimesFM vs Chronos vs ARIMA vs LSTM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--total", type=int, default=200,
                       help="Nombre total de pas de temps")
    parser.add_argument("--horizon", type=int, default=50,
                       help="Horizon de prédiction")
    parser.add_argument("--material", type=str, default="steel",
                       choices=list(MATERIALS.keys()))
    parser.add_argument("--nx", type=int, default=20,
                       help="Points de grille en x")
    parser.add_argument("--ny", type=int, default=20,
                       help="Points de grille en y")
    parser.add_argument("--nz", type=int, default=20,
                       help="Points de grille en z")
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--dy", type=float, default=0.01)
    parser.add_argument("--dz", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--disruption", type=str, default=None)
    parser.add_argument("--skip-arima", action="store_true",
                       help="Sauter ARIMA (lent)")
    parser.add_argument("--skip-lstm", action="store_true",
                       help="Sauter LSTM")
    parser.add_argument("--chronos-model", type=str, 
                       default="amazon/chronos-t5-small",
                       choices=["amazon/chronos-t5-tiny", 
                               "amazon/chronos-t5-small",
                               "amazon/chronos-t5-base"],
                       help="Modèle Chronos à utiliser")
    parser.add_argument("--lstm-epochs", type=int, default=20)
    
    args = parser.parse_args()
    
    if args.horizon >= args.total:
        raise ValueError(f"horizon < total requis")
    
    context_len = args.total - args.horizon
    
    # Parser disruption
    disruption = None
    if args.disruption:
        try:
            disruption_str = args.disruption.strip('[]')
            values = [float(x.strip()) for x in disruption_str.split(',')]
            
            if len(values) != 5:
                raise ValueError("5 valeurs requises")
            
            disruption = {
                'x': int(values[0]),
                'y': int(values[1]),
                'z': int(values[2]),
                'temp': values[3],
                'instant': int(values[4])
            }
            
            if not (0 <= disruption['x'] < args.nx and
                   0 <= disruption['y'] < args.ny and
                   0 <= disruption['z'] < args.nz and
                   0 <= disruption['instant'] < args.total):
                raise ValueError("Valeurs hors limites")
                
            print(f"\n🔥 Perturbation: ({disruption['x']}, {disruption['y']}, {disruption['z']}) → {disruption['temp']}°C à t={disruption['instant']}")
        
        except Exception as e:
            raise ValueError(f"Erreur parsing --disruption: {e}")
    
    print("=" * 70)
    print("COMPARAISON TIMESFM vs CHRONOS vs ARIMA vs LSTM")
    print("Diffusion thermique 3D")
    print("=" * 70)
    print(f"\nParamètres:")
    print(f"  Total: {args.total}, Contexte: {context_len}, Horizon: {args.horizon}")
    print(f"  Grille: {args.nx}×{args.ny}×{args.nz}")
    print(f"  Matériau: {args.material}")
    
    # ============================================================
    # 1. SIMULATION
    # ============================================================
    print("\n" + "=" * 70)
    print("SIMULATION PHYSIQUE")
    print("=" * 70)
    
    start_sim = time.time()
    volume = simulate_heat_diffusion_3d(
        nx=args.nx, ny=args.ny, nz=args.nz,
        nt=args.total,
        dx=args.dx, dy=args.dy, dz=args.dz,
        dt=args.dt,
        material=args.material,
        disruption=disruption
    )
    time_sim = time.time() - start_sim
    
    print(f"\n✓ Simulation: {time_sim:.2f}s, shape={volume.shape}")
    
    # Visualiser simulation
    print("\n📊 Visualisation simulation...")
    show_3d_volume(volume, -1,
                  title=f"Dernier pas contexte (t={context_len})",
                  filename="simulation_last_context")
    
    # Séparer
    volume_context = volume[:context_len]
    volume_future = volume[context_len:context_len + args.horizon]
    
    timeseries_context = volume_to_timeseries(volume_context)
    print(f"\n📈 Séries: {timeseries_context.shape[0]} × {timeseries_context.shape[1]}")
    
    # ============================================================
    # 2. PRÉDICTIONS
    # ============================================================
    errors_dict = {}
    predictions_dict = {}
    spatial_shape = (args.nx, args.ny, args.nz)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    
    # TimesFM
    print("\n" + "=" * 70)
    print("TIMESFM")
    print("=" * 70)
    
    forecaster_timesfm = TimesFMForecaster(max_context=context_len, max_horizon=args.horizon)
    start = time.time()
    pred_timesfm = forecaster_timesfm.forecast(timeseries_context, args.horizon, batch_size=64)
    time_timesfm = time.time() - start
    
    vol_pred_timesfm = timeseries_to_volume(pred_timesfm, spatial_shape)
    
    mae_tf = np.mean(np.abs(volume_future - vol_pred_timesfm))
    rmse_tf = np.sqrt(np.mean((volume_future - vol_pred_timesfm)**2))
    errors_dict['TimesFM'] = {'mae': mae_tf, 'rmse': rmse_tf, 'time': time_timesfm}
    predictions_dict['TimesFM'] = vol_pred_timesfm
    
    print(f"\n✓ TimesFM: MAE={mae_tf:.4f}°C, RMSE={rmse_tf:.4f}°C, Temps={time_timesfm:.2f}s")
    
    show_3d_volume(vol_pred_timesfm, 10,
                  title="Prédiction TimesFM",
                  filename="prediction_timesfm")
    
    # Chronos
    print("\n" + "=" * 70)
    print(f"CHRONOS ({args.chronos_model})")
    print("=" * 70)
    
    forecaster_chronos = ChronosForecaster(model_name=args.chronos_model, device=device)
    start = time.time()
    pred_chronos = forecaster_chronos.forecast(timeseries_context, args.horizon, batch_size=64)
    time_chronos = time.time() - start
    
    vol_pred_chronos = timeseries_to_volume(pred_chronos, spatial_shape)
    
    mae_ch = np.mean(np.abs(volume_future - vol_pred_chronos))
    rmse_ch = np.sqrt(np.mean((volume_future - vol_pred_chronos)**2))
    errors_dict['Chronos'] = {'mae': mae_ch, 'rmse': rmse_ch, 'time': time_chronos}
    predictions_dict['Chronos'] = vol_pred_chronos
    
    print(f"\n✓ Chronos: MAE={mae_ch:.4f}°C, RMSE={rmse_ch:.4f}°C, Temps={time_chronos:.2f}s")
    
    show_3d_volume(vol_pred_chronos, 10,
                  title="Prédiction Chronos",
                  filename="prediction_chronos")
    
    # ARIMA
    if not args.skip_arima:
        print("\n" + "=" * 70)
        print("ARIMA")
        print("=" * 70)
        
        start = time.time()
        pred_arima = forecast_arima(timeseries_context, args.horizon)
        time_arima = time.time() - start
        
        vol_pred_arima = timeseries_to_volume(pred_arima, spatial_shape)
        
        mae_ar = np.mean(np.abs(volume_future - vol_pred_arima))
        rmse_ar = np.sqrt(np.mean((volume_future - vol_pred_arima)**2))
        errors_dict['ARIMA'] = {'mae': mae_ar, 'rmse': rmse_ar, 'time': time_arima}
        predictions_dict['ARIMA'] = vol_pred_arima
        
        print(f"\n✓ ARIMA: MAE={mae_ar:.4f}°C, RMSE={rmse_ar:.4f}°C, Temps={time_arima:.2f}s")
    else:
        print("\n⚠️  ARIMA ignoré")
        vol_pred_arima = np.zeros_like(volume_future)
        predictions_dict['ARIMA'] = vol_pred_arima
    
    # LSTM
    if not args.skip_lstm:
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
        
        mae_lstm = np.mean(np.abs(volume_future - vol_pred_lstm))
        rmse_lstm = np.sqrt(np.mean((volume_future - vol_pred_lstm)**2))
        errors_dict['LSTM'] = {'mae': mae_lstm, 'rmse': rmse_lstm, 'time': time_lstm}
        predictions_dict['LSTM'] = vol_pred_lstm
        
        print(f"\n✓ LSTM: MAE={mae_lstm:.4f}°C, RMSE={rmse_lstm:.4f}°C, Temps={time_lstm:.2f}s")
    else:
        print("\n⚠️  LSTM ignoré")
        vol_pred_lstm = np.zeros_like(volume_future)
        predictions_dict['LSTM'] = vol_pred_lstm
    
    # ============================================================
    # 3. VISUALISATIONS COMPARATIVES
    # ============================================================
    print("\n" + "=" * 70)
    print("VISUALISATIONS COMPARATIVES")
    print("=" * 70)
    
    show_3d_volume(volume_future, 10,
                  title="Vérité terrain",
                  filename="ground_truth")
    
    compare_all_models(volume_future, vol_pred_timesfm, vol_pred_chronos,
                      vol_pred_arima, vol_pred_lstm,
                      timestep=10, filename="comparison_all_models")
    
    plot_errors_comparison(errors_dict, filename="performance_comparison")
    
    # Évolution temporelle en points clés
    points = [
        (args.nx // 4, args.ny // 2, args.nz // 2, "left_quarter"),
        (args.nx // 2, args.ny // 2, args.nz // 2, "center"),
        (3 * args.nx // 4, args.ny // 2, args.nz // 2, "right_quarter"),
    ]
    
    print("\n📊 Courbes d'évolution temporelle...")
    for x, y, z, label in points:
        plot_temperature_at_point(
            true_vol=volume,
            predictions_dict=predictions_dict,
            context_len=context_len,
            point=(x, y, z),
            title=f"Évolution température - {label}",
            filename=f"evolution_{label}"
        )
    
    # ============================================================
    # 4. RAPPORT FINAL
    # ============================================================
    print("\n" + "=" * 70)
    print("RAPPORT FINAL")
    print("=" * 70)
    
    print(f"\n📊 Performances:")
    print(f"\n{'Modèle':<15} {'MAE (°C)':<12} {'RMSE (°C)':<12} {'Temps (s)':<12}")
    print("-" * 55)
    
    for model_name in errors_dict:
        mae = errors_dict[model_name]['mae']
        rmse = errors_dict[model_name]['rmse']
        t = errors_dict[model_name]['time']
        print(f"{model_name:<15} {mae:<12.4f} {rmse:<12.4f} {t:<12.2f}")
    
    # Speedup par rapport à la simulation
    print(f"\n⚡ Speedup vs simulation exacte ({time_sim:.2f}s):")
    for model_name in errors_dict:
        speedup = time_sim / errors_dict[model_name]['time']
        print(f"  {model_name}: {speedup:.2f}x")
    
    # Meilleurs modèles
    best_mae = min(errors_dict, key=lambda x: errors_dict[x]['mae'])
    best_speed = min(errors_dict, key=lambda x: errors_dict[x]['time'])
    
    print(f"\n🏆 Meilleur modèle (précision): {best_mae} "
          f"(MAE={errors_dict[best_mae]['mae']:.4f}°C)")
    print(f"⚡ Modèle le plus rapide: {best_speed} "
          f"({errors_dict[best_speed]['time']:.2f}s)")
    
    print("\n" + "=" * 70)
    print(f"✓ Résultats sauvegardés dans ./{RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()