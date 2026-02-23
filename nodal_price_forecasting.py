import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import timesfm
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. CONFIGURATION
# ============================================================

RESULTS_DIR = "results_opf"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Zones géographiques
ZONES = ['Hokkaido', 'Tohoku', 'Tokyo', 'Chubu', 'Hokuriku', 
         'Kansai', 'Chugoku', 'Shikoku', 'Kyushu']

# ============================================================
# 1. CHARGEMENT ET PREPROCESSING DES DONNÉES
# ============================================================

def load_data(file_2024, file_2025):
    """Charge et nettoie les données de prix nodaux"""
    print("\n📁 Chargement des données...")
    
    # Lire les CSV
    df_2024 = pd.read_csv(file_2024)
    df_2025 = pd.read_csv(file_2025)
    
    # Combiner
    df = pd.concat([df_2024, df_2025], ignore_index=True)
    
    # Nettoyer les noms de colonnes (enlever espaces)
    df.columns = df.columns.str.strip()
    
    # Créer une colonne datetime
    df['datetime'] = pd.to_datetime(df['Date']) + pd.to_timedelta((df['n° tranche horaire'] - 1) * 30, unit='m')
    
    # Trier par date
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"✓ Chargé: {len(df)} lignes ({df['datetime'].min()} → {df['datetime'].max()})")
    print(f"  Zones: {ZONES}")
    
    return df


def get_zone_series(df, zone, start_date=None, end_date=None):
    """Extrait la série temporelle d'une zone"""
    zone_col = zone + '  (JPY/kWh)' if zone + '  (JPY/kWh)' in df.columns else zone + ' (JPY/kWh)'
    
    if start_date and end_date:
        mask = (df['datetime'] >= start_date) & (df['datetime'] < end_date)
        return df.loc[mask, zone_col].values, df.loc[mask, 'datetime'].values
    else:
        return df[zone_col].values, df['datetime'].values


def get_day_data(df, date_str, zone):
    """Récupère les 48 points d'une journée pour une zone"""
    date = pd.to_datetime(date_str)
    next_day = date + timedelta(days=1)
    series, datetimes = get_zone_series(df, zone, date, next_day)
    return series, datetimes


# ============================================================
# 2. TIMESFM FORECASTER
# ============================================================

class TimesFMForecaster:
    def __init__(self, max_context=2048, max_horizon=512):
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
                force_flip_invariance=False,
                infer_is_positive=True,  # Prix toujours positifs
                fix_quantile_crossing=True,
            )
        )
        print("✓ TimesFM chargé et compilé")
    
    def forecast_single(self, context, horizon):
        """Prédiction pour une seule série"""
        inputs = [context.astype(np.float32)]
        points, quantiles = self.model.forecast(horizon=horizon, inputs=inputs)
        return points[0], quantiles[0]
    
    def forecast_batch(self, contexts, horizon):
        """Prédiction pour plusieurs séries"""
        inputs = [c.astype(np.float32) for c in contexts]
        points, quantiles = self.model.forecast(horizon=horizon, inputs=inputs)
        return points, quantiles


# ============================================================
# 3. FONCTION DE VISUALISATION
# ============================================================

def plot_forecast(context, true_future, pred_mean, pred_quantiles, 
                 datetimes_context, datetimes_future,
                 title, filename, zone_name=""):
    """
    Visualise la prédiction avec contexte, vérité terrain et intervalles de confiance
    
    pred_quantiles: shape (horizon, n_quantiles) avec quantiles [0.1, 0.25, 0.5, 0.75, 0.9]
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Contexte
    ax.plot(datetimes_context, context, 'b-', linewidth=2, 
            label='Contexte (historique)', alpha=0.7)
    
    # Vérité terrain
    ax.plot(datetimes_future, true_future, 'g-', linewidth=2, 
            label='Réel', alpha=0.8)
    
    # Prédiction moyenne
    ax.plot(datetimes_future, pred_mean, 'r--', linewidth=2, 
            label='Prédiction TimesFM', alpha=0.8)
    
    # Intervalles de confiance (quantiles)
    if pred_quantiles is not None and pred_quantiles.shape[1] >= 5:
        # 80% intervalle (quantiles 10% et 90%)
        ax.fill_between(datetimes_future, 
                       pred_quantiles[:, 0],  # Q10
                       pred_quantiles[:, 4],  # Q90
                       color='red', alpha=0.15, label='IC 80%')
        
        # 50% intervalle (quantiles 25% et 75%)
        ax.fill_between(datetimes_future,
                       pred_quantiles[:, 1],  # Q25
                       pred_quantiles[:, 3],  # Q75
                       color='red', alpha=0.25, label='IC 50%')
    
    # Ligne de séparation
    if len(datetimes_context) > 0 and len(datetimes_future) > 0:
        ax.axvline(x=datetimes_context[-1], color='gray', 
                  linestyle=':', linewidth=1.5, label='Début prédiction')
    
    ax.set_xlabel('Date/Heure', fontsize=12)
    ax.set_ylabel('Prix nodal (JPY/kWh)', fontsize=12)
    ax.set_title(f'{title}\n{zone_name}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Rotation des labels de date
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{filename}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Sauvegardé: {filename}.png")


def compute_metrics(true, pred):
    """Calcule MAE, RMSE, MAPE"""
    mae = np.mean(np.abs(true - pred))
    rmse = np.sqrt(np.mean((true - pred)**2))
    mape = np.mean(np.abs((true - pred) / (true + 1e-8))) * 100
    return mae, rmse, mape


# ============================================================
# 4. SCÉNARIOS DE PRÉDICTION
# ============================================================

def scenario_1_daily(df, forecaster):
    """
    Scénario 1: Prédiction journalière - 2 pas de temps suivants
    5 jours différents, zones différentes
    """
    print("\n" + "=" * 70)
    print("SCÉNARIO 1: PRÉDICTION JOURNALIÈRE (2 pas futurs)")
    print("=" * 70)
    
    # 5 jours de test différents avec zones variées
    test_cases = [
        ('2024-04-15', 'Tokyo', 20),      # Printemps
        ('2024-07-20', 'Hokkaido', 30),   # Été
        ('2024-10-10', 'Kansai', 25),     # Automne
        ('2025-01-05', 'Chubu', 35),      # Hiver
        ('2025-02-14', 'Kyushu', 28),     # Hiver tardif
    ]
    
    results = []
    
    for date_str, zone, context_hours in test_cases:
        print(f"\n📅 {date_str} - {zone} (contexte: {context_hours}h)")
        
        date = pd.to_datetime(date_str)
        
        # Contexte: context_hours avant
        start_context = date - timedelta(hours=context_hours)
        series, datetimes = get_zone_series(df, zone, start_context, date + timedelta(days=1))
        
        # Séparer contexte et futur
        n_context = context_hours * 2  # 2 points par heure
        context = series[:n_context]
        true_future = series[n_context:n_context + 2]  # 2 pas (1h)
        
        datetimes_context = datetimes[:n_context]
        datetimes_future = datetimes[n_context:n_context + 2]
        print("Taille context:", len(context))
        print("Context head:", context[:5])
        print("Nb NaN:", np.isnan(context).sum())
        # Prédire
        pred_mean, pred_quantiles = forecaster.forecast_single(context, horizon=2)
        
        # Métriques
        mae, rmse, mape = compute_metrics(true_future, pred_mean)
        results.append({'date': date_str, 'zone': zone, 'mae': mae, 'rmse': rmse, 'mape': mape})
        
        print(f"   MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
        
        # Visualiser
        plot_forecast(context, true_future, pred_mean, pred_quantiles,
                     datetimes_context, datetimes_future,
                     f"Prédiction journalière - {date_str}",
                     f"daily_{date_str}_{zone}",
                     zone_name=zone)
    
    # Résumé
    df_results = pd.DataFrame(results)
    print(f"\n📊 Résumé Scénario 1:")
    print(f"   MAE moyen: {df_results['mae'].mean():.4f} JPY/kWh")
    print(f"   MAPE moyen: {df_results['mape'].mean():.2f}%")
    
    return df_results


def scenario_2_weekly(df, forecaster):
    """
    Scénario 2: Prédiction hebdomadaire
    Prédire 1 semaine (7*48 = 336 points) pour toutes les zones
    """
    print("\n" + "=" * 70)
    print("SCÉNARIO 2: PRÉDICTION HEBDOMADAIRE (7 jours)")
    print("=" * 70)
    
    # Semaine de test: 2025-03-03 à 2025-03-09
    test_start = pd.to_datetime('2025-03-03')
    test_end = test_start + timedelta(days=7)
    
    # Contexte: 3 semaines avant
    context_start = test_start - timedelta(days=21)
    
    results = []
    
    for zone in ZONES[:3]:  # Tester 3 zones pour rapidité
        print(f"\n📍 Zone: {zone}")
        
        # Récupérer les données
        series, datetimes = get_zone_series(df, zone, context_start, test_end)
        
        n_context = 21 * 48  # 3 semaines
        n_horizon = 7 * 48   # 1 semaine
        
        context = series[:n_context]
        true_future = series[n_context:n_context + n_horizon]
        
        datetimes_context = datetimes[:n_context]
        datetimes_future = datetimes[n_context:n_context + n_horizon]
        
        # Prédire
        pred_mean, pred_quantiles = forecaster.forecast_single(context, horizon=n_horizon)
        
        # Métriques
        mae, rmse, mape = compute_metrics(true_future, pred_mean)
        results.append({'zone': zone, 'mae': mae, 'rmse': rmse, 'mape': mape})
        
        print(f"   MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
        
        # Visualiser
        plot_forecast(context, true_future, pred_mean, pred_quantiles,
                     datetimes_context, datetimes_future,
                     f"Prédiction hebdomadaire - {test_start.date()}",
                     f"weekly_{zone}",
                     zone_name=zone)
    
    df_results = pd.DataFrame(results)
    print(f"\n📊 Résumé Scénario 2:")
    print(f"   MAE moyen: {df_results['mae'].mean():.4f} JPY/kWh")
    print(f"   MAPE moyen: {df_results['mape'].mean():.2f}%")
    
    return df_results


def scenario_3_monthly(df, forecaster):
    """
    Scénario 3: Prédiction mensuelle
    Prédire septembre 2025 en connaissant septembre 2024
    """
    print("\n" + "=" * 70)
    print("SCÉNARIO 3: PRÉDICTION MENSUELLE (Sept 2025 from Sept 2024)")
    print("=" * 70)
    
    zone = 'Tokyo'  # Zone principale
    
    # Septembre 2024 comme contexte
    sept_2024_start = pd.to_datetime('2024-09-01')
    sept_2024_end = pd.to_datetime('2024-10-01')
    
    # Septembre 2025 à prédire
    sept_2025_start = pd.to_datetime('2025-09-01')
    sept_2025_end = pd.to_datetime('2025-10-01')
    
    # Contexte: août + septembre 2024
    context_start = sept_2024_end - timedelta(days=20)
    future_start = sept_2025_end - timedelta(days=20)

    print(context_start)
    
    series_context, datetimes_context = get_zone_series(df, zone, context_start, sept_2024_end)
    series_future, datetimes_future = get_zone_series(df, zone, future_start, sept_2025_end)
    
    n_horizon = len(series_future)  # ~30 jours * 48
    
    print(f"   Contexte: {len(series_context)} points")
    print(f"   Horizon: {n_horizon} points ({n_horizon/48:.1f} jours)")
    
    # Prédire
    pred_mean, pred_quantiles = forecaster.forecast_single(series_context, horizon=n_horizon)
    
    # Métriques
    mae, rmse, mape = compute_metrics(series_future, pred_mean)
    print(f"\n   MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    
    # Visualiser
    plot_forecast(series_context, series_future, pred_mean, pred_quantiles,
                 datetimes_context, datetimes_future,
                 f"Prédiction mensuelle - Sept 2025 (contexte Sept 2024)",
                 f"monthly_sept_2025_{zone}",
                 zone_name=zone)
    
    return {'zone': zone, 'mae': mae, 'rmse': rmse, 'mape': mape}


def scenario_4_multisite(df, forecaster):
    """
    Scénario 4: Multi-site
    Prédire 1 semaine pour Tokyo en connaissant:
    - 4 semaines de Tokyo
    - 5 semaines complètes des autres zones
    """
    print("\n" + "=" * 70)
    print("SCÉNARIO 4: PRÉDICTION MULTI-SITE")
    print("=" * 70)
    
    target_zone = 'Tokyo'
    other_zones = [z for z in ZONES if z != target_zone][:4]  # 4 autres zones
    
    # Période de test
    test_start = pd.to_datetime('2025-02-01')
    test_end = test_start + timedelta(days=7)
    
    # Contexte Tokyo: 4 semaines
    tokyo_context_start = test_start - timedelta(days=28)
    
    # Contexte autres zones: 5 semaines
    others_context_start = test_start - timedelta(days=35)
    
    print(f"   Cible: {target_zone}")
    print(f"   Autres zones: {other_zones}")
    
    # Construire le contexte multi-site
    # Format: [Tokyo_4weeks, Zone1_5weeks, Zone2_5weeks, ...]
    
    tokyo_series, _ = get_zone_series(df, target_zone, tokyo_context_start, test_start)
    
    # Concaténer toutes les séries
    context_combined = tokyo_series.copy()
    
    for zone in other_zones:
        zone_series, _ = get_zone_series(df, zone, others_context_start, test_start)
        context_combined = np.concatenate([context_combined, zone_series])
    
    # Vérité terrain
    true_future, datetimes_future = get_zone_series(df, target_zone, test_start, test_end)
    n_horizon = len(true_future)
    
    print(f"   Taille contexte combiné: {len(context_combined)} points")
    print(f"   Horizon: {n_horizon} points")
    
    # Prédire
    pred_mean, pred_quantiles = forecaster.forecast_single(context_combined, horizon=n_horizon)
    
    # Métriques
    mae, rmse, mape = compute_metrics(true_future, pred_mean)
    print(f"\n   MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    
    # Visualiser (utiliser Tokyo context pour le graphique)
    _, datetimes_context = get_zone_series(df, target_zone, tokyo_context_start, test_start)
    
    plot_forecast(tokyo_series, true_future, pred_mean, pred_quantiles,
                 datetimes_context, datetimes_future,
                 f"Prédiction multi-site - {target_zone}",
                 f"multisite_{target_zone}",
                 zone_name=f"{target_zone} (avec contexte {', '.join(other_zones)})")
    
    return {'zone': target_zone, 'mae': mae, 'rmse': rmse, 'mape': mape}


def scenario_5_peak(df, forecaster):
    """
    Scénario 5: Prédiction de pic
    Prédire les points autour du 23/09/2024 à Chubu
    """
    print("\n" + "=" * 70)
    print("SCÉNARIO 5: PRÉDICTION DE PIC (23/09/2024 - Chubu)")
    print("=" * 70)
    
    zone = 'Chubu'
    peak_date = pd.to_datetime('2024-09-23')
    
    # Contexte: 2 semaines avant
    context_start = peak_date - timedelta(days=14)
    
    # Prédire: 3 jours autour du pic (1 jour avant, jour du pic, 1 jour après)
    pred_start = peak_date - timedelta(days=1)
    pred_end = peak_date + timedelta(days=2)
    
    # Récupérer données
    series_context, datetimes_context = get_zone_series(df, zone, context_start, pred_start)
    series_future, datetimes_future = get_zone_series(df, zone, pred_start, pred_end)
    
    n_horizon = len(series_future)
    
    print(f"   Zone: {zone}")
    print(f"   Contexte: {len(series_context)} points (2 semaines)")
    print(f"   Horizon: {n_horizon} points (3 jours)")
    
    # Analyser le pic
    peak_value = series_future.max()
    peak_idx = series_future.argmax()
    peak_time = datetimes_future[peak_idx]
    
    print(f"   Pic réel: {peak_value:.2f} JPY/kWh à {peak_time}")
    
    # Prédire
    pred_mean, pred_quantiles = forecaster.forecast_single(series_context, horizon=n_horizon)
    
    # Métriques
    mae, rmse, mape = compute_metrics(series_future, pred_mean)
    
    # Erreur spécifique au pic
    peak_pred = pred_mean[peak_idx]
    peak_error = abs(peak_value - peak_pred)
    peak_error_pct = (peak_error / peak_value) * 100
    
    print(f"\n   Métriques globales:")
    print(f"     MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    print(f"   Métriques du pic:")
    print(f"     Valeur prédite: {peak_pred:.2f} JPY/kWh")
    print(f"     Erreur pic: {peak_error:.2f} JPY/kWh ({peak_error_pct:.1f}%)")
    
    # Visualiser
    plot_forecast(series_context, series_future, pred_mean, pred_quantiles,
                 datetimes_context, datetimes_future,
                 f"Prédiction de pic - {peak_date.date()}",
                 f"peak_{zone}_{peak_date.date()}",
                 zone_name=f"{zone} (pic: {peak_value:.2f} JPY/kWh)")
    
    return {
        'zone': zone,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'peak_value': peak_value,
        'peak_pred': peak_pred,
        'peak_error': peak_error,
        'peak_error_pct': peak_error_pct
    }


# ============================================================
# 5. SCRIPT PRINCIPAL
# ============================================================

def main():
    print("=" * 70)
    print("PRÉDICTION DE PRIX NODAUX AVEC TIMESFM")
    print("Optimal Power Flow - Japon 2024-2025")
    print("=" * 70)
    
    # Charger les données
    df = load_data('spot_summary_2024.csv', 'spot_summary_2025.csv')
    
    # Initialiser TimesFM
    forecaster = TimesFMForecaster(max_context=8192, max_horizon=1024)
    
    # Exécuter les scénarios
    print("\n" + "=" * 70)
    print("EXÉCUTION DES 5 SCÉNARIOS")
    print("=" * 70)
    
    results_all = {}
    
    # Scénario 1: Journalier
    results_all['scenario_1'] = scenario_1_daily(df, forecaster)
    
    # Scénario 2: Hebdomadaire
    results_all['scenario_2'] = scenario_2_weekly(df, forecaster)
    
    # Scénario 3: Mensuel
    results_all['scenario_3'] = scenario_3_monthly(df, forecaster)
    
    # Scénario 4: Multi-site
    results_all['scenario_4'] = scenario_4_multisite(df, forecaster)
    
    # Scénario 5: Pic
    results_all['scenario_5'] = scenario_5_peak(df, forecaster)
    
    # Rapport final
    print("\n" + "=" * 70)
    print("RAPPORT FINAL")
    print("=" * 70)
    
    print("\n📊 Résumé des performances TimesFM:")
    print("\nScénario 1 (Journalier - 2 pas):")
    if isinstance(results_all['scenario_1'], pd.DataFrame):
        df_s1 = results_all['scenario_1']
        print(f"  MAE moyen: {df_s1['mae'].mean():.4f} JPY/kWh")
        print(f"  MAPE moyen: {df_s1['mape'].mean():.2f}%")
    
    print("\nScénario 2 (Hebdomadaire - 7 jours):")
    if isinstance(results_all['scenario_2'], pd.DataFrame):
        df_s2 = results_all['scenario_2']
        print(f"  MAE moyen: {df_s2['mae'].mean():.4f} JPY/kWh")
        print(f"  MAPE moyen: {df_s2['mape'].mean():.2f}%")
    
    print("\nScénario 3 (Mensuel - Sept 2025):")
    s3 = results_all['scenario_3']
    print(f"  MAE: {s3['mae']:.4f} JPY/kWh")
    print(f"  MAPE: {s3['mape']:.2f}%")
    
    print("\nScénario 4 (Multi-site):")
    s4 = results_all['scenario_4']
    print(f"  MAE: {s4['mae']:.4f} JPY/kWh")
    print(f"  MAPE: {s4['mape']:.2f}%")
    
    print("\nScénario 5 (Pic 23/09/2024 - Chubu):")
    s5 = results_all['scenario_5']
    print(f"  MAE global: {s5['mae']:.4f} JPY/kWh")
    print(f"  MAPE global: {s5['mape']:.2f}%")
    print(f"  Erreur sur le pic: {s5['peak_error']:.2f} JPY/kWh ({s5['peak_error_pct']:.1f}%)")
    
    print("\n" + "=" * 70)
    print(f"✓ Tous les résultats sauvegardés dans ./{RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()