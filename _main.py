from utilities1 import (
    simulate_heat_diffusion,
    volume_to_timeseries,
    timeseries_to_volume,
    TimesFMForecaster,
    show_slice,
    compare_true_vs_pred,
    plot_temperature_at_point
)

# ============================================================
# 1. GENERATION DU DATASET
# ============================================================

print("Simulating heat diffusion...")
volume = simulate_heat_diffusion(nx=32, ny=32, nt=200)
print(f"Volume shape: {volume.shape}")

# ============================================================
# 2. PRETRAITEMENT
# ============================================================

print("Converting volume to time-series...")
timeseries = volume_to_timeseries(volume)
print(f"Number of time-series: {timeseries.shape[0]}")

# ============================================================
# 3. FORECASTING
# ============================================================

print("Running TimesFM forecasting...")
forecaster = TimesFMForecaster()

point_forecast, quantile_forecast = forecaster.forecast(
    timeseries,
    horizon=20,
    batch_size=64
)

# ============================================================
# 4. RECONSTRUCTION
# ============================================================

predicted_volume = timeseries_to_volume(
    point_forecast,
    spatial_shape=volume.shape[1:]
)

# ============================================================
# 5. VISUALISATION (SAVED)
# ============================================================

t_vis = min(10, volume.shape[0] - 1)

show_slice(volume, t_vis, title="Ground Truth", filename="ground_truth")
show_slice(predicted_volume, t_vis, title="TimesFM Prediction", filename="prediction")
compare_true_vs_pred(volume, predicted_volume, t_vis, filename="comparison")

print("Results saved in ./results/")
print("Main script finished successfully!")


# ============================================================
# 6. TEMPÉRATURE À UN POINT D’INTÉRÊT
# ============================================================

# Choix d'un point stratégique : centre de la plaque
point_of_interest = (volume.shape[1]//2, volume.shape[2]//2)

print(f"Plotting temperature at point {point_of_interest}...")
plot_temperature_at_point(
    true_vol=volume,
    pred_vol=predicted_volume,
    point=point_of_interest,
    title="Center of Plate"
)
