# Tester que la clé fonctionne
python -c "import cdsapi; cdsapi.Client()"

# Puis lancer avec ERA5 réel
python forecast_meteo_timesfm.py --era5

# Résolution plus fine
#python forecast_meteo_timesfm.py --nx 32 --ny 32 --nz 16 --hist 48 --pred 12

# Sans l'approche latente (plus rapide)
#python forecast_meteo_timesfm.py --skip-latent

