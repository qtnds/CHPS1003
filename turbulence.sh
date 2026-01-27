#python _turbulence.py --nx 24 --ny 24 --nz 24 --total 300 --horizon 50
#python _turbulence.py --nx 32 --ny 32 --nz 32 --total 400 --horizon 60 --Ra 2000
#python _turbulence.py --total 250 --horizon 40 --skip-arima --lstm-epochs 20
#python _turbulence.py --Ra 5000 --total 500 --horizon 80


# Turbulence forte (recommandé)
#python turbulence.py --Ra 50000 --total 400 --skip-arima

# Turbulence TRÈS forte (attention : lent !)
python turbulence.py --Ra 100000 --total 500 --nx 32 --ny 32 --nz 32 --skip-arima

# Test rapide
#python turbulence.py --Ra 20000 --total 200 --nx 20 --ny 20 --nz 20 --skip-arima