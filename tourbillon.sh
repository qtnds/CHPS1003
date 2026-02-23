# Configuration optimale pour voir l'allée de Karman
python tourbillon.py --Re 100 --total 3500 --horizon 200 --gif-fps 30

# Pour Re plus élevé (tourbillons plus serrés)
#python tourbillon.py --Re 200 --total 1200 --horizon 200 --skip-arima

# Test rapide (800 pas minimum)
#python tourbillon.py --total 800 --horizon 150 --skip-arima

