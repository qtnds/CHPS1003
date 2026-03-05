# Configuration optimale pour voir l'allée de Karman
#python tourbillon.py --Re 100 --total 3500 --horizon 200 --gif-fps 30

# Pour Re plus élevé (tourbillons plus serrés)
#python tourbillon.py --Re 200 --total 1200 --horizon 200 --skip-arima

# Test rapide (800 pas minimum)
#python tourbillon.py --total 800 --horizon 150 --skip-arima

python tourbillon_chronos.py \
  --nx-win 400 --ny-win 100 \
  --pad-x 100 --pad-y 50 \
  --total 500 --horizon 64 \
  --Re 150 \
  --obstacle-radius 15 \
  --chronos-model amazon/chronos-t5-small \
  --chronos-device cuda \
  --lstm-epochs 20 \
  --gif-fps 20


python tourbillon_chronos.py \
  --nx-win 200 --ny-win 60 \
  --pad-x 60 --pad-y 30 \
  --total 2500 --horizon 64 \
  --Re 80 \
  --obstacle-radius 10 \
  --lstm-epochs 20 \
  --chronos-model amazon/chronos-t5-small \
  --chronos-batch 8 \
  --chronos-samples 20 \
  --gif-fps 20