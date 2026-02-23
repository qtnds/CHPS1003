# Test rapide (basse résolution)
python 3D_tourbillon.py --nx-win 120 --ny-win 60 --nz-win 40 \
    --warmup 500 --total 400 --Re 150

# Recommandé (bonne qualité)
#python karman_3d_final.py --nx-win 200 --ny-win 80 --nz-win 60 \
#    --warmup 800 --total 600 --Re 180 --U-inlet 0.05

# Haute résolution
#python karman_3d_final.py --nx-win 300 --ny-win 120 --nz-win 80 \
#    --warmup 1200 --total 800 --Re 200 --cyl-radius 12