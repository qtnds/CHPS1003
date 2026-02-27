# Exemple basique
#python main3d.py --total 200 --horizon 50

# Grille plus fine
#python main3d.py --nx 32 --ny 32 --nz 32 --total 150 --horizon 40

# Matériau différent
#python main3d.py --material copper --total 180


# Perturbation chaude au centre à t=50
#python main3d_disrup.py --nx 32 --ny 32 --nz 32 --total 150 --horizon 40 --disruption=[16,16,16,200,50]

# Perturbation froide sur le côté gauche
#python main3d_disrup.py --disruption=[5,12,12,10,30] --total 200 --horizon 60

# Perturbation très chaude proche de la source
#python main3d_disrup.py --disruption=[28,16,16,300,40] --material copper


## Convection
# Refroidissement par convection en haut
#python main.py --convection=[10,15,face,80,top] --total 200 --horizon 40

# Chauffage localisé + refroidissement
#python main.py \
#  --disruption=[8,12,12,250,40] \
# --convection=[5,12,rectangle,100,top,8,20,8,20] \
#  --total 180 --horizon 50 --material copper

python main_chronos.py \
  --total 200 --horizon 50 \
  --nx 20 --ny 20 --nz 20 \
  --material steel

# Convection circulaire intense
#python main.py --convection=[0,20,circle,50,bottom,12,12,7] --nx 24 --ny 24 --nz 24 --material steel