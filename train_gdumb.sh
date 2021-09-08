
TIME=`(date +%Y-%m-%d-%H-%M-%S)`
NUM_EPS=15
LR=5e-4

OUTBASE=/mnt/nfs/scratch1/jpayan/Lamolrelease

mkdir -p $OUTBASE/logs/${TIME}/train/real
mkdir -p $OUTBASE/logs/${TIME}/train/finetune

for K in 300 500 1000 1500 1800; do
  for SEED in {0..9}; do
    sbatch -J gdumb_${K}_${SEED} \
            -e $OUTBASE/logs/${TIME}/train/finetune/lamol_train_gdumb_${K}_${SEED}.err \
            -o $OUTBASE/logs/${TIME}/train/finetune/lamol_train_gdumb_${K}_${SEED}.log \
            --mem=15G \
            --partition=m40-long \
            --time=02-00:00:00 \
            --gres=gpu:1 \
            ./setupandrunexp.sh 0.25 0.2 $LR $NUM_EPS "gdumb_${K}_${SEED}_1 gdumb_${K}_${SEED}_2 gdumb_${K}_${SEED}_3 gdumb_${K}_${SEED}_4 gdumb_${K}_${SEED}_5" so_data/so_labels \
          finetune $OUTBASE/models/${TIME} ~/Lamolrelease