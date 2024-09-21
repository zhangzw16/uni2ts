# DEVICE=0
# export CUDA_VISIBLE_DEVICES=${DEVICE}

# take input arguments as DATASET
DATASET=${1}

python -m cli.train \
  -cp conf/pretrain \
  run_name=${DATASET} \
  model=moirai_small \
  data=${DATASET}