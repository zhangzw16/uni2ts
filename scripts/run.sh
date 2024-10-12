# DEVICE=0
# export CUDA_VISIBLE_DEVICES=${DEVICE}

# take input arguments as DATASET
DATASET=${1}
NAME=${2}

python -m cli.train \
  -cp conf/pretrain \
  run_name=${NAME} \
  model=moirai_small \
  data=${DATASET}