# DEVICE=0
# export CUDA_VISIBLE_DEVICES=${DEVICE}

# take input arguments as DATASET
DATASET=${1}
# NAME=${2}

python -m cli.train \
  -cp conf/pretrain \
  --config-name default_val.yaml \
  run_name=moirai_small_${DATASET} \
  model=moirai_small \
  data=${DATASET} \
  val_data=lotsa_v1_all