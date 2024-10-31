DATA=${1}
# VAL_DATA=${2}
# MODEL=${3:-moirai_1.0_R_small}
MODEL=moirai_1.0_R_small

python -m cli.train \
  -cp conf/finetune \
  --config-name cpt.yaml \
  run_name=${MODEL}_ft_${DATA}_lr_1em4_seed_321 \
  model=${MODEL} \
  data=${DATA} \
  # val_data=${VAL_DATA}