DATA=${1}
VAL_DATA=${2}
MODEL=${3:-moirai_1.0_R_small}

python -m cli.train \
  -cp conf/finetune \
  run_name=${MODEL}_ft_${DATA}_val_${VAL_DATA} \
  model=${MODEL} \
  data=${DATA} \
  val_data=${VAL_DATA}