
MODEL=moirai_1.0_R_small

for dataset in etth1 etth2 ettm1 ettm2 traffic electricity exchange weather
do 
    python -m cli.train \
    -cp conf/finetune \
    run_name=${MODEL}_ft_${DATA}_val_${VAL_DATA} \
    model=${MODEL} \
    data=${dataset} \
    val_data=${dataset}
done