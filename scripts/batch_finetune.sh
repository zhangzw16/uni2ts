
MODEL=moirai_1.0_R_small

for dataset in lotsa_v1_weighted_group1 lotsa_v1_weighted_group2 lotsa_v1_weighted_group3 lotsa_v1_weighted_group4 lotsa_v1_weighted_group5
do 
    python -m cli.train \
    -cp conf/finetune \
    --config-name group.yaml \
    run_name=${MODEL}_ft_${dataset} \
    model=${MODEL} \
    data=${dataset}
done
