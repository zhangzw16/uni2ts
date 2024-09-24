
for data in others proenfo subseasonal; do
  python -m cli.get_weights \
    -cp conf/pretrain \
    run_name=first_run \
    model=moirai_small \
    data=$data
done