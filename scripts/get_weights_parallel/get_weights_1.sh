
for data in buildings_900k buildings_bench cloudops_tsf cmip6; do
  python -m cli.get_weights \
    -cp conf/pretrain \
    run_name=first_run \
    model=moirai_small \
    data=$data
done