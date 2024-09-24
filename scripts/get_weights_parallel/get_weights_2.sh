
for data in era5 gluonts largest lib_city; do
  python -m cli.get_weights \
    -cp conf/pretrain \
    run_name=first_run \
    model=moirai_small \
    data=$data
done