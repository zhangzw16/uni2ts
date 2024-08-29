python -m cli.eval \
  -cp conf/eval \
  model=moirai_lightning_ckpt \
  run_name=eval_etth1 \
  model.patch_size=32 \
  model.context_length=96 \
  data=lsf_test \
  data.dataset_name=ETTh1 \
  data.prediction_length=96 