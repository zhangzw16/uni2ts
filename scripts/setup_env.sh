mkdir -p dataset

ln -s /data/Blob_EastUS/v-zhenwzhang/tsfm_datasets/lotsa_data ./dataset/
ln -s /data/Blob_WestJP/v-jiawezhang/data/all_datasets ./dataset

python -m uni2ts.data.builder.simple ETTh1 dataset/all_datasets/ETT-small/ETTh1.csv --date_offset '2017-06-26 00:00:00'  # 12 * 30 * 24
python -m uni2ts.data.builder.simple ETTh2 dataset/all_datasets/ETT-small/ETTh2.csv --date_offset '2017-06-26 00:00:00'  # 12 * 30 * 24
python -m uni2ts.data.builder.simple ETTm1 dataset/all_datasets/ETT-small/ETTm1.csv --date_offset '2017-06-26 00:00:00'  # 12 * 30 * 24 * 4
python -m uni2ts.data.builder.simple ETTm2 dataset/all_datasets/ETT-small/ETTm2.csv --date_offset '2017-06-26 00:00:00'  # 12 * 30 * 24 * 4

# train / val / test split: 0.7 / 0.1 / 0.2 for the following datasets
python -m uni2ts.data.builder.simple Traffic dataset/all_datasets/traffic/traffic.csv --date_offset '2017-11-24 18:00:00'
python -m uni2ts.data.builder.simple Electricity dataset/all_datasets/electricity/electricity.csv --date_offset '2018-08-07 06:00:00'
python -m uni2ts.data.builder.simple Weather dataset/all_datasets/weather/weather.csv --date_offset '2020-09-13 05:20:00'
python -m uni2ts.data.builder.simple Exchange dataset/all_datasets/exchange_rate/exchange_rate.csv --offset 5311