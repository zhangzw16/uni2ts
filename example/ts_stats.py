import datasets
from uni2ts.common.env import env
import pyarrow.compute as pc
import pandas as pd

from uni2ts.data.builder.lotsa_v1 import (
	Buildings900KDatasetBuilder,
	BuildingsBenchDatasetBuilder,
	CloudOpsTSFDatasetBuilder,
	CMIP6DatasetBuilder,
	ERA5DatasetBuilder,
	GluonTSDatasetBuilder,
	LargeSTDatasetBuilder,
	LibCityDatasetBuilder,
	OthersLOTSADatasetBuilder,
	ProEnFoDatasetBuilder,
	SubseasonalDatasetBuilder,
)


def compute_stats(dataset, field):
    rows = len(dataset)

    if dataset[0][field].ndim > 1:
        var_dim = dataset[0][field].shape[0]
        var_point_list = pc.list_value_length(
            pc.list_flatten(pc.list_slice(dataset.data.column(field), 0, var_dim))
        ).to_numpy()
        
        total_points = var_point_list.sum()
        var_len_avg = total_points / rows / var_dim
        var_len_min = var_point_list.min()
        var_len_max = var_point_list.max()
    else:
        var_dim = 1
        var_point_list = pc.list_value_length(dataset.data.column(field)).to_numpy()
        total_points = var_point_list.sum()
        var_len_avg = total_points / rows
        var_len_min = var_point_list.min()
        var_len_max = var_point_list.max()
    
    return var_dim, total_points, var_len_avg, var_len_min, var_len_max
    

def get_dataset_stats(dataset_name):
    dataset = datasets.load_from_disk(str(env.LOTSA_V1_PATH / dataset_name)).with_format("numpy")
    
    rows = len(dataset)
    freq = dataset[0]['freq']

    fields = ["target"]
    optional_fields = ["past_feat_dynamic_real"]

    stats = {}
    for field in fields + optional_fields:
        if field not in dataset.column_names and field in optional_fields:
            continue
        stats[field] = compute_stats(dataset, field)
    
    return dataset_name, rows, freq, stats

def add_domain_col(df, domain_dict: dict):
    df["domain"] = None
    for domain, builder in domain_dict.items():
        datasets = []
        if isinstance(builder, list):
            for b in builder:
                datasets += b.dataset_list
        else:
            datasets = builder.dataset_list
        
        df.loc[datasets, "domain"] = domain

    return df

def add_points2day_col(df):
    # all possible freqs: 4S, T, 5T, 10T, 15T, 30T, H, 6H, D, W, M, Q, A
    def points2day(row):
        if row["freq"] == "D":
            return 1
        elif row["freq"] == "H":
            return 24
        elif row["freq"] == "T":
            return 1440
    return df

if __name__ == "__main__":

    domain_dict = {
        "buildings_bench": [BuildingsBenchDatasetBuilder, Buildings900KDatasetBuilder],
        "cloud_ops_tsf": CloudOpsTSFDatasetBuilder,
        "climate": [CMIP6DatasetBuilder, ERA5DatasetBuilder, SubseasonalDatasetBuilder],
        "gluon_ts": GluonTSDatasetBuilder,
        "large_st": LargeSTDatasetBuilder,
        "lib_city": LibCityDatasetBuilder,
        "pro_en_fo": ProEnFoDatasetBuilder,
        "others_lotsa": OthersLOTSADatasetBuilder,
    }

    # df = pd.read_csv("dataset_stats.csv", index_col="dataset")
    # df = add_domain_col(df, domain_dict)
    # df.to_csv("dataset_stats.csv")

    dataset_list = (
        Buildings900KDatasetBuilder.dataset_list +
        BuildingsBenchDatasetBuilder.dataset_list +
        CloudOpsTSFDatasetBuilder.dataset_list +
        CMIP6DatasetBuilder.dataset_list +
        ERA5DatasetBuilder.dataset_list +
        GluonTSDatasetBuilder.dataset_list +
        LargeSTDatasetBuilder.dataset_list +
        LibCityDatasetBuilder.dataset_list +
        OthersLOTSADatasetBuilder.dataset_list +
        ProEnFoDatasetBuilder.dataset_list +
        SubseasonalDatasetBuilder.dataset_list
    )

    print(f"Total datasets: {len(dataset_list)}")
    print(f"Datasets: {dataset_list}")

    column_names = ["dataset", "rows", "freq", "target_dim", "target_points", "target_len_avg", "target_len_min", "target_len_max", "past_feat_dynamic_real_dim", "past_feat_dynamic_real_points", "past_feat_dynamic_real_len_avg", "past_feat_dynamic_real_len_min", "past_feat_dynamic_real_len_max"]
    df = pd.DataFrame(columns=column_names)

    # dataset_list = ['hog', 'sceaux', 'bdg-2_panther', 'nn5_weekly']
    for dataset in dataset_list:
        dataset_name, rows, freq, stats = get_dataset_stats(dataset)
        target_dim, target_points, target_len_avg, target_len_min, target_len_max = stats["target"]
        past_feat_dynamic_real_dim, past_feat_dynamic_real_points, past_feat_dynamic_real_len_avg, past_feat_dynamic_real_len_min, past_feat_dynamic_real_len_max = stats.get("past_feat_dynamic_real", (0, 0, 0, 0, 0))

        new_row = pd.DataFrame([{
            "dataset": dataset_name,
            "rows": rows,
            "freq": freq,
            "target_dim": target_dim,
            "target_points": target_points,
            "target_len_avg": target_len_avg,
            "target_len_min": target_len_min,
            "target_len_max": target_len_max,
            "past_feat_dynamic_real_dim": past_feat_dynamic_real_dim,
            "past_feat_dynamic_real_points": past_feat_dynamic_real_points,
            "past_feat_dynamic_real_len_avg": past_feat_dynamic_real_len_avg,
            "past_feat_dynamic_real_len_min": past_feat_dynamic_real_len_min,
            "past_feat_dynamic_real_len_max": past_feat_dynamic_real_len_max,
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    df.set_index("dataset", inplace=True)
    df = add_domain_col(df, domain_dict)
    df.to_csv("dataset_stats.csv")    




