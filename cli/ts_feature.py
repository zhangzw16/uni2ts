import argparse
from collections import defaultdict
from pathlib import Path

import datasets
import pandas as pd
from tsfeatures import entropy, hurst, lumpiness, stability, stl_features, tsfeatures

from uni2ts.common.env import env
from tqdm import tqdm
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

class TSFeature:
	def __init__(self, features=None, threads=8):
		if features is None:
			self.features = [entropy, stability, lumpiness, hurst, stl_features]
		else:
			self.features = features
		self.threads = threads
		self.freq_dict = defaultdict(lambda: 1, {
			'4S': 1,
			'S': 1,
			'T': 1440,
			'2T': 720,
			'5T': 288,
			'10T': 144,
			'15T': 96,
			'30T': 48,
			'H': 24,
			'6H': 4,
			'D': 1,
			'W': 1,
			'Y': 1,
			'Q': 4,
			'M': 12,
		})


	@staticmethod
	def hf_dataset_to_panel(dataset, start_idx, end_idx):
		# Convert a slice of HuggingFace dataset to pandas DataFrame
		panel_data = []

		for i in range(start_idx, end_idx):
			data = dataset[i]
			item_id = data['item_id']
			start_time = data['start']
			target = data['target']
			freq = data['freq']

			# Create a date range starting from start_time with the given frequency
			date_range = pd.date_range(start=str(start_time), periods=target.shape[-1], freq=freq)

			if target.ndim == 1:
				# Create a DataFrame for the current item if target is 1D
				df_tmp = pd.DataFrame({
					'unique_id': item_id,
					'ds': date_range,
					'y': target
				})
				panel_data.append(df_tmp)
			elif target.ndim == 2:
				# Create DataFrames for each variable if target is 2D
				for i in range(target.shape[0]):
					df_tmp = pd.DataFrame({
						'unique_id': f"{item_id}_{i}",
						'ds': date_range,
						'y': target[i]
					})
					panel_data.append(df_tmp)

		# Concatenate all DataFrames into a single DataFrame
		panel = pd.concat(panel_data, ignore_index=True)
		return panel, freq

	def calculate_features(self, dataset, is_save=False, dataset_name=None, K=64):
		tsf_path = env.LOTSA_V1_PATH.parent / "lotsa_features"
		# if csv file exist, return it
		if is_save and (tsf_path / f"{dataset_name}_tsf.csv").exists():
			print(f"Loading features from {tsf_path / f'{dataset_name}_tsf.csv'}")
			return pd.read_csv(tsf_path / f"{dataset_name}_tsf.csv")
		
		total_length = len(dataset)
		row_length = dataset[0]['target'].size
		print(f"Total number of rows: {total_length}, each row len: {row_length}")
		all_results = []

		with tqdm(total=total_length, desc="Calculating features") as pbar:
			for start_idx in range(0, total_length, K):
				end_idx = min(start_idx + K, total_length)

				# Convert a slice of HuggingFace dataset to pandas DataFrame
				panel, freq_str = self.hf_dataset_to_panel(dataset, start_idx, end_idx)
				freq = self.freq_dict[freq_str]
				
				# Calculate features for the current slice
				result = tsfeatures(panel, freq=freq, threads=self.threads, features=self.features)
				all_results.append(result)

				# Update progress bar
				pbar.update(end_idx - start_idx)

		# Concatenate all results into a single DataFrame
		final_result = pd.concat(all_results, ignore_index=True)

		# Save features to disk
		if is_save:
			# folder not exist, create it
			if not tsf_path.exists():
				tsf_path.mkdir(parents=True)

			final_result.to_csv(tsf_path / f"{dataset_name}_tsf.csv", index=True)
		
		return final_result

# Example usage
# Assuming `dataset` is a HuggingFace dataset with the specified format
# ts_feature = TSFeature(features=[acf_features], freq=7)
# features_result = ts_feature.calculate_features(dataset)


# some code to test the class
if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description="Calculate time series features for given datasets.")
	parser.add_argument("dataset_names", type=str, nargs='+', help="Names of the datasets to process")
	args = parser.parse_args()
	print(f"Processing datasets: {args.dataset_names}")

	# # Example dataset
	# # dataset = datasets.load_from_disk(str(env.LOTSA_V1_PATH / "bdg-2_fox")).with_format("numpy")
	# dataset = datasets.load_from_disk(str(env.LOTSA_V1_PATH / "PEMS04")).with_format("numpy")
	# # panel = TSFeature.hf_dataset_to_panel(dataset)
	# # print(panel.head())
	# # print(len(panel))

	# ts_feat = TSFeature()
	# features_result = ts_feat.calculate_features(dataset)
	# print(features_result)

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

	if args.dataset_names[0] == "all":
		args.dataset_names = dataset_list

	for dataset in args.dataset_names:
		assert dataset in dataset_list, f"Dataset {dataset} not found in the list of available datasets."

		print(f"Processing dataset: {dataset}")
		ts_data = datasets.load_from_disk(str(env.LOTSA_V1_PATH / dataset)).with_format("numpy")
		ts_feat = TSFeature()
		# try:
		ts_feat.calculate_features(ts_data, is_save=True, dataset_name=dataset)
		# except Exception as e:
			# print(f"Error processing dataset {dataset}: {e}")
	
	# dataset can be any of the following datasets
	# ['buildings_900k',
	# 'sceaux',
	# 'borealis',
	# 'ideal',
	# 'bdg-2_panther',
	# 'bdg-2_fox',
	# 'bdg-2_rat',
	# 'bdg-2_bear',
	# 'smart',
	# 'lcl',
	# 'azure_vm_traces_2017',
	# 'borg_cluster_data_2011',
	# 'alibaba_cluster_trace_2018',
	# 'cmip6_1850',
	# 'cmip6_1855',
	# 'cmip6_1860',
	# 'cmip6_1865',
	# 'cmip6_1870',
	# 'cmip6_1875',
	# 'cmip6_1880',
	# 'cmip6_1885',
	# 'cmip6_1890',
	# 'cmip6_1895',
	# 'cmip6_1900',
	# 'cmip6_1905',
	# 'cmip6_1910',
	# 'cmip6_1915',
	# 'cmip6_1920',
	# 'cmip6_1925',
	# 'cmip6_1930',
	# 'cmip6_1935',
	# 'cmip6_1940',
	# 'cmip6_1945',
	# 'cmip6_1950',
	# 'cmip6_1955',
	# 'cmip6_1960',
	# 'cmip6_1965',
	# 'cmip6_1970',
	# 'cmip6_1975',
	# 'cmip6_1980',
	# 'cmip6_1985',
	# 'cmip6_1990',
	# 'cmip6_1995',
	# 'cmip6_2000',
	# 'cmip6_2005',
	# 'cmip6_2010',
	# 'era5_1989',
	# 'era5_1990',
	# 'era5_1991',
	# 'era5_1992',
	# 'era5_1993',
	# 'era5_1994',
	# 'era5_1995',
	# 'era5_1996',
	# 'era5_1997',
	# 'era5_1998',
	# 'era5_1999',
	# 'era5_2000',
	# 'era5_2001',
	# 'era5_2002',
	# 'era5_2003',
	# 'era5_2004',
	# 'era5_2005',
	# 'era5_2006',
	# 'era5_2007',
	# 'era5_2008',
	# 'era5_2009',
	# 'era5_2010',
	# 'era5_2011',
	# 'era5_2012',
	# 'era5_2013',
	# 'era5_2014',
	# 'era5_2015',
	# 'era5_2016',
	# 'era5_2017',
	# 'era5_2018',
	# 'taxi_30min',
	# 'uber_tlc_daily',
	# 'uber_tlc_hourly',
	# 'wiki-rolling_nips',
	# 'london_smart_meters_with_missing',
	# 'wind_farms_with_missing',
	# 'wind_power',
	# 'solar_power',
	# 'oikolab_weather',
	# 'elecdemand',
	# 'covid_mobility',
	# 'kaggle_web_traffic_weekly',
	# 'extended_web_traffic_with_missing',
	# 'm5',
	# 'm4_yearly',
	# 'm1_yearly',
	# 'm1_quarterly',
	# 'monash_m3_yearly',
	# 'monash_m3_quarterly',
	# 'tourism_yearly',
	# 'm4_hourly',
	# 'm4_daily',
	# 'm4_weekly',
	# 'm4_monthly',
	# 'm4_quarterly',
	# 'm1_monthly',
	# 'monash_m3_monthly',
	# 'monash_m3_other',
	# 'nn5_daily_with_missing',
	# 'nn5_weekly',
	# 'tourism_monthly',
	# 'tourism_quarterly',
	# 'cif_2016_6',
	# 'cif_2016_12',
	# 'traffic_hourly',
	# 'traffic_weekly',
	# 'australian_electricity_demand',
	# 'rideshare_with_missing',
	# 'saugeenday',
	# 'sunspot_with_missing',
	# 'temperature_rain_with_missing',
	# 'vehicle_trips_with_missing',
	# 'weather',
	# 'car_parts_with_missing',
	# 'fred_md',
	# 'pedestrian_counts',
	# 'hospital',
	# 'covid_deaths',
	# 'kdd_cup_2018_with_missing',
	# 'bitcoin_with_missing',
	# 'us_births',
	# 'largest_2017',
	# 'largest_2018',
	# 'largest_2019',
	# 'largest_2020',
	# 'largest_2021',
	# 'BEIJING_SUBWAY_30MIN',
	# 'HZMETRO',
	# 'LOOP_SEATTLE',
	# 'LOS_LOOP',
	# 'M_DENSE',
	# 'PEMS03',
	# 'PEMS04',
	# 'PEMS07',
	# 'PEMS08',
	# 'PEMS_BAY',
	# 'Q-TRAFFIC',
	# 'SHMETRO',
	# 'SZ_TAXI',
	# 'kdd2022',
	# 'godaddy',
	# 'favorita_sales',
	# 'favorita_transactions',
	# 'restaurant',
	# 'hierarchical_sales',
	# 'china_air_quality',
	# 'beijing_air_quality',
	# 'residential_load_power',
	# 'residential_pv_power',
	# 'cdc_fluview_ilinet',
	# 'cdc_fluview_who_nrevss',
	# 'project_tycho',
	# 'gfc12_load',
	# 'gfc14_load',
	# 'gfc17_load',
	# 'spain',
	# 'pdb',
	# 'elf',
	# 'bull',
	# 'cockatoo',
	# 'hog',
	# 'covid19_energy',
	# 'subseasonal',
	# 'subseasonal_precip']