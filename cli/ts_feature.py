import pandas as pd

import datasets


from uni2ts.common.env import env
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

from tsfeatures import tsfeatures
from tsfeatures import entropy, stability, lumpiness, hurst, stl_features
from collections import defaultdict


class TSFeature:
	def __init__(self, features=None, threads=64):
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
	def hf_dataset_to_panel(dataset):
		# Convert HuggingFace dataset to pandas DataFrame
		# data format of hf:
		# {'item_id': 'Fox_assembly_Cathy', 'start': array('2016-01-01T00:00:00', dtype='datetime64[s]'), 'freq': 'H', 'target': array([43.29, 43.26, 43.13, ...,  2.3 ,  0.75,  3.76], dtype=float32)}
		# data format of panel:
		# unique_id, ds, y

		panel_data = []
		for data in dataset:
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

	def calculate_features(self, dataset, is_save=False, dataset_name=None):
		# Convert HuggingFace dataset to pandas DataFrame
		panel, freq_str = self.hf_dataset_to_panel(dataset)
		freq = self.freq_dict[freq_str]
		
		# Calculate features
		result = tsfeatures(panel, freq=freq, threads=self.threads, features=self.features)

		# Save features to disk
		if is_save:
			result.to_csv(env.LOTSA_V1_PATH / "features" / f"{dataset_name}_tsf.csv", index=True)
		
		return result

# Example usage
# Assuming `dataset` is a HuggingFace dataset with the specified format
# ts_feature = TSFeature(features=[acf_features], freq=7)
# features_result = ts_feature.calculate_features(dataset)


# some code to test the class
if __name__ == '__main__':
	# # Example dataset
	import datasets
	from uni2ts.common.env import env
	
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

	for dataset in dataset_list:
		print(f"Processing dataset: {dataset}")
		dataset = datasets.load_from_disk(str(env.LOTSA_V1_PATH / dataset)).with_format("numpy")
		ts_feat = TSFeature()
		try:
			ts_feat.calculate_features(dataset, is_save=True, dataset_name=dataset)
		except Exception as e:
			print(f"Error in dataset: {dataset}")
			print(e)
			continue
		
