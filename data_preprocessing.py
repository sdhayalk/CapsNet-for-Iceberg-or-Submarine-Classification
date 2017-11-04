import numpy as np
import pandas as pd
import os

def get_dataset_in_np(path, labels_available=False):
	data = pd.read_json(path)
	data_id = data.id.values
	data_band_1 = data.band_1.values
	data_band_2 = data.band_2.values
	data_label = data.is_iceberg.values

	dataset_features = []
	for i in range(0, data_band_1.shape[0]):
		temp = []
		temp.append(np.array(data_band_1[i]).reshape((75,75)))
		temp.append(np.array(data_band_2[i]).reshape((75,75)))
		temp = np.array(temp)
		temp = temp.reshape((75,75,2))
		dataset_features.append(temp)

	dataset_features = np.array(dataset_features)

	if labels_available:
		dataset_labels = []
		
		for i in range(0, data_band_1.shape[0]):
			temp = np.zeros(2)
			temp[data_label[i]] = 1
			dataset_labels.append(temp)
		
		dataset_labels = np.array(dataset_labels)
		return dataset_features, dataset_labels

	return dataset_features

