import numpy as np
import pandas as pd
import os

def get_dataset_features(path, labels_available=False):
	train = pd.read_json(path)
	train_id = train.id.values
	train_band_1 = train.band_1.values
	train_band_2 = train.band_2.values
	train_label = train.is_iceberg.values

	dataset_features = []
	dataset_features.append(train_band_1)
	dataset_features.append(train_band_2)
	dataset_features = np.array(dataset_features)

	if labels_available:
		dataset_labels = []
		dataset_labels.append(train_label)
		dataset_labels = np.array(dataset_labels)

		return dataset_features, dataset_labels

	return dataset_features

