import numpy as np

from data_preprocessing import get_dataset_in_np

TRAIN_FOLDER_PATH = "G:/DL/Iceberg Classifier Challenge/train/data/processed/train.json"
# TEST_FOLDER_PATH = "G:/DL/Iceberg Classifier Challenge/test/data/processed/test.json"

dataset_train_features, dataset_train_labels = get_dataset_in_np(TRAIN_FOLDER_PATH, labels_available=True, normalize=True)
# dataset_test_features = get_dataset_in_np(TEST_FOLDER_PATH, labels_available=False, , normalize=True)

print('dataset_train_features.shape:', dataset_train_features.shape, 'dataset_train_labels.shape:', dataset_train_labels.shape)
