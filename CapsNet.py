import nunmpy as np

from data_preprocessing import get_dataset_features

TRAIN_FOLDER_PATH = "G:/DL/Iceberg Classifier Challenge/train/data/processed/train.json"
TEST_FOLDER_PATH = "G:/DL/Iceberg Classifier Challenge/test/data/processed/test.json"

dataset_train_features, dataset_train_labels = get_dataset_features(TRAIN_FOLDER_PATH, labels_available=True)
dataset_test_features = get_dataset_features(TEST_FOLDER_PATH, labels_available=False)

