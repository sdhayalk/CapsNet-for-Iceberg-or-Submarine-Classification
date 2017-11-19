'''this is not a Capsnet implementation, 
but a simple Convolution Neural Network implementation to do basic testing and getting performances
'''

import tensorflow as tf
import numpy as np

from data_preprocessing import get_dataset_in_np, normalize

def weights(shape, name=None):
	return tf.Variable(tf.random_normal(shape), name=name)

def biases(shape, name=None):
	return tf.Variable(tf.random_normal(shape), name=name)


TRAIN_FOLDER_PATH = "G:/DL/Iceberg Classifier Challenge/train/data/processed/train.json"
TEST_FOLDER_PATH = "G:/DL/Iceberg Classifier Challenge/test/data/processed/test.json"

dataset_train_features, dataset_train_labels = get_dataset_in_np(TRAIN_FOLDER_PATH, labels_available=True)
# dataset_test_features = get_dataset_in_np(TEST_FOLDER_PATH, labels_available=False)
dataset_train_features, dataset_test_features = normalize(dataset_train_features, dataset_train_features) # TODO: change to dataset_test_features

print('dataset_train_features.shape:', dataset_train_features.shape, 'dataset_train_labels.shape:', dataset_train_labels.shape)

DIMENSION = dataset_train_features.shape[1]
NUM_CHANNELS = dataset_train_features.shape[3]
NUM_CLASSES = dataset_train_labels.shape[1]
NUM_EXAMPLES = dataset_train_features.shape[0]
BATCH_SIZE = 16
NUM_EPOCHS = 100


x = tf.placeholder(tf.float32, shape=[None, DIMENSION, DIMENSION, NUM_CHANNELS])
y = tf.placeholder(tf.float32, shape=[None, dataset_train_labels.shape[1]])

def convolutional_neural_network(x):
	layer_conv_1 = tf.nn.conv2d(x, weights([3,3,NUM_CHANNELS,32], name='w_conv_1'), strides=[1,1,1,1], padding='SAME') + biases([32], name='b_conv_1')
	layer_conv_1 = tf.nn.relu(layer_conv_1, name='layer_conv_1')

	layer_conv_2 = tf.nn.conv2d(layer_conv_1, weights([3,3,32,64], name='w_conv_2'), strides=[1,1,1,1], padding='SAME') + biases([64], name='b_conv_2')
	layer_conv_2 = tf.nn.relu(layer_conv_2, name='layer_conv_2')

	layer_conv_3 = tf.nn.conv2d(layer_conv_2, weights([3,3,64,128], name='w_conv_3'), strides=[1,1,1,1], padding='SAME') + biases([128], name='b_conv_3')
	layer_conv_3 = tf.nn.relu(layer_conv_3, name='layer_conv_3')

	layer_conv_3_shape = layer_conv_3.get_shape()
	num_features = layer_conv_3[1:4].num_elements()
	layer_3_flattened = tf.reshape(layer_conv_3, [-1, num_features])

	layer_fc_4 = tf.matmul(layer_3_flattened, weights([num_features, 256], name='w_fc_4')) + biases([256], name='b_fc_4')
	layer_fc_4 = tf.nn.relu(layer_fc_4, name='layer_fc_4')

	layer_fc_5 = tf.matmul(layer_fc_4, weights([256, NUM_CLASSES], name='w_fc_5')) + biases([NUM_CLASSES], name='b_fc_5')
	return layer_fc_5


