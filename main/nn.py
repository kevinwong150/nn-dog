import os
import tensorflow as tf
import numpy as np
from PIL import Image
import random
import util
import time

# Constant, modify the constant to adjust the training of neural network
theta_1 = None
theta_2 = None
theta_3 = None
Theta = [theta_1, theta_2, theta_3]
TOTAL_TRAIN_DATA = None
TOTAL_TEST_DATA = None
RANDOM_SEED = 1 # Configurable
LEARNING_RATE = 0.03 # Configurable
(BATCH_SIZE, BATCH_LIST, TOTAL_TRAIN_BATCH) = (None, None, None)
AUGMENTATION_METHOD = [
    util.augmentation_flip,
    util.augmentation_translate,
    util.augmentation_rotate,
    util.augmentation_blur,
    util.augmentation_identity
    ] # Configurable
TOTAL_AUGMENTATION_METHOD = len(AUGMENTATION_METHOD)
PROGRAM_START_TIME = time.time()
DEBUG_EPOCH_INTERVAL = 5 # Configurable

TOTAL_CLASS = 25 # Configurable
INPUT_LAYER = 30000 # Configurable
LAYER_2 = 3000 # Configurable
LAYER_3 = 800 # Configurable
OUTPUT_LAYER = TOTAL_CLASS

TEST_DATA_PATH = "../data/testing.npy"
TRAIN_DATA_PATH = "../data/train_data_array.npy"
TRAIN_LABEL_PATH = "../data/train_label_onehot.npy"
TRAIN_THETA_PATH = "../data/trained_theta.npy"
TRAIN_INFO_PATH = "../data/training_info.txt"

# Load Test Data + Test Label
Dictionary = np.load(TEST_DATA_PATH, encoding="latin1").item()
print("Test set dictionary loaded.")
# (['shapes', 'file_name', 'original', 'reshaped', 'label'])
test_label = (Dictionary['label'])
test_label = np.asarray(test_label)
test_label_onehot = list(map(lambda x: util.toOnehot(x, TOTAL_CLASS), test_label))
test_label_onehot = np.asarray(test_label_onehot) # (250, 25)
(TOTAL_TEST_DATA, _) = test_label_onehot.shape

test_data_array = (Dictionary['reshaped'])
test_data_array = list(map(lambda x: x.astype(np.uint8).flatten(), test_data_array))
test_data_array = np.asarray(test_data_array) # (250, 30000)
test_data_array_mean = np.sum(test_data_array, axis=0)/TOTAL_TEST_DATA
test_data_array = list(map(lambda x: (x - test_data_array_mean) / 255, test_data_array.tolist()))
test_data_array = np.asarray(test_data_array) # (250, 30000)

# Load Train Data + Train Label
train_label_onehot = np.load(TRAIN_LABEL_PATH)
train_label = util.onehot_to_label(train_label_onehot)
train_data_array = np.load(TRAIN_DATA_PATH)

# Get total # of data
(TOTAL_TRAIN_DATA, _) = train_label_onehot.shape

# Input layer = 30000 features + 1 bias
# Output Layer = 25 class
# Designed neural network = 4 layer = 30001 + 3001 + 301 + 25

# Fixed random seed
np.random.seed(RANDOM_SEED)

# Try to load trained Theta
exists = os.path.isfile(TRAIN_THETA_PATH)
if exists:
    Theta_temp = np.load(TRAIN_THETA_PATH)
    theta_1 = Theta_temp[0] # (3000, 30001)
    theta_2 = Theta_temp[1] # (300, 3001)
    theta_3 = Theta_temp[2] # (25, 301)
    print("Reload trained theta.")
else:
    theta_1 = 2 * np.random.random((LAYER_2, INPUT_LAYER + 1)) - 1 # (3000, 30001)
    theta_2 = 2 * np.random.random((LAYER_3, LAYER_2 + 1)) - 1 # (300, 3001)
    theta_3 = 2 * np.random.random((OUTPUT_LAYER, LAYER_3 + 1)) - 1 # (25, 301)
    print("Initialize theta.")

total_iteration = 0

# adjust the batch-size and batch number for training, (maximum batch size = TOTAL_TRAIN_DATA, maximun batch number * batch size <= TOTAL_TRAIN_DATA)
(BATCH_SIZE, BATCH_LIST, TOTAL_TRAIN_BATCH) = util.setBatch(1000, 3) # Configurable

for epoch in range(10000):

    start_time = time.time()

    util.debug("Epoch #" + str(epoch))

    DEBUG_EPOCH = (epoch, DEBUG_EPOCH_INTERVAL)

    # Name the variable for epoch training
    Theta = [theta_1, theta_2, theta_3]
    X = train_data_array
    y = train_label_onehot

    util.debug("Shuffle and get batch", DEBUG_EPOCH)
    shuffle_order = np.arange(TOTAL_TRAIN_DATA)
    np.random.shuffle(shuffle_order)
    X_shuffle = X[shuffle_order]
    X_batch = np.split(X_shuffle, BATCH_LIST) # batch size = (1000, 1000, 1000, 1728)
    y_shuffle = y[shuffle_order]
    y_batch = np.split(y_shuffle, BATCH_LIST) # batch size = (1000, 1000, 1000, 1728)
    train_label_shuffle = train_label[shuffle_order]
    train_label_batch = np.split(train_label_shuffle, BATCH_LIST) # batch size = (1000, 1000, 1000, 1728)

    for iteration in range(TOTAL_TRAIN_BATCH):

        DEBUG_ITERATION = (total_iteration, DEBUG_EPOCH_INTERVAL * TOTAL_TRAIN_BATCH)

        # Data Augmentation
        util.debug("Data augmentation and preprocessing", DEBUG_ITERATION)
        operaion = np.random.randint(TOTAL_AUGMENTATION_METHOD, size=(BATCH_SIZE))
        for data_number in range(BATCH_SIZE):
            data = X_batch[iteration][data_number]
            method_code = operaion[data_number]
            X_batch[iteration][data_number] = AUGMENTATION_METHOD[method_code](data)

        # Preprocess data
        X_batch[iteration] = util.flattenImage(X_batch[iteration])
        X_batch[iteration] = util.meanNormalize(X_batch[iteration], BATCH_SIZE)

        # Forward propagation
        util.debug("Forward propagation", DEBUG_ITERATION)
        input_layer = util.withBiasColumn(X_batch[iteration]) # (BATCH_SIZE, 30001)
        hidden_layer_2 = util.sigmoid(np.dot(input_layer,theta_1.T)) # (BATCH_SIZE, 3000)
        hidden_layer_2 = util.withBiasColumn(hidden_layer_2) # (BATCH_SIZE, 3001)
        hidden_layer_3 = util.sigmoid(np.dot(hidden_layer_2,theta_2.T)) # (BATCH_SIZE, 300)
        hidden_layer_3 = util.withBiasColumn(hidden_layer_3) # (BATCH_SIZE, 301)
        output_layer = util.sigmoid(np.dot(hidden_layer_3,theta_3.T)) # (BATCH_SIZE, 25)

        # Predict Accuracy of Train Data
        train_accuracy = util.get_accuracy(output_layer, train_label_batch[iteration], BATCH_SIZE, DEBUG_ITERATION)
        util.debug("Predict Train Data Accuracy = " + str(train_accuracy), DEBUG_ITERATION)

        # Predict Accuracy of Test Data
        util.debug("Predict Test Data", DEBUG_ITERATION)
        prediction = util.prediction(test_data_array, test_label_onehot, Theta, DEBUG_ITERATION)
        test_accuracy = util.get_accuracy(prediction, test_label, TOTAL_TEST_DATA, DEBUG_ITERATION)
        util.debug("Predict Test Data Accuracy = " + str(test_accuracy), DEBUG_ITERATION)

        # Compute output error
        output_layer_error = y_batch[iteration] - util.nn_output_to_onehot(output_layer) # (BATCH_SIZE, 25)
        cost = util.cost(output_layer, y_batch[iteration], BATCH_SIZE) # (1,) (scalar)
        util.debug("Cost: " + str(cost), DEBUG_ITERATION)

        # Backward propagation
        util.debug("Backward propagation", DEBUG_ITERATION)
        hidden_layer_3_error = np.multiply(
                                np.dot(output_layer_error, theta_3), # (BATCH_SIZE, 301)
                                util.sigmoid(util.withBiasColumn(np.dot(hidden_layer_2, theta_2.T)), True) # (BATCH_SIZE, 301)
                                ) # (BATCH_SIZE, 301)
        hidden_layer_3_error = util.withoutBiasColumn(hidden_layer_3_error) # (BATCH_SIZE, 300)

        hidden_layer_2_error = np.multiply(
                                np.dot(hidden_layer_3_error, theta_2), # (BATCH_SIZE, 3001)
                                util.sigmoid(util.withBiasColumn(np.dot(input_layer, theta_1.T)), True) # (BATCH_SIZE, 3001)
                                ) # (BATCH_SIZE, 3001)
        hidden_layer_2_error = util.withoutBiasColumn(hidden_layer_2_error) # (BATCH_SIZE, 3000)

        theta_3_delta = np.dot(output_layer_error.T, hidden_layer_3) / BATCH_SIZE # (25, 301)

        theta_2_delta = np.dot(hidden_layer_3_error.T, hidden_layer_2) / BATCH_SIZE # (300, 3001)

        theta_1_delta = np.dot(hidden_layer_2_error.T, input_layer) / BATCH_SIZE # (3000, 30001)

        theta_1 = theta_1 + LEARNING_RATE * theta_1_delta

        theta_2 = theta_2 + LEARNING_RATE * theta_2_delta

        theta_3 = theta_3 + LEARNING_RATE * theta_3_delta

        total_iteration = total_iteration + 1
        util.debug("Iteration Complete", DEBUG_ITERATION)

    util.debug("Save Theta", DEBUG_EPOCH)
    Theta = [theta_1, theta_2, theta_3]
    util.saveTheta(Theta, TRAIN_THETA_PATH, DEBUG_EPOCH)
    util.writeInfo(TRAIN_INFO_PATH, DEBUG_EPOCH)
    # a = np.load('../data/trained_theta.npy')

    util.debug("Time for epoch " + str(epoch) + " = " + str(time.time() - start_time) + " seconds", DEBUG_EPOCH)
    util.debug("Total program time = " + str((time.time() - PROGRAM_START_TIME)/60) + " minutes", DEBUG_EPOCH)
