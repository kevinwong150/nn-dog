import numpy as np
from scipy.ndimage import zoom, interpolation, rotate, gaussian_filter
import os
import random

# Data preprocessing -----------------------------------------------------------------------
# input (list of ndarray) and return 2d ndarray ( DATA_SIZE * TOTAL_FEATURES )
def flattenImage(train_data_array):
    flatten_list = list(map(lambda x: x.flatten(), train_data_array))
    flatten_array = np.asarray(flatten_list)
    return flatten_array

# input (ndarray) and return 2d ndarray ( DATA_SIZE * TOTAL_FEATURES )
def meanNormalize(train_data_array, batch_size):
    mean = np.sum(train_data_array, 0) / batch_size
    mean_normalized_list = list(map(lambda x: (x - mean) / 255, train_data_array))
    mean_normalized_array = np.asarray(mean_normalized_list)
    return mean_normalized_array

# Data Augmentation -------------------------------------------------------------------------
# return image itself (no augmentation)
def augmentation_identity(image):
    return image

# input unflattened array and return image flipped along y axis
def augmentation_flip(image):
    return np.fliplr(image)


def augmentation_translate(image):
    return interpolation.shift(image, (random.randint(1,20), random.randint(1,20), 0), order=5, mode='nearest')

def augmentation_rotate(image):
    return rotate(image, random.uniform(-30,30), reshape=False, order=5, mode='nearest')

def augmentation_blur(image):
    return gaussian_filter(image, random.randint(1,5))

# Training Utilities ------------------------------------------------------------------------
def toOnehot(label, totalClass):
    temp = np.zeros((totalClass), dtype=float)
    temp[label] = 1
    return temp

def sigmoid(z, gradient=False):
    if (gradient==True):
        return np.multiply( sigmoid(z), (-sigmoid(z) + 1) )
    else:
        return ( 1 / ( 1 + np.exp(-z) ) )

def cost(y_predict, y, total_train_data):
    yIsOne = np.multiply((-y), np.log(y_predict))
    yIsZero = np.multiply((1-y), np.log(1 - y_predict))
    return (np.sum(yIsOne-yIsZero) / (-1 * total_train_data))

# return 2D ndarray (row = train data, columns = features(with bias column))
def withBiasColumn(m):
    (row, column) = m.shape
    temp = np.ones((row, column+1))
    temp[:,1:] = m
    return temp

def withoutBiasColumn(m):
    (row, column) = m.shape
    temp = np.zeros((row, column-1))
    temp = m[:,1:]
    return temp

# return 1D ndarray (columns = onehot predicted result)
def nn_output_to_onehot_row(output):
    column_index = np.where(output == np.amax(output))[0]
    temp = np.zeros(output.shape)
    temp[column_index] = 1
    return temp

# return 2D ndarray (row = train data, columns = onehot predicted result)
def nn_output_to_onehot(y_predict):
    return np.apply_along_axis(nn_output_to_onehot_row, 1, y_predict)

# return a scalar (the readable label)
def onehot_row_to_label(y_predict):
    return np.where(y_predict == 1)[0][0]

# return 2D ndarray (one column vector) (row = train data, column = readable label)
def onehot_to_label(y_predict):
    return np.apply_along_axis(onehot_row_to_label, 1, y_predict)

def writeInfo(filepath, loop, num):
    if loop % num != 0:
        return

    loop = loop + 1

    exists = os.path.isfile(filepath)
    if exists:
        file = open(filepath, "r")
        epoch = int(file.readlines(1)[0].rstrip().split(":")[1])
        file.close()

        os.remove(filepath)

        file = open(filepath, "w")
        file.write("Epoch Trained: " + str(epoch + num) + "\n")
        file.close()

    else:
        file = open(filepath, "w")
        file.write("Epoch Trained: " + str(loop) + "\n")
        file.close()

def saveTheta(theta, filepath, loop, num):
    if loop % num != 0:
        return

    exists = os.path.isfile(filepath)
    if exists:
        os.remove(filepath)

    Theta_temp = np.asarray(theta)
    np.save(filepath, Theta_temp)

def debug(str, loop=1, num=1):
    if loop % num != 0:
        return
    print(str)

# Return 2D array of onehot prediction result
def prediction(X, y, theta, loop=1, num=1):
    if loop % num != 0:
        return

    input_layer = withBiasColumn(X) # (TOTAL_DATA, 30001)
    hidden_layer_2 = sigmoid(np.dot(input_layer,theta[0].T)) # (TOTAL_DATA, 3000)
    hidden_layer_2 = withBiasColumn(hidden_layer_2) # (TOTAL_DATA, 3001)
    hidden_layer_3 = sigmoid(np.dot(hidden_layer_2,theta[1].T)) # (TOTAL_DATA, 300)
    hidden_layer_3 = withBiasColumn(hidden_layer_3) # (TOTAL_DATA, 301)
    output_layer = sigmoid(np.dot(hidden_layer_3,theta[2].T)) # (TOTAL_DATA, 25)

    return output_layer # (TOTAL_DATA, 25)

# Return accuracy of nn result
def get_accuracy(output, y, TOTAL_DATA, loop=1, num=1):
    if loop % num != 0:
        return

    prediction = onehot_to_label(nn_output_to_onehot(output))

    return np.sum(prediction == y) / TOTAL_DATA
