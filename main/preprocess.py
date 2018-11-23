import os
import numpy as np
import PIL
import random
import util
import time

# Constant
RAW_TRAIN_DATA_PATH = "../train_reshaped/"
RAW_TRAIN_LABEL_PATH = "../train_label.txt"

# GET TRAINING DATA ======================================================
file = open(RAW_TRAIN_LABEL_PATH, "r")
train_data_file_name = []
train_data_array = []
train_label = []
train_label_onehot = []

image_array_mean = np.zeros((100, 100, 3), dtype=float)

print("Start loading training data.")
for line in file.readlines():
    line = line.rstrip().split(",")

    image_path = RAW_TRAIN_DATA_PATH + line[0]
    image_label = int(line[1])

    # -------------------- Read image
    image = PIL.Image.open(image_path)
    # image.show()
    image_array = np.array(image)

    # -------------------- Generate image onehot label
    image_label_onehot = util.toOnehot(image_label, TOTAL_CLASS)

    #-------------------- Add to list
    train_data_file_name.append(image_path)
    train_data_array.append(image_array)
    train_label.append(image_label)
    train_label_onehot.append(image_label_onehot)

print("Training data loaded.")

# Turn list into ndarray
train_data_array = np.asarray(train_data_array) # (TOTAL_TRAIN_DATA, 30000)
train_label_onehot = np.asarray(train_label_onehot) # (TOTAL_TRAIN_DATA, 25)

# Save/Load by savetxt, loadtxt
# np.savetxt('../data/train_data_array.txt', train_data_array)
# a = np.loadtxt('../data/train_data_array.txt', dtype=float)
# np.savetxt('../data/train_label_onehot.txt', train_label_onehot)
# b = np.loadtxt('../data/train_label_onehot.txt', dtype=float)

np.save('../data/train_data_array.npy', train_data_array)
# c = np.load('../data/train_data_array.npy')
np.save('../data/train_label_onehot.npy', train_label_onehot)
# d = np.load('../data/train_label_onehot.npy')
print("Training data saved.")
