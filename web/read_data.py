
import numpy as np
import os
import random
import glob

from skimage import data, io, filters

NORMALIZED_HEIGHT = 20
NORMALIZED_WIDTH = 10


ANBANI = ['ა', 'ბ', 'გ', 'დ', 'ე', 'ვ', 'ზ', 'თ', 'ი', 'კ', 'ლ', 'მ', 'ნ', 'ო', 'პ', 'ჟ', 'რ', 'ს', 'ტ', 'უ', 'ფ', 'ქ', 'ღ', 'ყ', 'შ', 'ჩ', 'ც', 'ძ', 'წ', 'ჭ', 'ხ', 'ჯ', 'ჰ']
BASE = 'data/'


def read_data(folders):
    # arr = np.zeros((2, 3))

    letter_count = 0

    labels = []

    for i in range(len(ANBANI)):

        curr_count = 0
        for font in folders:
            path = BASE + font + '/' + ANBANI[i]
            if os.path.exists(path):
                images = os.listdir(path)
                curr_count += len(images)
        letter_count += curr_count
        labels += [i] * curr_count


    data_arr = np.zeros((letter_count, NORMALIZED_HEIGHT, NORMALIZED_WIDTH), dtype=np.float64)

    img_count = 0
    for char in ANBANI:
        for font in folders:
            path = BASE + font + '/' + char
            if os.path.exists(path):
                images = os.listdir(path)
                for img_name in images:
                    # print(img_name)
                    img = io.imread(path+'/'+img_name, as_gray = True)
                    img = img.astype(np.float64)
                    data_arr[img_count] = img
                    img_count += 1

    #print(data_arr.shape)
    indexArr = np.arange(data_arr.shape[0])
    indexArr = np.random.permutation(indexArr)
    data_arr = data_arr[indexArr]
    labels = np.array(labels)[indexArr]

    num_train = int(data_arr.shape[0]*0.7)
    remain = data_arr.shape[0] - num_train
    num_val = int(remain/2)
    num_test = remain-num_val
    X_train = data_arr[0:num_train]
    y_train = labels[0:num_train]
    X_val = data_arr[num_train:num_train+num_val]
    y_val = labels[num_train:num_train+num_val]
    X_test = data_arr[num_train+num_val:]
    y_test = labels[num_train+num_val:]

    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }