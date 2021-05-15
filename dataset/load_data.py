import os
from keras_preprocessing.image import img_to_array, load_img
import pickle
import numpy as np
import random

IMG_SIZE = 200


def convert_img_to_numpy(path):
    files = os.listdir(path)

    data_array = np.array([])
    for filename in files:
        try:
            print('Convert to numpy : {}....'.format(filename), end='')
            img = load_img(path + filename)
            img_array = img_to_array(img)
            data_array = np.append(data_array, img_array)
            print('GOOD!')
        except:
            print('Exception !!')
    data_array = np.reshape(data_array, [-1, IMG_SIZE, IMG_SIZE, 3])
    print('RESHAPE SUCCESS!!')

    return data_array


def load_data_convert_numpy(flag):
    if flag == 'train':
        mask_file = 'mask_train_data.pickle'
        no_mask_file = 'no_mask_train_data.pickle'
        mask_path = 'processing3/train/mask/'
        no_mask_path = 'processing3/train/no_mask/'
    elif flag == 'test':
        mask_file = 'mask_test_data.pickle'
        no_mask_file = 'no_mask_test_data.pickle'
        mask_path = 'processing3/test/mask/'
        no_mask_path = 'processing3/test/no_mask/'
    elif flag == 'valid':
        mask_file = './mask_valid_data.pickle'
        no_mask_file = './no_mask_valid_data.pickle'
        mask_path = 'processing3/valid/mask/'
        no_mask_path = 'processing3/valid/no_mask/'
    else:
        print('flag require test or train!')
        return

    if os.path.isfile(mask_file) and os.path.isfile(no_mask_file):
        with open(mask_file, 'rb') as f:
            mask_data = pickle.load(f)
        with open(no_mask_file, 'rb') as f:
            no_mask_data = pickle.load(f)
    else:
        mask_data = convert_img_to_numpy(mask_path)
        no_mask_data = convert_img_to_numpy(no_mask_path)

        with open(mask_file, 'wb') as f:
            pickle.dump(mask_data, f)
        with open(no_mask_file, 'wb') as f:
            pickle.dump(no_mask_data, f)

    y_data = np.array([])
    for i in range(len(mask_data)):
        y_data = np.append(y_data, [1, 0])

    for i in range(len(no_mask_data)):
        y_data = np.append(y_data, [0, 1])

    y_data = np.reshape(y_data, [-1, 2])
    y_data = y_data.astype(np.int)
    x_data = np.append(mask_data, no_mask_data)
    x_data = np.reshape(x_data, [-1, IMG_SIZE, IMG_SIZE, 3])

    return x_data, y_data


def load_dataset(flag, size, is_random):
    x_data, y_data = load_data_convert_numpy(flag=flag)

    if is_random:
        dataset = [(x, y) for x, y in zip(x_data, y_data)]
        random.shuffle(dataset)
        x_data = np.array([x for x, y in dataset])
        y_data = np.array([y for x, y in dataset])

    x_data = x_data[:size]
    y_data = y_data[:size]
    x_data /= 255.

    return x_data, y_data
