import os
import cv2  # random
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.utils import np_utils

# input the directory for the images you are using for training
TRAIN_DIR = ' '
# input the directory for the images you are using for test
TEST_DIR = ' '
# Input the names of the class folders within your training folder
CATEGORIES = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
ROWS = 180  # 720
COLS = 320  # 1280
CHANNELS = 3
seed = 1


def get_images(training):
    """Load files from train folder"""
    training_dir = TRAIN_DIR+'{}'.format(training)
    images = [training+'/'+im for im in os.listdir(training_dir)]
    return images


def read_image(src):
    """Read and resize individual images"""
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (COLS, ROWS), interpolation=cv2.INTER_CUBIC)
    return im


files = []
y_all = []

for training in CATEGORIES:
    training_files = get_images(training)
    files.extend(training_files)

    y_training = np.tile(training, len(training_files))
    y_all.extend(y_training)
    print("{0} photos of {1}".format(len(training_files), training))

y_all = np.array(y_all)

X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, im in enumerate(files):
    X_all[i] = read_image(TRAIN_DIR+im)
    if i % 100 == 0:
        print('Processed {} of {}'.format(i, len(files)))

print('Shape of X_all is {}.'.format(X_all.shape))

# One Hot Encoding Labels
y_all = LabelEncoder().fit_transform(y_all)
y_all = np_utils.to_categorical(y_all)

# Split into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all,
                                                      test_size=0.2,
                                                      random_state=seed,
                                                      stratify=y_all)

print('Shape of X_train is {}.'.format(X_train.shape))
print('Shape of X_valid is {}.'.format(X_valid.shape))
print('Shape of y_train is {}.'.format(y_train.shape))
print('Shape of y_valid is {}.'.format(y_valid.shape))

test_files = [im for im in os.listdir(TEST_DIR)]
test = np.ndarray((len(test_files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, im in enumerate(test_files):
    test[i] = read_image(TEST_DIR+im)
    if i % 100 == 0:
        print('Processed {} of {}'.format(i, len(files)))

print('Shape of test is {}.'.format(test.shape))
