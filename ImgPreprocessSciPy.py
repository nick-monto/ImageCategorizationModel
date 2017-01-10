import os
import numpy as np

from scipy.misc import imread
from scipy.misc import imresize

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.utils import np_utils

# input training directory as a string
TRAIN_DIR = ' '
# input test directory as a string
TEST_DIR = ' '
# input the names of the sub folders in trainig directory
CATEGORIES = [' ', ' ']
ROWS = 180  # 720
COLS = 320  # 1280
CHANNELS = 3
seed = 1


def get_images(trainImg):
    """Load files from train folder"""
    trainImg_dir = TRAIN_DIR+'{}'.format(trainImg)
    images = [trainImg+'/'+im for im in os.listdir(trainImg_dir)]
    return images


def read_image(src):
    """Read and resize individual images"""
    im = imread(src, mode='RGB')
    im = imresize(im, (ROWS, COLS), interp='cubic')
    return im


files = []
y_all = []

for trainImg in CATEGORIES:
    trainImg_files = get_images(trainImg)
    files.extend(trainImg_files)

    y_trainImg = np.tile(trainImg, len(trainImg_files))
    y_all.extend(y_trainImg)
    print("{0} photos of {1}".format(len(trainImg_files), trainImg))

y_all = np.array(y_all)

X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, im in enumerate(files):
    X_all[i] = read_image(TRAIN_DIR+im)
    if i % 100 == 0:
        print('Read {} of {} images.'.format(i, len(files)))

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

print('Shape of test is {}.'.format(test.shape))
