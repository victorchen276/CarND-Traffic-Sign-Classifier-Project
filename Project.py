# Load pickled data
import pickle

import numpy as np
from pandas.io.parsers import read_csv
import random
import matplotlib.pyplot as plt

import tensorflow as tf

# # TODO: Fill this in based on where you saved the training and testing data
#
# training_file = "./traffic-signs-data/train.p"
# validation_file = "./traffic-signs-data/valid.p"
# testing_file = "./traffic-signs-data/test.p"
#
# with open(training_file, mode='rb') as f:
#     train = pickle.load(f)
# with open(validation_file, mode='rb') as f:
#     valid = pickle.load(f)
# with open(testing_file, mode='rb') as f:
#     test = pickle.load(f)
#
# X_train, y_train = train['features'], train['labels']
# X_valid, y_valid = valid['features'], valid['labels']
# X_test, y_test = test['features'], test['labels']
#
#
#
# ### Replace each question mark with the appropriate value.
# ### Use python, pandas or numpy methods rather than hard coding the results
#
name_values = np.genfromtxt('signnames.csv', skip_header=1, dtype=[('myint','i8'), ('mysring','S55')], delimiter=',')
#
# # TODO: Number of training examples
# n_train = y_train.shape[0]
#
# # TODO: Number of validation examples
# n_validation = y_valid.shape[0]
#
# # TODO: Number of testing examples.
# n_test = y_test.shape[0]
#
# # TODO: What's the shape of an traffic sign image?
# image_shape = X_train[0].shape
#
# # TODO: How many unique classes/labels there are in the dataset.
# sign_classes, class_indices, class_counts = np.unique(y_train, return_index=True, return_counts=True)
#
# n_classes = class_counts.shape[0]
#
#
# print("Number of training examples =", n_train)
# print("Number of testing examples =", n_test)
# print("Image data shape =", image_shape)
# print("Number of classes =", n_classes)
#
#
#
def draw_images_examples(image_array, grid_x, grid_y, title):
    fig = plt.figure(figsize=(grid_x, grid_y))
    fig.suptitle(title, fontsize=20)

    for i in range(1, grid_y * grid_x):
        index = random.randint(0, len(image_array))
        image = image_array[index].squeeze()

        plt.subplot(grid_y, grid_x, i)
        plt.imshow(image)
        plt.axis('off')

    plt.show()
#
#
# draw_images_examples(X_train, 16, 4, 'Examples of images from training set')
#
# fig = plt.figure(figsize=(12, 4))
# n, bins, patches = plt.hist(y_train, n_classes)
# plt.xlabel('Labels')
# plt.ylabel('No. of samples')
# plt.title('Histogram of training samples')
# plt.show()
#
#
# # X_train_one_label = X_train[np.where(y_train == 0)]
# # draw_images_examples(X_train_one_label, 16, 4, 'Examples of images of the same type - Speed limit (20km/h)')
#
#
# import numpy as np
# from sklearn.utils import shuffle
# from skimage import exposure
#
#
def plot_figures(figures, nrows=1, ncols=1, labels=None):
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 14))
    axs = axs.ravel()
    for index, title in zip(range(len(figures)), figures):
        axs[index].imshow(figures[title], plt.gray())
        if (labels != None):
            axs[index].set_title(labels[index])
        else:
            axs[index].set_title(title)

        axs[index].set_axis_off()

    plt.tight_layout()
    plt.show()

from skimage import exposure
from sklearn.utils import shuffle

def preprocess_dataset(X, y=None):
    print('preprocess_dataset')
    # Convert to grayscale, e.g. single Y channel
    X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]
    # Scale features to be in [0, 1]
    X = (X / 255.).astype(np.float32)

    # Apply localized histogram localization
    for i in range(X.shape[0]):
        X[i] = exposure.equalize_adapthist(X[i])

    if y is not None:
        # Convert to one-hot encoding. Convert back with `y = y.nonzero()[1]`
        y = np.eye(43)[y]
        # Shuffle the data
        X, y = shuffle(X, y)

    # Add a single grayscale channel
    X = X.reshape(X.shape + (1,))
    return X, y


def load_pickled_data(file, columns):
    print('load_pickled_data')
    with open(file, mode='rb') as f:
        dataset = pickle.load(f)
    return tuple(map(lambda c: dataset[c], columns))

def load_and_process_data(pickled_data_file):
    print('load_and_process_data')
    X, y = load_pickled_data(pickled_data_file, columns=['features', 'labels'])
    X, y = preprocess_dataset(X, y)
    return (X, y)


training_file = "./traffic-signs-data/train.p"
validation_file = "./traffic-signs-data/valid.p"
testing_file = "./traffic-signs-data/test.p"

training_file_preprocessed_file = './traffic-signs-data/train_preprocessed.p'
validation_file_preprocessed_file = './traffic-signs-data/valid_preprocessed.p'
test_file_preprocessed_file = './traffic-signs-data/test_preprocessed.p'


X_train_gray, y_train = load_pickled_data(training_file_preprocessed_file, columns=['features', 'labels'])
# X_valid_gray, _ = preprocess_dataset(X_valid, y_train)
# X_test_gray, _ = preprocess_dataset(X_test, y_train)
#
number_to_stop = 8
figures = {}
labels = {}
random_signs = []
n_train = X_train_gray.shape[0]
for i in range(number_to_stop):
    index = random.randint(0, n_train - 1)
    # labels[i] = name_values[y_train[index]][1].decode('ascii')
    figures[i] = X_train_gray[index].squeeze()
    random_signs.append(index)

print(random_signs)
plot_figures(figures, 4, 2, labels=None)

# draw_images_examples(X_train_gray, 16, 4, 'Examples of images from training set')



# X_train, y_train = load_and_process_data(training_file)
# pickle.dump({
#         "features" : X_train,
#         "labels" : y_train
#     }, open(training_file_preprocessed_file, "wb"))
# print("Preprocessed balanced training dataset saved in", training_file_preprocessed_file)
#
# X_train, y_train = load_and_process_data(validation_file)
# pickle.dump({
#         "features":X_train,
#         "labels":y_train
#     }, open(validation_file_preprocessed_file, "wb" ) )
# print("Preprocessed extended training dataset saved in", validation_file_preprocessed_file)
#
# X_test, y_test = load_and_process_data(testing_file)
# pickle.dump({
#         "features":X_test,
#         "labels":y_test
#     }, open(test_file_preprocessed_file, "wb" ) )
# print("Preprocessed extended testing dataset saved in", test_file_preprocessed_file)
