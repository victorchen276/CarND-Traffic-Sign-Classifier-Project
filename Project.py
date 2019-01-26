# Load pickled data
import pickle

import numpy as np
from pandas.io.parsers import read_csv
import random
import matplotlib.pyplot as plt

import tensorflow as tf

# TODO: Fill this in based on where you saved the training and testing data

training_file = "./traffic-signs-data/train.p"
validation_file = "./traffic-signs-data/valid.p"
testing_file = "./traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']



### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = y_train.shape[0]

# TODO: Number of validation examples
n_validation = y_valid.shape[0]

# TODO: Number of testing examples.
n_test = y_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
sign_classes, class_indices, class_counts = np.unique(y_train, return_index=True, return_counts=True)

n_classes = class_counts.shape[0]


print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# signnames = read_csv("signnames.csv").values[:, 1]
# col_width = max(len(name) for name in signnames)
#
# for c, c_index, c_count in zip(sign_classes, class_indices, class_counts):
#     print("Class %i: %-*s  Count: %s " % (c, col_width, signnames[c], str(c_count)))
#     fig = pyplot.figure(figsize=(6, 1))
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
#     random_indices = random.sample(range(c_index, c_index + c_count), 10)
#     for i in range(5):
#         axis = fig.add_subplot(1, 10, i + 1, xticks=[], yticks=[])
#         axis.imshow(X_train[random_indices[i]])
#     pyplot.show()
#
# pyplot.bar(np.arange(43), class_counts, align='center')
# pyplot.xlabel('Class')
# pyplot.ylabel('Number of training examples')
# pyplot.xlim([-1, 43])
# pyplot.show()

# get_ipython().magic('matplotlib inline')


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


draw_images_examples(X_train, 16, 4, 'Examples of images from training set')

fig = plt.figure(figsize=(12, 4))
n, bins, patches = plt.hist(y_train, n_classes)
plt.xlabel('Labels')
plt.ylabel('No. of samples')
plt.title('Histogram of training samples')
plt.show()


# X_train_one_label = X_train[np.where(y_train == 0)]
# draw_images_examples(X_train_one_label, 16, 4, 'Examples of images of the same type - Speed limit (20km/h)')


import numpy as np
from sklearn.utils import shuffle
from skimage import exposure


def preprocess_dataset(X, y=None):
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


