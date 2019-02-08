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

name_values = np.genfromtxt('signnames.csv', skip_header=1, dtype=[('myint','i8'), ('mysring','S55')], delimiter=',')

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
    X, y = load_pickled_data(pickled_data_file, columns=['features', 'labels'])
    X, y = preprocess_dataset(X, y)
    return (X, y)


training_file = "./traffic-signs-data/train.p"
validation_file = "./traffic-signs-data/valid.p"
testing_file = "./traffic-signs-data/test.p"

training_file_preprocessed_file = './traffic-signs-data/train_preprocessed.p'
validation_file_preprocessed_file = './traffic-signs-data/valid_preprocessed.p'
test_file_preprocessed_file = './traffic-signs-data/test_preprocessed.p'

# load saved dataset
X_train_gry, y_train = load_pickled_data(training_file_preprocessed_file, columns=['features', 'labels'])
X_validation_gry, y_validation = load_pickled_data(validation_file_preprocessed_file, columns=['features', 'labels'])
X_testing_gry, y_testing = load_pickled_data(test_file_preprocessed_file, columns=['features', 'labels'])

# X_train_gry = np.sum(X_train_gray/3, axis=3, keepdims=True)
# X_validation_gry = np.sum(X_validation_gray/3, axis=3, keepdims=True)
# X_testing_gry = np.sum(X_testing_gray/3, axis=3, keepdims=True)

print('begin assert')
assert(len(X_train_gry) == len(y_train))
assert(len(X_validation_gry) == len(y_validation))
assert(len(X_testing_gry) == len(y_testing))
print('end assert')
#
# number_to_stop = 8
# figures = {}
# labels = {}
# random_signs = []
# n_train = X_train_gray.shape[0]
# for i in range(number_to_stop):
#     index = random.randint(0, n_train - 1)
#     # labels[i] = name_values[y_train[index]][1].decode('ascii')
#     figures[i] = X_train_gray[index].squeeze()
#     random_signs.append(index)
#
# print(random_signs)
# plot_figures(figures, 4, 2, labels=None)

# draw_images_examples(X_train_gray, 16, 4, 'Examples of images from training set')


# preprocess dataset and dump to file
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


import tensorflow as tf
from tensorflow.contrib.layers import flatten

EPOCHS = 60
BATCH_SIZE = 100


def LeNet(x):
    # # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    # mu = 0
    # sigma = 0.1
    #
    # # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    # conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    # conv1_b = tf.Variable(tf.zeros(6))
    # conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    #
    # # SOLUTION: Activation.
    # conv1 = tf.nn.relu(conv1)
    #
    # # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    # conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #
    # # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    # conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    # conv2_b = tf.Variable(tf.zeros(16))
    # conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    #
    # # SOLUTION: Activation.
    # conv2 = tf.nn.relu(conv2)
    #
    # # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    # conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #
    # # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    # fc0 = flatten(conv2)
    #
    # # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    # fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    # fc1_b = tf.Variable(tf.zeros(120))
    # fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    #
    # # SOLUTION: Activation.
    # fc1 = tf.nn.relu(fc1)
    #
    # # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    # fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    # fc2_b = tf.Variable(tf.zeros(84))
    # fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    #
    # # SOLUTION: Activation.
    # fc2 = tf.nn.relu(fc2)
    #
    # # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    # fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=mu, stddev=sigma))
    # fc3_b = tf.Variable(tf.zeros(10))
    # logits = tf.matmul(fc2, fc3_W) + fc3_b
    #
    # return logits
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    W1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    x = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='VALID')
    b1 = tf.Variable(tf.zeros(6))
    x = tf.nn.bias_add(x, b1)
    print("layer 1 shape:", x.get_shape())

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    W2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    x = tf.nn.conv2d(x, W2, strides=[1, 1, 1, 1], padding='VALID')
    b2 = tf.Variable(tf.zeros(16))
    x = tf.nn.bias_add(x, b2)

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    x = flatten(x)

    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    W3 = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    b3 = tf.Variable(tf.zeros(120))
    x = tf.add(tf.matmul(x, W3), b3)

    # TODO: Activation.
    x = tf.nn.relu(x)

    # Dropout
    x = tf.nn.dropout(x, keep_prob)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    W4 = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    b4 = tf.Variable(tf.zeros(84))
    x = tf.add(tf.matmul(x, W4), b4)

    # TODO: Activation.
    x = tf.nn.relu(x)

    # Dropout
    x = tf.nn.dropout(x, keep_prob)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    W5 = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    b5 = tf.Variable(tf.zeros(43))
    logits = tf.add(tf.matmul(x, W5), b5)

    return logits


from tensorflow.contrib.layers import flatten
def LeNet2(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    W1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma), name="W1")
    x = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='VALID')
    b1 = tf.Variable(tf.zeros(6), name="b1")
    x = tf.nn.bias_add(x, b1)
    print("layer 1 shape:", x.get_shape())

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer1 = x

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    W2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma), name="W2")
    x = tf.nn.conv2d(x, W2, strides=[1, 1, 1, 1], padding='VALID')
    b2 = tf.Variable(tf.zeros(16), name="b2")
    x = tf.nn.bias_add(x, b2)

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer2 = x

    # TODO: Layer 3: Convolutional. Output = 1x1x400.
    W3 = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 400), mean=mu, stddev=sigma), name="W3")
    x = tf.nn.conv2d(x, W3, strides=[1, 1, 1, 1], padding='VALID')
    b3 = tf.Variable(tf.zeros(400), name="b3")
    x = tf.nn.bias_add(x, b3)

    # TODO: Activation.
    x = tf.nn.relu(x)
    layer3 = x

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    layer2flat = flatten(layer2)
    print("layer2flat shape:", layer2flat.get_shape())

    # Flatten x. Input = 1x1x400. Output = 400.
    xflat = flatten(x)
    print("xflat shape:", xflat.get_shape())

    # Concat layer2flat and x. Input = 400 + 400. Output = 800
    # x = tf.concat_v2([xflat, layer2flat], 1)
    x = tf.concat([xflat, layer2flat], 1)
    print("x shape:", x.get_shape())


    # Dropout
    x = tf.nn.dropout(x, keep_prob)

    # TODO: Layer 4: Fully Connected. Input = 800. Output = 43.
    W4 = tf.Variable(tf.truncated_normal(shape=(800, 43), mean=mu, stddev=sigma), name="W4")
    b4 = tf.Variable(tf.zeros(43), name="b4")
    logits = tf.add(tf.matmul(x, W4), b4)

    # TODO: Activation.
    # x = tf.nn.relu(x)

    # TODO: Layer 5: Fully Connected. Input = 120. Output = 84.
    # W5 = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    # b5 = tf.Variable(tf.zeros(84))
    # x = tf.add(tf.matmul(x, W5), b5)

    # TODO: Activation.
    # x = tf.nn.relu(x)

    # TODO: Layer 6: Fully Connected. Input = 84. Output = 43.
    # W6 = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    # b6 = tf.Variable(tf.zeros(43))
    # logits = tf.add(tf.matmul(x, W6), b6)

    return logits

'''
Features and Labels
Train LeNet to classify MNIST data.

x is a placeholder for a batch of input images. y is a placeholder for a batch of output labels.

You do not need to modify this section.
'''

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)  # probability to keep units
one_hot_y = tf.one_hot(y, 43)

'''
Training Pipeline
Create a training pipeline that uses the model to classify MNIST data.

You do not need to modify this section.
'''

rate = 0.0009

# logits = LeNet(x)
logits = LeNet2(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

'''
Model Evaluation
Evaluate how well the loss and accuracy of the model for a given dataset.

You do not need to modify this section.
'''

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        # accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

'''
Train the Model
Run the training data through the training pipeline to train the model.

Before each epoch, shuffle the training set.

After each epoch, measure the loss and accuracy of the validation set.

Save the model after training.

You do not need to modify this section.
'''

# X_validation_gray, y_validation
# X_testing_gray, y_testing

# X_train_gray, y_train

print('train the model')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_gry)

    print("Training...")
    print('num_examples: '+str(num_examples))
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train_gry, y_train)

        assert (len(X_train) == len(y_train))
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            # sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            # break
        #
        validation_accuracy = evaluate(X_validation_gry, y_validation)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        # break;

    saver.save(sess, './lenet')
    print("Model saved")


'''
Evaluate the Model
Once you are completely satisfied with your model, evaluate the performance of the model on the test set.

Be sure to only do this once!

If you were to measure the performance of your trained model on the test set, then improve your model, and then measure the performance of your model on the test set again, that would invalidate your test results. You wouldn't get a true measure of how well your model would perform against real data.

You do not need to modify this section.
'''


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_testing_gry, y_testing)
    print("Test Accuracy = {:.3f}".format(test_accuracy))