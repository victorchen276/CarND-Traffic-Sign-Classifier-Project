# Reinitialize and re-import if starting a new kernel here
import matplotlib.pyplot as plt
# %matplotlib inline

import tensorflow as tf
import numpy as np
import cv2

print('done')

### Load the images and plot them here.
### Feel free to use as many code cells as needed.

#reading in an image
import glob
import matplotlib.image as mpimg

fig, axs = plt.subplots(2,4, figsize=(4, 2))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()

my_images = []

for i, img in enumerate(glob.glob('./my-found-traffic-signs/*x.png')):
    image = cv2.imread(img)
    axs[i].axis('off')
    axs[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    my_images.append(image)

my_images = np.asarray(my_images)

my_images_gry = np.sum(my_images/3, axis=3, keepdims=True)

my_images_normalized = (my_images_gry - 128)/128

print(my_images_normalized.shape)

### Visualize the softmax probabilities here.
### Feel free to use as many code cells as needed.

softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=3)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./lenet.meta')
    saver.restore(sess, "./lenet")
    my_softmax_logits = sess.run(softmax_logits, feed_dict={x: my_images_normalized, keep_prob: 1.0})
    my_top_k = sess.run(top_k, feed_dict={x: my_images_normalized, keep_prob: 1.0})

    fig, axs = plt.subplots(len(my_images), 4, figsize=(12, 14))
    fig.subplots_adjust(hspace=.4, wspace=.2)
    axs = axs.ravel()

    for i, image in enumerate(my_images):
        axs[4 * i].axis('off')
        axs[4 * i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[4 * i].set_title('input')
        guess1 = my_top_k[1][i][0]
        index1 = np.argwhere(y_validation == guess1)[0]
        axs[4 * i + 1].axis('off')
        axs[4 * i + 1].imshow(X_validation[index1].squeeze(), cmap='gray')
        axs[4 * i + 1].set_title('top guess: {} ({:.0f}%)'.format(guess1, 100 * my_top_k[0][i][0]))
        guess2 = my_top_k[1][i][1]
        index2 = np.argwhere(y_validation == guess2)[0]
        axs[4 * i + 2].axis('off')
        axs[4 * i + 2].imshow(X_validation[index2].squeeze(), cmap='gray')
        axs[4 * i + 2].set_title('2nd guess: {} ({:.0f}%)'.format(guess2, 100 * my_top_k[0][i][1]))
        guess3 = my_top_k[1][i][2]
        index3 = np.argwhere(y_validation == guess3)[0]
        axs[4 * i + 3].axis('off')
        axs[4 * i + 3].imshow(X_validation[index3].squeeze(), cmap='gray')
        axs[4 * i + 3].set_title('3rd guess: {} ({:.0f}%)'.format(guess3, 100 * my_top_k[0][i][2]))