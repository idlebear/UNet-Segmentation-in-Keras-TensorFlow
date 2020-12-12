from PIL import Image
from timeit import default_timer as timer
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

import PIL
from PIL import ImageOps, Image
import matplotlib.pyplot as plt
import numpy as np
import random
import os

from image_processing import ImagePreprocessFisherize
from image_processing import ImagePreprocessGradient
from image_processing import ImagePreprocessStretchedGradient

# test parameters -- the current model uses 4 compression layers, each halving the image size.
# To ensure that the size of the output is the same as the input, resize the images to be a factor of
# 32 (2^5) -- otherwise we need to do some odd cropping
original_image_size = (375, 1242)
image_size = (224, 768)  # (352, 1216)
batch_size = 14
num_classes = 3

# we're randomly cropping the images -- random factor allows the list to be multiplied to account for this
# data augmentation
random_factor = 3

# reserve 10% of the training images for test -- because there's no ground truth for kitti test
test_reserve = 10


def load_data_list(image_path, mask_path=None):
    image_list = None
    mask_list = None

    image_list = sorted(
        [os.path.join(image_path, fname) for fname in os.listdir(image_path)
            if fname.endswith(".png")]
    )
    if mask_path is not None:
        mask_list = sorted(
            [os.path.join(mask_path, fname) for fname in os.listdir(mask_path)
                if '_road' in fname]
        )

    #     # Display an example training image and accompanying mask
    #     display(Image(filename=training_image_list[9]))
    #     img = PIL.ImageOps.autocontrast(load_img(training_gt_list[9]))
    #     display(img)

    return image_list, mask_list


training_data_list, training_mask_list = load_data_list('./data_road/training/image_2',
                                                        './data_road/training/gt_image_2')


r = random.randint(0, len(training_data_list))

img = Image.open(training_data_list[r])
mask = Image.open(training_mask_list[r])

# ## Model Definition
#
# ### Convolutional Blocks


def display_results(data, masks, result):
    #fig = plt.figure()
    #fig.subplots_adjust(hspace=0.4, wspace=0.4)
    t = np.shape(data)

    for i in range(t[0]):

        fig = plt.figure(figsize=(15, 45))
        ax = fig.add_subplot(1, 4, 1)
        image = np.reshape(data[i], image_size+(3,))
        ax.imshow(image)

        ax = fig.add_subplot(1, 4, 2)
        ax.imshow(np.reshape(masks[i]*255, image_size), cmap="gray")

        ax = fig.add_subplot(1, 4, 3)
        res = np.reshape(result[i]*255, image_size)
        ax.imshow(res, cmap="gray")

        ax = fig.add_subplot(1, 4, 4)
        ax.imshow(image)
        ax.imshow(res, 'Oranges', interpolation='none', alpha=0.7)


def calculate_error(results, mask_list):

    # for each mask in the list of masks, compare the ground truth mask with the
    # prediction

    n = len(mask_list)
    precision_total = 0
    recall_total = 0
    f1_total = 0
    acc_total = 0

    for i in range(n):
        mask = np.array(load_img(mask_list[i])).astype(np.uint8)
        mask = np.squeeze(mask[:, :, 2] == 255)

        mask_size = np.shape(mask)

        # rescale the result to the proper image size
        res = np.squeeze(tf.image.resize(results[i], mask_size) > 0)

        # precision is (True Positive) / (True Positive + False Positive) (Fritsch2013ITSC)
        tp = np.bitwise_and(res, mask)
        tp = np.sum(tp)
        fp = np.bitwise_and(res, np.invert(mask))
        fp = np.sum(fp)
        if not tp and not fp:
            print('Precision Error: {}'.format(mask_list[i]))
            precision = 0
        else:
            precision = tp / (tp + fp)

        # recall is (True Positive) / (True Positive + False Negative)
        # which is the correct bits in the result divided by the total bits in the ground truth
        fn = np.sum(mask) - tp
        tn = np.bitwise_and(np.invert(res), np.invert(mask))
        tn = np.sum(tn)
        if not tp and not fn:
            print('Recall Error: {}'.format(mask_list[i]))
            recal = 0
        else:
            recall = tp / (tp + fn)

        #
        # F1 measure is (1 + beta^2) * ( precision * recall ) / ( beta^2 * precision + recall)
        # and beta is 1
        beta = 1.0
        if not precision and not recall:
            f1 = 0
        else:
            f1 = (1.0+beta**2) * (precision * recall) / \
                (beta**2 * precision + recall)

        # finally, accuracy is (True Positive + True Negative) /
        #                     (True Positive + False Positive + True Negative + False Negative)
        if not tp + fp + tn + fn:
            acc = 0
        else:
            acc = (tp + tn) / (tp + fp + tn + fn)

        # update the totals
        recall_total += recall
        precision_total += precision
        f1_total += f1
        acc_total += acc

    return (precision_total/n, recall_total/n, f1_total/n, acc_total/n)


# # Gradient Smoothing
#
# We apply several levels of gradient smoothing followed by gradient smoothing to add emphasis to the edge regions of the image.
#

img = np.array(img)


prepro = ImagePreprocessGradient([1, 2, 5])
prepro = ImagePreprocessFisherize(prepro)
sm_grad = prepro.preprocess()(img)

fig = plt.figure()
plt.imshow(img)

fig = plt.figure()
plt.imshow(sm_grad)


prepro = ImagePreprocessGradient([1, 2, 5], keep_image=False)
sm_grad = prepro.preprocess()(img)
# sm_grad = smoothed_gradients(np.round(x[0]*255), [0,1,2,3,5,10])

fig = plt.figure()
plt.imshow(sm_grad)

prepro = ImagePreprocessGradient([1, 2, 5], keep_image=True)
sm_grad = prepro.preprocess()(img)
# sm_grad = smoothed_gradients(np.round(x[0]*255), [0,1,2,3,5,10])

fig = plt.figure()
plt.imshow(sm_grad)

plt.show()


#
# ## Training!
#


# Split our img paths into a training and a validation set
test_samples = len(training_data_list) // test_reserve
val_samples = test_samples
