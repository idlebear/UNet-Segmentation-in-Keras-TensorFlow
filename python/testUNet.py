
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

# from IPython.display import Image, display
import PIL
from PIL import ImageOps

from timeit import default_timer as timer
import matplotlib.pyplot as plt

import numpy as np
import os
import random

# local imports
from model import UNet
from image_processing import ImagePreprocessGradient, ImagePreprocessMask
from image_processing import ImagePreprocessStretchedGradient

print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))


# test parameters -- the current model uses 4 compression layers, each
# halving the image size.  To ensure that the size of the output is the
# same as the input, resize the images to be a factor of
# 32 (2^5) -- otherwise we need to do some odd cropping
original_image_size = (375, 1242)
image_size = (224, 768)  # (352, 1216)
batch_size = 14
num_classes = 3

# we're randomly cropping the images -- random factor allows the list to be
# multiplied to account for this data augmentation
random_factor = 3

# reserve 10% of the training images for test -- because there's no ground
# truth for kitti test
test_reserve = 10


# ### Set up data generator
#
# Data is stored in a training/gt directory -- clean up the images as they are
# loaded
#
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

    return image_list, mask_list


class RoadSeq(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, image_size, training_image_list, training_gt_list=None,
                 augment_data=True, preprocess_fn=None):
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_list = training_image_list
        self.gt_list = training_gt_list
        self.augment_data = augment_data
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.image_list) // self.batch_size

    def __getitem__(self, idx):
        # Now with data augmentation function...

        i = idx * self.batch_size
        batch_image_list = self.image_list[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) +
                     self.image_size + (3,), dtype="float32")

        if self.gt_list is not None:
            batch_gt_list = self.gt_list[i: i + self.batch_size]
            y = np.zeros((self.batch_size,) +
                         self.image_size + (1,), dtype="float32")
        else:
            y = None

        for j in range(len(batch_image_list)):
            image_path = batch_image_list[j]
            if self.gt_list is not None:
                mask_path = batch_gt_list[j]

            if not self.augment_data:
                # load the image and resize it to fit through our filter
                img = load_img(image_path, target_size=self.image_size)
                if self.gt_list is not None:
                    mask = load_img(mask_path, target_size=self.image_size)
            else:
                # load the full size image
                img = load_img(image_path)
                if self.gt_list is not None:
                    mask = load_img(mask_path)

            try:
                img = self.preprocess_fn(img)
            except Exception:
                pass

            img = np.array(img).astype('float32')
            if self.gt_list is not None:
                mask = np.array(mask).astype('float32')

            if self.augment_data:
                if random.random() < 0.3:
                    # resize the image to the new bounds
                    img = tf.image.resize(img, self.image_size)
                    if self.gt_list is not None:
                        mask = tf.image.resize(mask, self.image_size)
                else:
                    # instead of shrinking the image, randomly crop out a rectangle
                    (dy, dx, depth) = np.shape(img)
                    x_offset = max(0, (dx - self.image_size[1])//2)
                    y_offset = max(0, (dy - self.image_size[0])//2)
                    x_offset = random.randint(0, x_offset)
                    y_offset = random.randint(0, y_offset)
                    img = tf.image.crop_to_bounding_box(
                        img, y_offset, x_offset, self.image_size[0], self.image_size[1])
                    if self.gt_list is not None:
                        mask = tf.image.crop_to_bounding_box(
                            mask, y_offset, x_offset, self.image_size[0], self.image_size[1])

                # randomly flip left/right
                if random.random() < 0.5:
                    img = tf.image.flip_left_right(img)
                    if self.gt_list is not None:
                        mask = tf.image.flip_left_right(mask)

            #
            # normalize
            x[j, :, :, :] = img / 255.0
            if self.gt_list is not None:
                # only keep the blue layer (road)
                y[j, :, :, 0] = mask[:, :, 2] / 255.0

        if y is not None:
            y = tf.convert_to_tensor(y, dtype=tf.float32)

        x = tf.convert_to_tensor(x, dtype=tf.float32)

        return x, y


training_data_list, training_mask_list = load_data_list('./data_road/testing/image_2',
                                                        None)


def display_results(data, masks, results):
    # fig = plt.figure()
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)
    t = np.shape(data)

    for j in range(5):

        # i = random.randint(0, t[0])

        plt.figure(figsize)
        # image = np.reshape(data[i], image_size+(3,))
        image = Image.open(data[i])
        image = image.resize((768, 224))
        plt.imshow(image)

        if masks is not None:
            plt.figure()
            plt.imshow(np.reshape(masks[i]*255, image_size), cmap="gray")

        for r in results:
            plt.figure()
            plt.imshow(image)
            plt.imshow(np.reshape(r[i]*255, image_size),
                       'Oranges', interpolation='none', alpha=0.7)

    plt.show()
    pass


def run_demo(name, experiment, image_size, training_data_list, training_mask_list,
             model_spec=[16, 32, 64, 128, 256], preprocess_list=None,
             preprocess_stretch=False, preprocess_mask=None, keep_image=True):

    # make copies of the input array before shuffling
    training_data_list = list(training_data_list)
    random.Random(experiment*42).shuffle(training_data_list)
    test_input_img_paths = training_data_list[-test_samples:]

    if training_mask_list is not None:
        training_mask_list = list(training_mask_list)
        random.Random(experiment*42).shuffle(training_mask_list)
        test_target_img_paths = training_mask_list[-test_samples:]
    else:
        test_target_img_paths = None

    pp = None
    # Chain of preprocessing functions, first one added is performed first
    if preprocess_list is not None:
        # Instantiate data Sequences for each split
        if not preprocess_stretch:
            pp = ImagePreprocessGradient(preprocess_list, keep_image, pp)
        else:
            pp = ImagePreprocessStretchedGradient(preprocess_list, pp)

    if preprocess_mask is not None:
        # Apply mask after gradients - masking first only gets overwritten
        pp = ImagePreprocessMask(preprocess_mask, pp)

    if pp is not None:
        # Instantiate pre-processed data sequences for each split
        test_gen = RoadSeq(len(test_input_img_paths), image_size,
                           test_input_img_paths, test_target_img_paths, augment_data=False,
                           preprocess_fn=pp.preprocess())

    else:
        # use the images as they are
        test_gen = RoadSeq(len(test_input_img_paths), image_size,
                           test_input_img_paths, test_target_img_paths, augment_data=False)

    model_name = name+'.'+str(experiment)+'.h5'
    model = UNet(image_size, model_spec)
    model.compile(optimizer="adam",
                  loss="binary_crossentropy", metrics=["acc"])
    model.load_weights(model_name)

    x, y = test_gen.__getitem__(0)
    results = model.predict(x)
    results = np.array(results > 0.5).astype(np.uint8)

    return results


# Split our img paths into a training and a validation set
test_samples = len(training_data_list) // test_reserve
val_samples = test_samples


model_specs = {
    # 'mini': [8, 16, 32, 64],       # ~0.1M parameters
    # 'mid': [8, 16, 32, 64, 128],   # ~0.5M parameters
    'reg': [16, 32, 64, 128, 256]  # ~2M parameters
}


img_grad = run_demo(name='gd_only_reg_0_1_2_5', experiment=1, image_size=image_size,
                    training_data_list=training_data_list,
                    training_mask_list=training_mask_list,
                    model_spec=model_specs['reg'], preprocess_list=[0, 1, 2, 5])

reg = run_demo(name='reg', experiment=1, image_size=image_size,
               training_data_list=training_data_list,
               training_mask_list=training_mask_list,
               model_spec=model_specs['reg'], preprocess_list=None)

grad = run_demo(name='gd_only_reg_1_2_5', experiment=1, image_size=image_size,
                training_data_list=training_data_list,
                training_mask_list=training_mask_list,
                model_spec=model_specs['reg'], preprocess_list=[1, 2, 5], preprocess_stretch=True)

display_results(
    training_data_list[-test_samples:], None, [img_grad, reg, grad])
