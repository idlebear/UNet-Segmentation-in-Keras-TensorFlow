
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

from IPython.display import Image, display
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

    #     # Display an example training image and accompanying mask
    #     display(Image(filename=training_image_list[9]))
    #     img = PIL.ImageOps.autocontrast(load_img(training_gt_list[9]))
    #     display(img)

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
        if self.gt_list is not None:
            batch_gt_list = self.gt_list[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) +
                     self.image_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) +
                     self.image_size + (1,), dtype="float32")

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

        return tf.convert_to_tensor(x, dtype=tf.float32), tf.convert_to_tensor(y, dtype=tf.float32)


training_data_list, training_mask_list = load_data_list('./data_road/training/image_2',
                                                        './data_road/training/gt_image_2')


def display_results(data, masks, result):
    # fig = plt.figure()
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)
    t = np.shape(data)

    for j in range(t[0]//5):

        i = random.randint(0, t[0])

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

    plt.show()


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
            recall = 0
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


def train_model(name, experiment, image_size, training_data_list, training_mask_list,
                model_spec=[16, 32, 64, 128, 256], preprocess_list=None,
                preprocess_stretch=False, preprocess_mask=None, keep_image=True,
                load_model=False, epochs=15):

    # make copies of the input array before shuffling
    training_data_list = list(training_data_list)
    training_mask_list = list(training_mask_list)

    random.Random(experiment*42).shuffle(training_data_list)
    random.Random(experiment*42).shuffle(training_mask_list)

    # we're augmenting data -- expand the list of training data
    train_input_img_paths = training_data_list[:-(test_samples
                                                  + val_samples)] * random_factor
    train_target_img_paths = training_mask_list[:-(test_samples
                                                   + val_samples)] * random_factor

    val_input_img_paths = training_data_list[-(
        test_samples + val_samples):-val_samples]
    val_target_img_paths = training_mask_list[-(
        test_samples + val_samples):-val_samples]

    test_input_img_paths = training_data_list[-test_samples:]
    test_target_img_paths = training_mask_list[-test_samples:]

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
        train_gen = RoadSeq(batch_size, image_size,
                            train_input_img_paths, train_target_img_paths, augment_data=True,
                            preprocess_fn=pp.preprocess())
        val_gen = RoadSeq(batch_size, image_size,
                          val_input_img_paths, val_target_img_paths, augment_data=False,
                          preprocess_fn=pp.preprocess())
        test_gen = RoadSeq(len(test_input_img_paths), image_size,
                           test_input_img_paths, test_target_img_paths, augment_data=False,
                           preprocess_fn=pp.preprocess())

    else:
        # use the images as they are
        train_gen = RoadSeq(batch_size, image_size,
                            train_input_img_paths, train_target_img_paths, augment_data=True)
        val_gen = RoadSeq(batch_size, image_size,
                          val_input_img_paths, val_target_img_paths, augment_data=False)
        test_gen = RoadSeq(len(test_input_img_paths), image_size,
                           test_input_img_paths, test_target_img_paths, augment_data=False)

    model_name = name+'.'+str(experiment)+'.h5'
    model = UNet(image_size, model_spec)
    model.compile(optimizer="adam",
                  loss="binary_crossentropy", metrics=["acc"])
    if load_model:
        model.load_weights(model_name)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_name, save_best_only=True)
    ]

    model.fit(train_gen, epochs=epochs, verbose=1,
              validation_data=val_gen, callbacks=callbacks)

    x, y = test_gen.__getitem__(0)
    start = timer()
    results = model.predict(x)
    end = timer()
    prediction_time = (end - start) / len(results)

    results = np.array(results > 0.5).astype(np.uint8)

    return calculate_error(results, test_target_img_paths) + (prediction_time,)


#
# ## Training!
#

# Split our img paths into a training and a validation set
test_samples = len(training_data_list) // test_reserve
val_samples = test_samples

gradient_levels = [
    # [1],
    # [2],
    # [1, 2],
    # [0, 1, 3],
    # [1, 3],
    # [1, 4],  # 3
    # [1, 5],
    # [1, 2, 3],
    [1, 2, 5],  # 1
    # [1, 2, 5],  # 1
    # [1, 3, 5],
    # [2, 4, 6],
    # [1, 2, 3, 4],
    # [0, 1, 2, 5, 7],
    # [1, 2, 5, 7],  # 2
    # [1, 2, 5, 10],
    # [1, 2, 3, 4, 5],
    # [1, 3, 5, 7, 10]
    # [0, 1, 2, 5, 7, 10],
    # [1, 2, 5, 7, 10],
]

model_specs = {
    # 'mini': [8, 16, 32, 64],       # ~0.1M parameters
    # 'mid': [8, 16, 32, 64, 128],   # ~0.5M parameters
    'reg': [16, 32, 64, 128, 256]  # ~2M parameters
}

experiments = 2
rounds = 20
epochs = 10
base = 0

small_image_size = (image_size[0] // 2, image_size[1] // 2)

f = open('statfile.csv', 'w')

f.write('name,experiment,round,precision,recall,f1,accuracy,time\n')  # header row

for name in model_specs:

    # for gl in gradient_levels:
    #     s_lead = '_'.join(['gd_only', name] + [str(i) for i in gl])

    #     for ex in range(3, 3+experiments):
    #         for r in range(base, rounds+base):
    #             data = train_model(name=s_lead, experiment=ex, image_size=image_size, training_data_list=training_data_list,
    #                             training_mask_list=training_mask_list, model_spec=model_specs[
    #                                 name], preprocess_list=gl,
    #                             preprocess_stretch=True, keep_image=False,
    #                             load_model=(r is not 0), epochs=epochs)
    #             s = ','.join([s_lead, str(ex), str(r)]
    #                          + [str(i) for i in data])
    #             s += '\n'
    #             f.write(s)
    #             f.flush()

    s_lead = name
    for ex in range(experiments):

        for r in range(base, rounds+base):
            data = train_model(name=s_lead, experiment=ex, image_size=image_size, training_data_list=training_data_list,
                               training_mask_list=training_mask_list, model_spec=model_specs[
                                   name], preprocess_list=None, load_model=(r is not 0),
                               epochs=epochs)
            s = ','.join([s_lead, str(ex), str(r)] + [str(i) for i in data])
            s += '\n'
            f.write(s)
            f.flush()

    # s_lead = 'gd_only_'+name
    # for gl in gradient_levels:
    #     for r in range(rounds):
    #         s = '_'.join([s_lead] + [str(i) for i in gl])

    #         data = train_model(name=s, image_size=image_size, training_data_list=training_data_list,
    #                         training_mask_list=training_mask_list, model_spec=model_specs[
    #                             name], preprocess_list=gl,
    #                         preprocess_stretch=False, keep_image=False, load_model=True, epochs=epochs)
    #         s = ','.join([s] + [str(i) for i in data])
    #         s += '\n'
    #         f.write(s)
    #         f.flush()


f.close()
