from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import numpy as np

# # Gradient Smoothing
#
# We apply several levels of gradient smoothing followed by gradient
# smoothing to add emphasis to the edge regions of the image.
#


def gradient_image(img):
    # kernel gradients
    dx = np.array([[-1, 0, 1]])
    dy = np.array([[-1], [0], [1]])

    shape = np.shape(img)
    grad_img = np.zeros(shape)
    for i in range(shape[2]):
        img_dx = convolve2d(img[:, :, i], dx, mode='same', boundary='symm')
        img_dy = convolve2d(img[:, :, i], dy, mode='same', boundary='symm')
        grad_img[:, :, i] = np.sqrt(np.square(img_dx) + np.square(img_dy))

    return np.round((grad_img / np.max(grad_img))*255.0).astype(np.uint8)


def gradient_layer(img):
    # kernel gradients
    dx = np.array([[-1, 0, 1]])
    dy = np.array([[-1], [0], [1]])

    grad_img = np.zeros(np.shape(img))
    img_dx = convolve2d(img, dx, mode='same', boundary='symm')
    img_dy = convolve2d(img, dy, mode='same', boundary='symm')
    grad_img = np.round(
        np.sqrt(np.square(img_dx) + np.square(img_dy))).astype(np.uint8)

    return grad_img


def smoothed_gradients(img, sigmas, keep_image):

    img = np.array(img)
    shape = np.shape(img)
    if keep_image:
        sg_img = np.copy(img).astype(np.uint8)
    else:
        sg_img = np.zeros(shape).astype(np.uint8)

    for sigma in sigmas:
        for i in range(shape[2]):
            # for each layer in the image

            if sigma:
                # if a non-zero sigma is supplied, use it to smooth the image
                # before collecting the gradients
                gaus_img = gaussian_filter(img[:, :, i], sigma)
            else:
                gaus_img = img[:, :, i]
            gaus_img = np.round(gaus_img).astype(np.uint8)

            sg_img[:, :, i] = np.bitwise_or(
                sg_img[:, :, i], gradient_layer(gaus_img))

    return sg_img


def stretch_image(img):
    min_pix = np.min(img)
    max_pix = np.max(img)
    r_img = np.round(np.array(img - min_pix).astype(np.float32)
                     * 255.0 / (max_pix - min_pix))
    return r_img.astype(np.uint8)


#
# stretch_smoothed_gradient -- extract the smoothed gradients at
#   defined values of sigma, then total before rescaling to regular
#   image values.
def stretch_smoothed_gradients(img, sigmas):
    img = np.array(img)
    shape = np.shape(img)
    sg_img = np.zeros(shape).astype(np.uint8)

    for sigma in sigmas:
        for i in range(shape[2]):
            # for each layer in the image

            if sigma:
                # if a non-zero sigma is supplied, use it to smooth the image
                # before collecting the gradients
                gaus_img = gaussian_filter(img[:, :, i], sigma)
                sg_img[:, :, i] += gradient_layer(gaus_img)
            else:
                sg_img[:, :, i] += img[:, :, i]

    return stretch_image(img+sg_img)

#
# fisherize -- use the fisher information theory to pre-segment the
#              image


def fisherize(img):
    shape = np.shape(img)

    colour_depth = 256

    sg_img = np.zeros(shape).astype(np.uint8)

    for i in range(shape[2]):
        # for each layer in the image

        layer = img[:, :, i]
        h, b = np.histogram(layer, bins=colour_depth,
                            range=(0, colour_depth-1))
        p = h / (shape[0] * shape[1])

        w = np.zeros(colour_depth)
        p_w = np.zeros((colour_depth, colour_depth))
        I_a = np.zeros(colour_depth)
        I_b = np.zeros(colour_depth)
        I = np.zeros(colour_depth)

        # calculate the proportion of Cat A vs Cat B
        for t in range(colour_depth):
            w[t] = np.sum(p[0:t])

            # # split into two groups by probability
            # if w[t]:
            #     for i in range(t):
            #         p_w[t, i] = p[i] / w[t]

            # if w[t] is not 1:
            #     for i in range(t, colour_depth):
            #         p_w[t, i] = p[i] / (1 - w[t])

        for t in range(1, colour_depth):
            diff = np.divide((p[1:t] - p[0:t-1])**2, p[0:t-1])
            if w[t]:
                diff[p[0:t-1] == 0] = 0
                I_a[t] = np.sum(diff)

            diff = np.divide(
                (p[t+2:colour_depth] - p[t+1:colour_depth-1])**2,
                p[t+1:colour_depth-1])
            if w[t]:
                diff[p[t+1:colour_depth-1] == 0] = 0
                I_b[t] = np.sum(diff)

            I[t] = I_a[t] + I_b[t]

       # had to implement an offset to keep it from tripping out on the base
        offset = 25
        t_opt = np.argmax(I[offset:-offset]) + offset

        sg_img[:, :, i] = (layer >= t_opt).astype(
            np.uint8) * (colour_depth - 1)

    return sg_img


def mask_image(image, mask):
    return np.bitwise_and(np.array(image).astype(np.uint8), mask)


class ImagePreprocess:
    def __init__(self, pp=None):
        self.pp = pp

    def __fn__(self, img):
        return img

    def __preprocess__(self, img):
        if self.pp is not None:
            img = self.pp.__preprocess__(img)
        return self.__fn__(img)

    def preprocess(self):
        return self.__preprocess__


class ImagePreprocessGradient(ImagePreprocess):
    def __init__(self, sigmas, keep_image=True, pp=None):
        super().__init__(pp)
        self.sigmas = sigmas
        self.keep_image = keep_image

    def __fn__(self, img):
        return smoothed_gradients(img, self.sigmas, self.keep_image)


class ImagePreprocessStretchedGradient(ImagePreprocess):
    def __init__(self, sigmas, pp=None):
        super().__init__(pp)
        self.sigmas = sigmas

    def __fn__(self, img):
        return stretch_smoothed_gradients(img, self.sigmas)


class ImagePreprocessFisherize(ImagePreprocess):
    def __init__(self, pp=None):
        super().__init__(pp)

    def __fn__(self, img):
        return fisherize(img)


class ImagePreprocessMask(ImagePreprocess):
    def __init__(self, mask, pp=None):
        super().__init__(pp)
        self.mask = mask

    def __fn__(self, img):
        return mask_image(img, self.mask)
