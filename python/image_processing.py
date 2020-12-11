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


def stretch_smoothed_gradients(img, sigmas):
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


def mask_image(image, mask):
    return np.bitwise_and(np.array(image).astype(np.uint8), mask)


class ImagePreprocess:
    def __init__(self, pp=None):
        self.pp = pp

    def __fn__(self, img):
        return img

    def __preprocess__(self, img):
        if self.pp is not None:
            img = self.pp(img)
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


class ImagePreprocessMask(ImagePreprocess):
    def __init__(self, mask, pp=None):
        super().__init__(pp)
        self.mask = mask

    def __fn__(self, img):
        return mask_image(img, self.mask)
