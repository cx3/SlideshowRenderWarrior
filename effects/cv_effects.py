import os
from typing import Iterable

import cv2
import numpy as np

from common import utils

image_types = utils.image_types
verify_alpha_channel = utils.verify_alpha_channel


def fit_images(images_list: list, resolution=(1920, 1080)) -> list:

    for i in range(len(images_list)):
        if resolution != utils.get_image_size(images_list[i]):
            images_list[i] = cv2.resize(images_list[i], resolution)

    return images_list


def corners_detect_shi(image: image_types) -> np.ndarray:
    """
    Returns a list of points to corners
    :param image: look at image_types at beginning of module
    :return: list of points as np.ndarray
    """
    # image = verify_alpha_channel(image)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = to_grayscale_effect(verify_alpha_channel(image))
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    return np.int0(corners)


"""
OpenCV based effects. 
All effects:
 * has an image as a first parameter (check image_types at beginning of module)
 * returns numpy.ndarray 
 * has _effect at the end of function name
"""


def invert_effect(image: image_types) -> np.ndarray:
    """
    Negative
    :param image: one of image_type
    :return: OpenCV image negative
    """
    image = utils.load_image(image, np.ndarray)
    return cv2.bitwise_not(image)


def invert_channel_effect(image: image_types, channels='R') -> np.ndarray:
    """
    Make negative on selected channels
    :param image: one of image_type
    :param channels: letters of channels to make negative
    :return: OpenCV image
    """
    b, g, r, a = cv2.split(verify_alpha_channel(image))
    ch = dict(b=b, g=g, r=r, a=a)

    for next_channel in channels.lower():
        ch[next_channel] = invert_effect(ch[next_channel])

    return cv2.merge(list(ch.values()))


def apply_color_overlay_effect(
        image: image_types, intensity=0.2, red=112, green=66, blue=20, result_mode='BGRA') -> np.ndarray:
    """
    Allows apply sepia effect or similar
    :param image: one of image_type
    :param intensity: value 0.0 - 1.0
    :param red: value of red channel
    :param green: value of green channel
    :param blue: value of blue channel
    :param result_mode: str name of color space or cv2.COLOR_???
    :return: OpenCV image
    """
    image = verify_alpha_channel(image)
    height, width, c = image.shape
    colors = (blue, green, red, 1)
    # noinspection PyTypeChecker
    overlay = np.full((height, width, 4), colors, dtype='uint8')
    cv2.addWeighted(overlay, intensity, image, 1.0, 0, image)
    if result_mode != 'BGRA':
        color_space = getattr(cv2, 'COLOR_BGRA2' + result_mode)
        return cv2.cvtColor(image, color_space)  # cv2.COLOR_BGRA2BGR
    return image


def threshold_effect(image: image_types, threshold=2, max_value=255) -> np.ndarray:
    """
    Segmentation effect for edges detection or other enhancements
    :param image: one of image_type
    :param threshold: int value 1 - 10
    :param max_value: int value 1 - 255
    :return: OpenCV image
    """
    gray = cv2.cvtColor(verify_alpha_channel(image), cv2.COLOR_BGRA2GRAY)
    return cv2.threshold(gray, thresh=threshold, maxval=max_value, type=cv2.THRESH_OTSU)[1]


def border_reflect_effect(
        image: image_types, top=30, bottom=30, left=30, right=30, dest_size=(1920, 1080)) -> np.ndarray:
    """
    Adds border frames to each side of image
    :param image: one of image_type
    :param top: pixels
    :param bottom: pixels
    :param left: pixels
    :param right: pixels
    :param dest_size: width, height of video frame resolution
    :return: OpenCV image
    """
    image = utils.load_image(image, np.ndarray)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    x, y = image.shape[:2]
    return cut_rect(image, (0, 0, x, y), dest_size)


def blend_images_effect(
            image1: image_types,
            image2: image_types,
            image1_alpha=0.7,
            image2_alpha=0.3,
            zeta=0,
            resolution=(1920, 1080),
        ) -> np.ndarray:
    """
    Alpha blending images
    :param image1: first image (one of image_type)
    :param image2: second image (one of image_type)
    :param image1_alpha: float value 0.0 - 1.0
    :param image2_alpha: float value 0.0 - 1.0
    :param zeta: additional int value
    :param resolution: both images resolution should match to it
    :return: OpenCV image
    """

    s1, s2 = utils.get_image_size(image1), utils.get_image_size(image2)

    if s1 != resolution:
        image1 = cv2.resize(image1, resolution)
    if s1 != resolution:
        image1 = cv2.resize(image2, resolution)

    return cv2.addWeighted(
        utils.load_image(image1, np.ndarray),
        image1_alpha,
        utils.load_image(image2, np.ndarray),
        image2_alpha,
        zeta
    )


def to_grayscale_effect(image: image_types) -> np.ndarray:
    image = utils.load_image(image, np.ndarray)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_black_white_effect(image: image_types) -> np.ndarray:
    image = to_grayscale_effect(image)
    return cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]


def contrast_stretch_basic_effect(image: image_types) -> np.ndarray:
    image = verify_alpha_channel(image)
    # original = image.copy()
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    return cv2.LUT(image, table)


def contrast_stretch_slower_effect(image: image_types, mode=0) -> np.ndarray:
    # https://stackoverflow.com/questions/42257173/contrast-stretching-in-python-opencv
    # normalize float versions
    norm_img1 = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_img2 = cv2.normalize(image, None, alpha=0, beta=1.2, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # scale to uint8
    norm_img1 = (255 * norm_img1).astype(np.uint8)
    norm_img2 = np.clip(norm_img2, 0, 1)
    norm_img2 = (255 * norm_img2).astype(np.uint8)

    if mode % 2 == 0:
        return norm_img1
    return norm_img2


def dilate_mask_effect(image: image_types, iterations=1) -> np.ndarray:
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/
    # py_morphological_ops.html

    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/
    # py_morphological_ops.html#structuring-element

    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(to_black_white_effect(image), kernel, iterations=iterations)


def canny_edges_effect(image: image_types, threshold1=100, threshold2=100) -> np.ndarray:
    return cv2.Canny(verify_alpha_channel(image), threshold1, threshold2)


def find_contour_effect(image: image_types, dot_size=10) -> np.ndarray:
    bw = to_black_white_effect(utils.load_image(image, np.ndarray))
    height, width = bw.shape[:2]

    blank = np.zeros((height, width, 3), np.uint8)
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return cv2.drawContours(blank, contours, -1, (0, 255, 0), dot_size)


def gradient_mask_effect(image: image_types, mode=1) -> np.ndarray:
    image = verify_alpha_channel(image)

    if mode < 0:
        mode = 0
    '''if mode > 2:
        mode = 2'''

    if mode == 0:
        return cv2.Laplacian(image, cv2.CV_64F)
    if mode == 1:
        return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    if mode == 2:
        return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    if mode == 3:  # tastes good
        return cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=5)
    if mode == 4:
        return np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5))


def corners_detect_effect(image: image_types) -> np.ndarray:

    image = cv2.cvtColor(verify_alpha_channel(image), cv2.COLOR_BGRA2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    image[dst > 0.01 * dst.max()] = [0, 0, 255]

    return image


def cut_irregular_shape(image: image_types, points: np.ndarray or Iterable) -> np.ndarray:
    # https://stackoverflow.com/questions/15341538/numpy-opencv-2-how-do-i-crop-non-rectangular-region/15343106
    image = verify_alpha_channel(image)

    # mask defaulting to black for 3-channel and transparent for 4-channel
    # (of course replace corners with yours)
    mask = np.zeros(image.shape, dtype=np.uint8)
    '''roi_corners = np.array([[(10, 10), (300, 300), (10, 300)]], dtype=np.int32)'''

    if isinstance(points, (list, tuple)):
        roi_corners = np.array([points], dtype=np.int32)
    elif isinstance(points, np.ndarray):
        roi_corners = points
    else:
        raise TypeError(f'cv_effects.cut_irregular_shape param points incorrect type {type(points)}')

    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex

    # apply the mask
    return cv2.bitwise_and(image, mask)


def equalize_histogram_effect(image: image_types) -> np.ndarray:
    return cv2.equalizeHist(to_grayscale_effect(image))


def auto_contrast_effect(image: image_types, alpha1=0.1, alpha2=0.9, zeta=0) -> np.ndarray:
    image = verify_alpha_channel(image)
    gray = to_grayscale_effect(image)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel1)
    div = np.float32(gray) / close
    res = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))

    image = cv2.cvtColor(res, cv2.COLOR_BGRA2RGB)
    res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)

    return blend_images_effect(image, res, alpha1, alpha2, zeta)


def inpaint_effect(image: image_types, mask_image: image_types, strength=3) -> np.ndarray:
    return cv2.inpaint(image, mask_image, strength, cv2.INPAINT_TELEA)


def cut_rect(image: image_types, src_box: list or tuple, size: list or tuple = (1920, 1080)) -> np.ndarray:

    if len(src_box) != 4:
        raise AttributeError('cv_effects.cut_rect param src_box must be tuple of 4 (x1, y1, x2, y2)')
    if len(size) != 2:
        raise AttributeError('cv_effects.cut_rect param size must be tuple of 2 (width, height)')

    x1, y1, x2, y2 = src_box
    w, h = abs(x2 - x1), abs(y2 - y1)

    image = utils.convert_image_type(image, np.ndarray)
    crop_img = image[y1:y1 + h, x1:x1 + w]

    return cv2.resize(crop_img, dsize=size)


def scale_images_dir(src_images_dir_path: str, dest_images_dir_path: str, dest_size=(1920, 1080)) -> int:
    if not os.path.isdir(src_images_dir_path):
        raise FileNotFoundError(f'src_images_dir_path="{src_images_dir_path}" is not a directory')
    if os.path.isfile(dest_images_dir_path):
        raise OSError(f'dest_images_dir_path="{dest_images_dir_path}" is file, not directory')

    result = 0
    if src_images_dir_path[-1] not in ['\\', '/']:
        src_images_dir_path += os.path.sep
    if dest_images_dir_path[-1] not in ['\\', '/']:
        dest_images_dir_path += os.path.sep

    for next_file in os.listdir(src_images_dir_path):
        if next_file.split('.')[-1].lower() in ['jpg', 'png', 'bmp']:
            img = utils.load_image(src_images_dir_path + next_file, np.ndarray)
            if utils.get_image_size(img) == dest_size:
                utils.save_image(img, dest_images_dir_path)
            else:
                utils.save_image(img, cv2.resize(img, dest_size))
            result += 1
    return result


def resize_image(image: image_types, dest_image_type: image_types or None = None, dest_resolution=(1920, 1080)):
    if not isinstance(dest_resolution, (list, tuple)):
        if not len(dest_resolution) == 2:
            raise ValueError('dest_resolution must be list/tuple of len 2. i.e. dest_resolution(720, 576)')
    resized = cv2.resize(utils.load_image(image, np.ndarray), dest_resolution)
    if dest_image_type is None:
        return resized
    return utils.convert_image_type(resized, dest_resolution)
