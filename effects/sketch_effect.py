# https://github.com/sundar19/Pencil-Sketch-python/blob/master/pencil_sketch.py
# https://www.freecodecamp.org/news/sketchify-turn-any-image-into-a-pencil-sketch-with-10-lines-of-code-cf67fa4f68ce/
# https://www.askaswiss.com/2016/01/how-to-create-pencil-sketch-opencv-python.html

import cv2
import numpy as np
import scipy.ndimage
from PIL import Image


def _dodge(front, back):
    result = front*255/(255-back)
    result[result > 255] = 255
    result[back == 255] = 255
    return result.astype('uint8')


def dodge_v2(image, mask):
    return cv2.divide(image, 255 - mask, scale=256)


def _grayscale(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def sketch_v1(image: Image.Image or str, sigma=10) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image)
    image = np.array(image)

    g = _grayscale(image)
    i = 255 - g

    b = scipy.ndimage.filters.gaussian_filter(i, sigma=sigma)
    r = _dodge(b, g)
    return Image.fromarray(r)


def sketch_v2(image: Image.Image or str) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image)
    jc = np.array(image)

    scale_percent = 0.60

    width = int(jc.shape[1] * scale_percent)
    height = int(jc.shape[0] * scale_percent)

    dim = (width, height)
    resized = cv2.resize(jc, dim, interpolation=cv2.INTER_AREA)

    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(resized, -1, kernel_sharpening)

    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    gauss = cv2.GaussianBlur(inv, ksize=(15, 15), sigmaX=0, sigmaY=0)

    pencil_jc = dodge_v2(gray, gauss)

    '''cv2.imshow('resized', resized)
    cv2.imshow('sharp', sharpened)
    cv2.imshow('gray', gray)
    cv2.imshow('inv', inv)
    cv2.imshow('gauss', gauss)
    cv2.imshow('pencil sketch', pencil_jc)'''

    return Image.fromarray(pencil_jc)


def sketch_v3(image: Image.Image or str, s_sigma=10, r_sigma=0.1) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image)

    image = np.array(image)
    # sketch_gray, sketch_color = cv2.pencilSketch(image, sigma_s=10, sigma_r=0.07, shade_factor=0.05)
    stylize = cv2.stylization(image, sigma_s=s_sigma, sigma_r=r_sigma)
    return Image.fromarray(stylize)


def sketch_v4(image: Image.Image or str, depth=10.0) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image)

    a = np.asarray(image).astype('float')
    # Reconstruct the image using the gradient value and virtual depth value between pixels, and simulate the distance
    # between human vision according to the grayscale change
    # depth = 10.  # 0-100 The preset depth is 10, the value range is 0-100

    if depth < 0:
        depth = 0.1
    if depth > 100.0:
        depth = 100.0

    grad = np.gradient(a)  # Take the gradient value of the image gray
    grad_x, grad_y = grad[:2]  # Extract gradient values ​​in the x and y directions
    grad_x = grad_x * depth / 100.
    grad_y = grad_y * depth / 100.  # Adjust the gradient values ​​in the x and y directions according to the depth

    # Construct a three-dimensional normalized unit coordinate system of x and y
    # gradients
    gradient = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.)

    uni_x = grad_x / gradient
    uni_y = grad_y / gradient
    uni_z = 1. / gradient

    # np.cos(vec_el) is the projection length of the unit ray on the ground plane
    vec_el = np.pi / 2.2  # Overhead angle of light source, radian value
    vec_az = np.pi / 4.  # Azimuth angle of light source, radian value
    # dx, dy, dz is the degree of influence of the light source on the x/y/z directions
    dx = np.cos(vec_el) * np.cos(vec_az)  # Influence of light source on X axis
    dy = np.cos(vec_el) * np.sin(vec_az)  # Influence of light source on y-axis
    dz = np.sin(vec_el)  # Influence of light source on z axis

    # Gradient interacts with the light source to convert the gradient to grayscale
    b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)
    b = b.clip(0, 255)  # In order to avoid data cross-border, crop the generated gray value to the range of 0-255

    return Image.fromarray(b.astype('uint8'))  # Reconstructed image
