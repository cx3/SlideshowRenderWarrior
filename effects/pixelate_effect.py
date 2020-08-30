# This script will pixelate most jpg and png images
# It will both show you the result and save it

import numpy as np
import PIL.Image as Image


def _load_img(filename):
    # boilerplate code to open an image and make it editable
    img = Image.open(filename)
    data = np.array(img)
    return data


def _all_square_pixels(row, col, square_h, square_w):
    # Every pixel for a single "square" (superpixel)
    # Note that different squares might have different dimensions in order to
    # not have extra pixels at the edge not in a square. Hence: int(round())
    for y in range(int(round(row*square_h)), int(round((row+1)*square_h))):
        for x in range(int(round(col*square_w)), int(round((col+1)*square_w))):
            yield y, x


def _make_one_square(img, row, col, square_h, square_w):
    # Sets all the pixels in img for the square given by (row, col) to that
    # square's average color
    pixels = []

    # get all pixels
    for y, x in _all_square_pixels(row, col, square_h, square_w):
        pixels.append(img[y][x])

    # get the average color
    av_r = 0
    av_g = 0
    av_b = 0
    for r, g, b in pixels:
        av_r += r
        av_g += g
        av_b += b
    av_r /= len(pixels)
    av_g /= len(pixels)
    av_b /= len(pixels)

    # set all pixels to that average color
    for y, x in _all_square_pixels(row, col, square_h, square_w):
        img[y][x] = (av_r, av_g, av_b)


def pixelate(image: str or Image.Image, num_cols: int = 256) -> Image.Image:

    if isinstance(image, str):
        image = _load_img(image)

    # Figure out the dimensions of each square
    # We want:
    # 1. Square width and height should be about the same
    # 2. No leftover pixels at the edges
    # This means that some squares might have one more or one less pixel
    # depending on rounding

    square_w = float(image.shape[1]) / num_cols
    num_rows = int(round(image.shape[0] / square_w))
    square_h = float(image.shape[0]) / num_rows

    # overwrite each square with the average color, one by one
    for row in range(num_rows):
        for col in range(num_cols):
            _make_one_square(image, row, col, square_h, square_w)

    return Image.fromarray(image)
