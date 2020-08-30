# source https://raw.githubusercontent.com/lutianming/cartoonizer/master/cartoonizer.py

# Python's imports
from collections import defaultdict

# Project's imports
import utils

# third party
import numpy as np
from scipy import stats
from PIL import Image
import cv2


image_types = utils.image_types
PillowImage = Image.Image


def cartoonize_v1(image: image_types) -> PillowImage:
    """
    convert image into cartoon-like image

    image: input PIL image
    """

    image = utils.load_image(image)

    output = np.array(image)
    x, y, c = output.shape
    # hists = []
    for i in range(c):
        output[:, :, i] = cv2.bilateralFilter(output[:, :, i], 5, 50, 50)
        # hist, _ = np.histogram(output[:, :, i], bins=np.arange(256+1))
        # hists.append(hist)
    edge = cv2.Canny(output, 100, 200)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)
    hists = []
    # H
    hist, _ = np.histogram(output[:, :, 0], bins=np.arange(180+1))
    hists.append(hist)
    # S
    hist, _ = np.histogram(output[:, :, 1], bins=np.arange(256+1))
    hists.append(hist)
    # V
    hist, _ = np.histogram(output[:, :, 2], bins=np.arange(256+1))
    hists.append(hist)

    centroids = []
    for h in hists:
        centroids.append(k_histogram(h))
    # print("centroids: {0}".format(centroids))

    output = output.reshape((-1, c))
    for i in range(c):
        channel = output[:, i]
        index = np.argmin(np.abs(channel[:, np.newaxis] - centroids[i]), axis=1)
        output[:, i] = centroids[i][index]
    output = output.reshape((x, y, c))
    output = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)

    contours, _ = cv2.findContours(edge,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    # for i in range(len(contours)):
    #     tmp = contours[i]
    #     contours[i] = cv2.approxPolyDP(tmp, 2, False)
    cv2.drawContours(output, contours, -1, 0, thickness=1)
    return output


def cartoonize_v2(
            image: image_types,
            median_blur_size=5,
            max_value=255,
            block_size=9,
            c_size=9,
            sigma_color=300,
            sigma_space=300) -> PillowImage:

    image = cv2.cvtColor(utils.load_image(image, np.ndarray), cv2.COLOR_RGB2BGR)
    # 1) Edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, median_blur_size)
    edges = cv2.adaptiveThreshold(gray, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c_size)

    # 2) Color
    color = cv2.bilateralFilter(image, 9, sigma_color, sigma_space)
    # 3) Cartoon
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return Image.fromarray(cartoon)


def _update_entroids(centroids, hist):
    """
    update centroids until they don't change
    """
    while True:
        groups = defaultdict(list)
        # assign pixel values
        for i in range(len(hist)):
            if hist[i] == 0:
                continue
            d = np.abs(centroids - i)
            index = np.argmin(d)
            groups[index].append(i)

        new_entroids = np.array(centroids)
        for i, indice in groups.items():
            if np.sum(hist[indice]) == 0:
                continue
            new_entroids[i] = int(np.sum(indice*hist[indice])/np.sum(hist[indice]))
        if np.sum(new_entroids - centroids) == 0:
            break
        centroids = new_entroids
    return centroids, groups


def k_histogram(hist):
    """
    choose the best K for k-means and get the centroids
    """
    alpha = 0.001  # p-value threshold for normaltest
    n = 80  # minimun group size for normaltest
    c = np.array([128])

    while True:
        c, groups = _update_entroids(c, hist)

        # start increase K if possible
        new_c = set()     # use set to avoid same value when seperating centroid
        for i, indice in groups.items():
            # if there are not enough values in the group, do not seperate
            if len(indice) < n:
                new_c.add(c[i])
                continue

            # judge whether we should seperate the centroid
            # by testing if the values of the group is under a
            # normal distribution
            z, pval = stats.normaltest(hist[indice])
            if pval < alpha:
                # not a normal dist, seperate
                left = 0 if i == 0 else c[i-1]
                right = len(hist)-1 if i == len(c)-1 else c[i+1]
                delta = right-left
                if delta >= 3:
                    c1 = (c[i]+left)/2
                    c2 = (c[i]+right)/2
                    new_c.add(c1)
                    new_c.add(c2)
                else:
                    # though it is not a normal dist, we have no
                    # extra space to seperate
                    new_c.add(c[i])
            else:
                # normal dist, no need to seperate
                new_c.add(c[i])
        if len(new_c) == len(c):
            break
        else:
            c = np.array(sorted(new_c))
    return c
