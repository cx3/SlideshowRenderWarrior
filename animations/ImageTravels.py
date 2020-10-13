import sys
from typing import List
from inspect import isclass, getmembers
from random import shuffle

import numpy as np
from common import utils
import effects.cv_effects as cv_effects

from animations.Transitions_v2 import image_types, NextFrame


class NextFrameMulti(NextFrame):
    def next_frame(
            self,
            images: List[image_types],
            frames_count: int,
            frame_resolution: list or tuple = (1920, 1080),
            **params) -> dict:
        ...


class ImageTravel(NextFrameMulti):
    def __init__(self,
                 images: List[image_types] or None = None,
                 frames_count: int or None = 0,
                 frame_resolution: list or tuple = (1920, 1080)):

        self.images = images
        self.frames_count = frames_count
        self.frame_resolution = frame_resolution
        self.crop_params = utils.CircleList([], 0)

        if images is not None and frames_count > 0:
            self.set_params(images, frames_count, frame_resolution)

    def set_params(self, images: List[image_types], frames_count: int, frame_resolution: list or tuple = (1920, 1080)):
        self.images = images
        self.frames_count = frames_count
        self.frame_resolution = frame_resolution

    def has_more(self) -> bool:
        return self.crop_params.has_more()


class ExampleTravel(ImageTravel):

    def set_params(self, images: List[image_types], frames_count: int, frame_resolution: list or tuple = (1920, 1080)):
        if len(frame_resolution) != 2:
            raise AttributeError('ImageTravel param frame_resolution must be tuple of 2 (width, height)')

        res_width, res_height = frame_resolution

        if len(images) > 0:

            # image = utils.load_image(image, np.ndarray)
            for i in range(len(images)):
                images[i] = utils.load_image(images[i], np.ndarray)

                frame_height, frame_width = images[0].shape[:2]

                if res_width > frame_width or res_height > frame_height:
                    raise AttributeError('ImageTravel image size is lower than frame_resolution')

        self.images = images
        self.frames_count = frames_count
        self.frame_resolution = frame_resolution
        crop_params = []

        frame_width_step = int(res_width // frames_count)
        frame_height_step = int(res_height // frames_count)

        for i in range(1, frames_count + 1):
            crop_params.append((0, 0, frame_width_step * i, frame_height_step * i))

        self.crop_params = utils.CircleList(crop_params, self.frames_count)

    def next_frame(self, **params) -> dict:
        n = self.crop_params.next()
        print(f'_ExampleTravel::next_frame n={n}')

        images = params.get('images', self.images)

        for index in range(len(images)):
            images[index] = cv_effects.cut_rect(images[index], n, self.frame_resolution)

        params['images'] = images
        return params

    def has_more(self):
        return self.crop_params.has_more()

    def get_circle_list(self) -> utils.CircleList:
        # utils.Callback(fun_ptr=self.next_frame)
        return utils.CircleList([self.next_frame], self.frames_count)


# BELOW CODE MUST BE TOTALY REFACTORED

'''
class LeftTopZoomFull(ImageTravel):
    def set_params(
            self,
            image: image_types,
            frames_count: int,
            frame_resolution: list or tuple = (1920, 1080)):

        if len(frame_resolution) != 2:
            raise AttributeError('ImageTravel param frame_resolution must be tuple of 2 (width, height)')

        res_width, res_height = frame_resolution

        image = utils.load_image(image, np.ndarray)
        frame_height, frame_width = image.shape[:2]

        if res_width > frame_width or res_height > frame_height:
            raise AttributeError('ImageTravel image size is lower than frame_resolution')

        self.image = image
        self.frame_resolution = frame_resolution
        crop_params = []

        frame_width_step = int(res_width // frames_count)
        frame_height_step = int(res_height // frames_count)

        for i in range(1, frames_count + 1):
            crop_params.append((0, 0, frame_width_step * i, frame_height_step * i))

        self.crop_params = utils.CircleList(crop_params, frames_count)

    def next_frame(self, **params) -> dict:
        image = self.image if 'image' not in params else params['image']
        print(f'@@ LeftTopZoomFull.next_frame {params.keys()}')
        n = self.crop_params.next()
        print(']$$$]', n)
        im1 = cv_effects.cut_rect(image, n, self.frame_resolution)
        params['im1'] = params['im2'] = im1
        params['to_save'] = im1
        return params

    def has_more(self):
        return self.crop_params.has_more()


class LeftBottomZoomIn(ImageTravel):
    def set_params(
            self,
            images: List[image_types],
            frames_count: int = 25,
            frame_resolution: list or tuple = (1920, 1080)):

        if len(frame_resolution) != 2:
            raise AttributeError('ImageTravel param frame_resolution must be tuple of 2 (width, height)')

        res_width, res_height = frame_resolution

        image = utils.verify_alpha_channel(utils.load_image(image, np.ndarray))
        frame_height, frame_width = image.shape[:2]

        if res_width > frame_width or res_height > frame_height:
            raise AttributeError('ImageTravel image size is lower than frame_resolution')

        self.images = images
        self.frame_resolution = frame_resolution
        crop_params = []

        base_width, base_height = int(0.75 * res_width), int(0.75 * res_height)

        frame_width_step = int((res_width - base_width) / frames_count)
        frame_height_step = int((res_height - base_height) / frames_count)

        for i in range(1, frames_count + 1):
            crop_params.append((0, 0, base_width + frame_width_step * i, base_height + frame_height_step * i))

        self.crop_params = utils.CircleList(crop_params, frames_count)

    def next_frame(self, **params) -> dict:
        image = self.image if 'image' not in params else params['image']
        # print(f'@@ travel.next_frame {params}')
        cut_region = self.crop_params.next()
        print(f'LeftBottomZoomFull: {cut_region}, params="{params.keys()}"')
        im1 = cv_effects.cut_rect(image, cut_region, self.frame_resolution)
        params['im1'] = params['im2'] = im1
        params['to_save'] = im1
        return params

    def has_more(self):
        return self.crop_params.has_more()


class LeftBottomZoomOut(ImageTravel):
    def set_params(
            self,
            image: image_types,
            frames_count: int = 25,
            frame_resolution: list or tuple = (1920, 1080)):

        if len(frame_resolution) != 2:
            raise AttributeError('ImageTravel param frame_resolution must be tuple of 2 (width, height)')

        res_width, res_height = frame_resolution

        image = utils.load_image(image, np.ndarray)
        frame_height, frame_width = image.shape[:2]

        if res_width > frame_width or res_height > frame_height:
            raise AttributeError('ImageTravel image size is lower than frame_resolution')

        self.image = image
        self.frame_resolution = frame_resolution
        crop_params = []

        base_width, base_height = int(0.75 * res_width), int(0.75 * res_height)

        frame_width_step = int((res_width - base_width) / frames_count)
        frame_height_step = int((res_height - base_height) / frames_count)

        for i in range(1, frames_count + 1):
            crop_params.append((
                0, 0,  # x1, y1
                res_width - frame_width_step * i,  # x2
                res_height - frame_height_step * i)
            )

        self.crop_params = utils.CircleList(crop_params, frames_count)

    def next_frame(self, **params) -> dict:
        image = self.image if 'image' not in params else params['image']
        cut_region = self.crop_params.next()
        print(f'LeftBottomZoomOut: {cut_region}, params="{params.keys()}"')
        im1 = cv_effects.cut_rect(image, cut_region, self.frame_resolution)
        params['im1'] = params['im2'] = im1
        params['to_save'] = im1
        return params


class RightBottomZoomIn(ImageTravel):
    def set_params(
            self,
            image: image_types,
            frames_count: int = 25,
            frame_resolution: list or tuple = (1920, 1080)):

        if len(frame_resolution) != 2:
            raise AttributeError('ImageTravel param frame_resolution must be tuple of 2 (width, height)')

        res_width, res_height = frame_resolution

        image = utils.load_image(image, np.ndarray)
        frame_height, frame_width = image.shape[:2]

        if res_width > frame_width or res_height > frame_height:
            raise AttributeError('ImageTravel image size is lower than frame_resolution')

        self.image = image
        self.frame_resolution = frame_resolution
        crop_params = []

        base_width, base_height = int(0.75 * res_width), int(0.75 * res_height)

        frame_width_step = int((res_width - base_width) / frames_count)
        frame_height_step = int((res_height - base_height) / frames_count)

        for i in range(1, frames_count + 1):
            crop_params.append((frame_width_step * i, frame_height_step * i, res_width, res_height))

        self.crop_params = utils.CircleList(crop_params, frames_count)

    def next_frame(self, **params) -> dict:
        image = self.image if 'image' not in params else params['image']
        # print(f'@@ travel.next_frame {params}')
        cut_region = self.crop_params.next()
        print(f'RightBottomZoomIn: {cut_region}, params="{params.keys()}"')
        im1 = cv_effects.cut_rect(image, cut_region, self.frame_resolution)
        params['im1'] = params['im2'] = im1
        params['to_save'] = im1
        return params


class RightBottomZoomOut(ImageTravel):
    def set_params(
            self,
            image: image_types,
            frames_count: int = 25,
            frame_resolution: list or tuple = (1920, 1080)):

        if len(frame_resolution) != 2:
            raise AttributeError('ImageTravel param frame_resolution must be tuple of 2 (width, height)')

        res_width, res_height = frame_resolution

        image = utils.load_image(image, np.ndarray)
        frame_height, frame_width = image.shape[:2]

        if res_width > frame_width or res_height > frame_height:
            raise AttributeError('ImageTravel image size is lower than frame_resolution')

        self.image = image
        self.frame_resolution = frame_resolution
        crop_params = []

        base_width, base_height = int(0.75 * res_width), int(0.75 * res_height)

        frame_width_step = int((res_width - base_width) / frames_count)
        frame_height_step = int((res_height - base_height) / frames_count)

        max_w, max_h = frame_width_step * frames_count, frame_height_step * frames_count

        for i in range(1, frames_count + 1):
            crop_params.append((max_w - frame_width_step * i, max_h - frame_height_step * i, res_width, res_height))

        self.crop_params = utils.CircleList(crop_params, frames_count)

    def next_frame(self, **params) -> dict:
        image = self.image if 'image' not in params else params['image']
        # print(f'@@ travel.next_frame {params}')
        cut_region = self.crop_params.next()
        print(f'LeftBottomZoomFull: {cut_region}, params="{params.keys()}"')
        im1 = cv_effects.cut_rect(image, cut_region, self.frame_resolution)
        params['im1'] = params['im2'] = im1
        params['to_save'] = im1
        return params'''


def get_random_travellers_circlelist(travels_count=-1, shuffle_list=True, debug=False) -> utils.CircleList:
    if debug:
        return utils.CircleList(
            list_or_tuple=[ExampleTravel] * 100,
        )

    result = []

    for item_name, item_obj in getmembers(sys.modules[__name__]):
        if isclass(item_obj):
            if 'Zoom' in item_name:
                result.append(item_obj)

    if shuffle_list:
        shuffle(result)

    return utils.CircleList(
        list_or_tuple=result,
        max_iterations=travels_count
    )
