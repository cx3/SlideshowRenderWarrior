# Third party imports
import numpy as np
from PIL.Image import Image as PillowImage

# Project imports
from common import utils

# -+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-


image_types = utils.image_types


# noinspection PyUnusedLocal
def no_effect_fun(image: image_types, **_kwargs) -> PillowImage or np.ndarray:
    return image


def create_empty_callbacks_cl(frames_count=25) -> utils.CircleList:
    return utils.CircleList(list_or_tuple=[no_effect_fun] * frames_count, max_iterations=frames_count)


# -+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-


class NextFrame:
    def init_machine(self, **params):
        self.__dict__['init_machine'] = params

    def next_frame(self, frame_index: int, im1: image_types, im2: image_types, **params) -> dict:
        result = locals()
        result['to_save'] = im1
        return result

    def has_more(self) -> bool:
        return False


class RenderFrames:
    def render_frames(self, **params):
        ...


class AbstractTransition(NextFrame, RenderFrames):

    def __init__(self, **kwargs):
        self.inited_with = {**kwargs}

    def init_machine(self):
        pass

    def next_frame(self, frame_index: int, im1: image_types, im2: image_types, **params) -> dict:
        print('Abstract!!!')
        input('Enter...')
        result = locals()
        result['to_save'] = im1
        return result

    def render_frames(self):
        pass

    @staticmethod
    def effects_to_set() -> tuple:
        return 'preprocess_first', 'preprocess_second', 'postprocess_first', 'postprocess_second'


directions = [
    'lu', 'ru',  # left up, right up
    'lb', 'rb',  # left bottom, right bottom
]
