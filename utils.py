import inspect
from collections import Callable
from typing import Any, Type, List, Tuple

import PIL.Image
from PIL.Image import Image as PillowImage
import cv2
import numpy as np


image_types = str or PillowImage or np.ndarray


class Callback(object):
    """
    Use object of this class to handle function pointer for invoking in further time.
    """

    def __init__(self, fun_ptr: Callable, needs_params: tuple = tuple(), *args, **kwargs):
        """

        :param fun_ptr:
            pointer to function which should be handled
        :param needs_params: list/tuple of str with function's parameter names that should be passed to avoid TypeError
            exception of with missing arguments
        :param args:
            args to be passed to function when calling it
        :param kwargs:
            keyword arguments that should be passed to function for reaching expected returned value
        """
        self.params = locals()
        self.params['fun_name'] = fun_ptr.__name__
        self.check_on_call = False  # unused by now, but in near future... ;>

    def needs_param(self, param_name: str) -> bool:
        """
        :param param_name: Tells if pointer to handled function to correct work needs argument
        :return: bool
        """
        return param_name in self.params['needs_params']

    def assign_later(self, args_names: list or tuple):
        """
        By now this method is not used
        :param args_names:
        :return:
        """
        if not isinstance(args_names, (list, tuple)):
            self.check_on_call: list = [args_names]
        self.check_on_call.extend(args_names)
        return len(self.check_on_call)

    def __call__(self, *args, **kwargs) -> Any or KeyError:
        """
        Invokes handled function pointer with passed args and kwargs and returns its returned value
        :param args: passed to arguments of handled function, merged with :param args: on Callback.__init__
        :param kwargs: passed to arguments of handled function merged with :param kwargs: on Callback.__init__
        :return: value returned by invoked handled function
        """

        if isinstance(self.check_on_call, (list, tuple)):
            for name in self.check_on_call:
                if name not in kwargs:
                    raise KeyError(f'fill params: {self.check_on_call}')
        if args is None and kwargs is None:
            print(f'Callback.__call__: name={self.params["fun_name"]},  args={self.params["args"]}  kwargs={kwargs}')
            return self.params['fun_ptr'](*self.params['args'], **self.params['kwargs'])
        _args = args + self.params['args']
        _kwargs = {**self.params['kwargs'], **kwargs}
        # return self.params['fun_ptr'](*self.params['args']*args, **kwargs)
        print(f'Callback.__call__: name={self.params["fun_name"]}, args={_args}  kwargs={_kwargs}')
        return self.params['fun_ptr'](*_args, **_kwargs)

    def add_one_arg(self, name, value):
        self.params['kwargs'][name] = value
        return self.__dict__

    def add_more_args(self, **kwargs):
        self.params['kwargs'] = {**self.params['kwargs'], **kwargs}
        return self.__dict__

    def run(self) -> Any or KeyError:
        return self.__call__()

    def run_no_args(self) -> Any:
        return self.params['fun_ptr']()

    def __str__(self) -> str:
        return '<utils.Callback.self:' + str(self.__dict__) + '>;\n'

    def __repr__(self):
        return self.__str__()


class CircleList:
    """
    An utils for iterating over list as if it was in shape of circle, when reaching last element, next element again is
    going to be with index of zero.
    """
    def __init__(self, list_or_tuple: list or tuple, max_iterations=-1):
        """
        Inits object.
        :param list_or_tuple: list/tuple with elements to work infinitely
        :param max_iterations: if set to -1, works infinitely, if set to positive int value, method has_more tells when
            max_iterations is reached
        """
        if not isinstance(list_or_tuple, (list, tuple)):
            list_or_tuple = [list_or_tuple]
        self.items = list_or_tuple
        self.counter = -1
        self.position = -1
        self.first_round = True
        self.max_iterations = max_iterations
        self.infinity_loop = True if max_iterations <= 0 else False

        self.to_str = "<CircleList>items:" + str(self.items) + ', max_iterations:' + str(self.max_iterations) + \
                      '/<CircleList>'

    def __len__(self) -> int:
        return len(self.items)

    def __str__(self):
        return self.to_str

    def get(self, index: int):
        if index < 0:
            if index < - len(self.items):
                index *= - 1

        if index > self.max_iterations:
            rest = index % len(self.items)
            return self.items[rest]
        return self.items[index]

    def is_finity(self) -> bool:
        return self.infinity_loop

    def has_more(self) -> bool:
        return True if self.infinity_loop else self.counter >= self.max_iterations - 1

    def get_position(self) -> int:
        return self.position

    def get_counter(self) -> int:
        return self.counter

    def next(self) -> Any:
        self.position += 1
        self.counter += 1
        if self.position > len(self.items) - 1:
            self.first_round = False
            self.position = 0
        print('$$$$ self.items[self.position]', self.position, len(self.items))
        return self.items[self.position]

    def current(self) -> Any:
        return self.items[self.position]

    def decrease_position(self):
        new_position = self.position - 1
        if new_position < 0:
            new_position = len(self.items) - 1
        self.position = new_position

    def to_list(self, max_len: int = None, as_tuple=False) -> list or tuple:
        if max_len is None:
            return self.items if not as_tuple else tuple(self.items)
        size = len(self.items)
        if max_len > size:
            result = self.items * ((max_len // size) + 1)
        else:
            result = self.items
        return result[:max_len]


def _run_predicates(image_inst, predicates: list or tuple or CircleList):
    if isinstance(predicates, CircleList):
        predicates = predicates.to_list()
    elif not isinstance(predicates, (list, tuple)):
        predicates = [predicates]
    for p in predicates:
        if callable(p):
            print(f'_run_predicates calls "{p}"')
            image_inst = p(image_inst)
        elif isinstance(p, Callback):
            if isinstance(p.check_on_call, (list, tuple)):
                image_inst = p(**{p.check_on_call[0]: image_inst})
            else:
                image_inst = p()

    return image_inst


def _text_type(some, after_last_dot=True, lowercase=False) -> str:

    some = type(some) if not inspect.isclass(some) else some
    _result = str(some).split("'")[1]

    if after_last_dot:
        if '.' in _result:
            _result = _result.split('.')[-1]
    return _result.lower() if lowercase else _result


def load_image(
                image_or_path: str or PillowImage or np.ndarray,
                result_type: Type[PillowImage] or Type[np.ndarray] = Type[PillowImage]) -> PillowImage or np.ndarray:

    """Loads image from file or converts image type between Pillow / numpy"""

    it, rt = _text_type(image_or_path, lowercase=True), _text_type(result_type, lowercase=True)

    if it not in ['str', 'image', 'ndarray']:
        if not isinstance(image_or_path, PillowImage):
            raise TypeError(f'utils.load_image incorrect image_or_path type: {type(result_type)}. it={it}, rt={rt}')

    if rt not in ['image', 'ndarray']:
        raise TypeError(f'utils.load_image incorrect result_type type: {type(result_type)}')

    if it == 'str':
        if rt == 'image':
            return PIL.Image.open(image_or_path)
        if rt == 'ndarray':
            return cv2.imread(image_or_path, )

    return convert_image_type(image_or_path, result_type)


def convert_image_type(
                source_image: PillowImage or np.ndarray or str,
                dest_type: Type[PillowImage] or Type[np.ndarray] = Type[PillowImage]) -> PillowImage or np.ndarray:

    """Loads image from file or converts image type between Pillow / numpy"""

    it, rt = _text_type(source_image, lowercase=True), _text_type(dest_type, lowercase=True)

    if it not in ['str', 'image', 'ndarray']:
        if not isinstance(source_image, PillowImage):
            raise TypeError(f'utils.convert_image incorrect source_image type: {type(source_image)}')

    if rt not in ['image', 'ndarray']:
        raise TypeError(f'utils.convert_image incorrect dest_type type: {type(dest_type)}')

    if it == 'str':
        return load_image(image_or_path=source_image, result_type=dest_type)
    if it == rt or (isinstance(source_image, PillowImage) and rt == 'image'):
        return source_image
    if rt == 'image':
        return PIL.Image.fromarray(source_image)
    if rt == 'ndarray':
        return np.asarray(source_image)


def verify_alpha_channel(image: image_types) -> np.ndarray:
    """
    Checks if loaded image has alpha channel. If not, adds it
    :param image: check variable image_types
    :return: cv2 image with alpha channel
    """

    if isinstance(image, PillowImage):
        print('$ utils.verify_alpha_channel for Pillow')
        image.save('tmp.png')
        result = PIL.Image.open('tmp.png')
        return result

    image = load_image(image, np.ndarray)
    if len(image.shape) < 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image


def get_image_size(image: image_types) -> tuple:
    if isinstance(image, str):
        image = load_image(image, PillowImage)
    if isinstance(image, PillowImage):
        return image.size
    elif isinstance(image, np.ndarray):
        return image.shape[:2][::-1]
    else:
        print(f'utils.get_image_size:  image is type({image})')
        raise AttributeError


def cv2_get_color_spaces() -> list:
    """
    Returns list of str with valid color spaces for converting between spaces of images
    :return: list of str
    """
    return sorted([attr for attr in dir(cv2) if 'COLOR_' in attr and '2' in attr and attr.count('_') == 1])


def convert_color_space(image: image_types, from_color='BGRA', to_color='RGB') -> np.ndarray or AttributeError:
    """
    Allows to convert color spaces of cv2 image. For list of available spaces use cv2_get_color_spaces().
    Raises an AttributeError if space is not available in OpenCV

    :param image: image to convert (instance of PIL.Image.Image, cv2 image, numpy.array or path to image on disk)
    :param from_color: string name of old color space. Case sensitive!
    :param to_color:  string name of new color space. Case sensitive
    :return: OpenCV image
    """
    image = load_image(image_or_path=image, result_type=np.ndarray)
    attr = f'COLOR_{from_color}2{to_color}'

    if not hasattr(cv2, attr):
        raise AttributeError(f'cv_effects.convert_color_space: OpenCV has no color space {attr}')

    return cv2.cvtColor(image, getattr(cv2, attr))


def save_image(image_inst: PillowImage or np.ndarray, full_path: str) -> str or OSError:
    ext_pos = full_path[::-1].index('.')
    ext = full_path[-ext_pos:]

    if ext not in ['jpg', 'png']:
        raise OSError(f'utils.save_image: incorrect extension "{ext}"')

    if ext == 'png':
        image_inst = convert_image_type(image_inst, np.ndarray)
        image_inst = convert_color_space(image_inst, 'BGR', 'BGRA')
        cv2.imwrite(full_path, image_inst)
    if ext == 'jpg':
        image_inst = convert_image_type(image_inst, np.ndarray)
        image_inst = convert_color_space(image_inst, 'BGRA', 'BGR')
        cv2.imwrite(full_path, image_inst)
    return full_path


def get_not_matching_resolution(images: list, resolution=(1920, 1080)) -> List[Tuple[int, image_types]]:
    result = []
    for pos, next_image in enumerate(images):
        if resolution != get_image_size(next_image):
            result.append((pos, next_image))
    return result
