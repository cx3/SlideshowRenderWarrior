from time import time
from inspect import isclass

import numpy as np
import PIL.Image as PilImage
import PIL

import utils
from videoclip import RenderControllers

from animations.Transitions_v2 import image_types, create_empty_callbacks_cl, NextFrame
from animations.ImageTravels import ImageTravel


PillowImage = PIL.Image.Image


class Transition(NextFrame):
    def __init__(
            self,
            image1: image_types or None = None,
            image2: image_types or None = None,
            dest_dir: str = 'rendered',
            frames_count=50,
            **kwargs
    ):  # -~/*^/-_/~-~/*^/-_/~-~/*^/-_/~-~/*^/-_/~-~/*^/-_/~-~/*^/-_/~-~/*^/-_/~-~/*^/-_/~-~/*^/-_/~-~/*^/-_/~-~/*^-.
        """
        Base class for animations between images.
        :param image1: look at utils.image_types
        :param image2: look at utils.image_types
        :param dest_dir: there rendered frames will be saved
        :param frames_count: number of frames to render
        :param kwargs:
            usable keys:
              > parent_timeline: Timeline
              > prefix: str -> prefix in file name of rendered frame
              > extension: str -> extension of output file (rendered frame)
              > load_as: [PillowImage, np.ndarray] -> tells how image file should be opened

              > preprocess_first, preprocess_second, postprocess_first, postprocess_second:

                - list of callbacks to predicates or CircleList(list of callbacks/predicates) or ImageTravel instance
                - Type name of ImageTravel class  (type name, not instance or its __init__ method call!

        """

        '''self.fill_later = False
        if kwargs.get('fill_later', False) is True:
            self.fill_later = True
            return'''

        self.im1 = self.im2 = None
        if image1 is not None and image2 is not None:
            self.im1, self.im2 = utils.verify_alpha_channel(image1), utils.verify_alpha_channel(image2)

        '''if image1 is None and image2 is None:
            err = 'If params image1 is None and image2 is None, pass parent_timeline argument of Timeline instance'
            if 'parent_timeline' not in kwargs:
                raise AttributeError(err)
            if not isinstance(kwargs['parent_timeline'], RenderController.Timeline):
                raise AttributeError(err)'''

        if 'prefix' not in kwargs:
            class_name = str(self.__class__).split("'")[1][:-2]
            if '.' in class_name:
                self.prefix = class_name.split('.')[-1]
        else:
            self.prefix = kwargs['prefix']

        if 'extension' not in kwargs:
            kwargs['extension'] = 'png'
        else:
            if kwargs['extension'] not in ['jpg', 'png']:
                kwargs['extension'] = 'png'

        if 'frame_resolution' not in kwargs:
            if self.im1 is None:
                tl: RenderControllers.TimelineModel = kwargs.get('parent_timeline', False)
                if not tl:
                    raise AttributeError('Cannot obtain resolution of frame')
                kwargs['frame_resolution'] = tl.get_frame_resolution()
            else:
                kwargs['frame_resolution'] = utils.get_image_size(self.im1)

        self.dest_dir = dest_dir
        self.frames_count = frames_count
        self.kwargs = kwargs
        self.name_counter = 0

        if 'load_as' not in kwargs:
            self.kwargs['load_as'] = PillowImage
            if isinstance(image1, np.ndarray):
                self.kwargs['load_as'] = np.ndarray

        self.effects = {}
        self.frame_index = -1

    def get_image(self, first=True) -> image_types:
        if first is True:
            if self.im1 is not None:
                return self.im1
            self.im1 = self.kwargs['parent_timeline'].current_frame_image(0)
            return self.im1
        else:
            if self.im2 is not None:
                return self.im2
            self.im2 = self.kwargs['parent_timeline'].current_frame_image(1)
            return self.im2

    # little bit unsafe, no corectness checking
    def set_image(self, image: image_types, first=True) -> bool:
        if first:
            self.im1 = image
        else:
            self.im2 = image
        return True

    def set_effects(self, kwargs: dict):
        self.effects = {}
        no_effects_list = create_empty_callbacks_cl(self.frames_count)

        for item in ('preprocess_first', 'preprocess_second', 'postprocess_first', 'postprocess_second'):

            value = kwargs.get(item, no_effects_list)

            print(f'@@ set_effects!! item={item} of type={type(value)}')

            if isclass(value):
                # noinspection PyTypeChecker
                if issubclass(value, ImageTravel):
                    value: ImageTravel = value(
                        image=self.im1,
                        frames_count=self.frames_count,
                        frame_resolution=utils.get_image_size(self.im1)
                    )

            if isinstance(value, ImageTravel):
                print('TRANSITION: if isinstance(value, ImageTravel):')
                cb = utils.Callback(fun_ptr=value.next_frame, needs_params=('frame_index', 'im1'))

                value: utils.CircleList = utils.CircleList(
                    list_or_tuple=[cb],
                    max_iterations=self.frames_count
                )

            if isinstance(value, (list, tuple)):
                value: utils.CircleList = utils.CircleList(list_or_tuple=value, max_iterations=self.frames_count)
            if not isinstance(value, utils.CircleList):
                raise TypeError
            value.max_iterations = self.frames_count
            self.effects[item] = value

    def _assign_next_effect(
            self, image: image_types, image_param_name='image', process_name='preprocess_first', frame_index=0):
        # print('_assign_next_efect: ' + process_name)
        cl: utils.CircleList = self.effects[process_name]
        call_back: utils.Callback = cl.next()

        if isinstance(call_back, utils.Callback):
            if call_back.needs_param('frame_index'):
                call_back.add_one_arg('frame_index', frame_index)
            if call_back.needs_param('im1'):
                call_back.add_one_arg('im1', image)
            if call_back.needs_param('im2'):
                call_back.add_one_arg('im2', image)

        image = call_back.__call__(**{image_param_name: image})

        if isinstance(image, dict):
            image = image['to_save']

        return utils.convert_image_type(
            source_image=image,
            dest_type=self.kwargs['load_as']
        )

    def _preprocess(self, first_image: image_types, second_image: image_types, frame_index: int) -> tuple:
        # print('>> Preprocess effects...')
        return (
            self._assign_next_effect(
                utils.convert_image_type(source_image=first_image, dest_type=self.kwargs['load_as']),
                process_name='preprocess_first',
                frame_index=frame_index
            ),
            self._assign_next_effect(
                utils.convert_image_type(source_image=second_image, dest_type=self.kwargs['load_as']),
                process_name='preprocess_second',
                frame_index=frame_index
            )
        )

    def _postprocess(self, first_image: image_types, second_image: image_types, frame_index: int) -> tuple:
        # print('>> Postprocess effects...')
        return (
            self._assign_next_effect(first_image, process_name='postprocess_first', frame_index=frame_index),
            self._assign_next_effect(second_image, process_name='postprocess_second', frame_index=frame_index)
        )

    def name(self, extension=None) -> str:
        """
        Creates next frame name using fields dest_dir, prefix, name_counter, extension
        :param extension: str name of image's extension, or None
        :return: path to rendered frame
        """
        if extension is None:
            extension = self.kwargs['extension']

        self.name_counter += 1
        return f"{self.dest_dir}/{self.prefix}{self.name_counter}.{extension}"

    def save(self, image: image_types, name=None) -> str:
        if name is None:
            name = self.name()

        fp = utils.save_image(image_inst=image, full_path=name)
        print('> Saved transition image:', fp)
        return fp

    def init_machine(self):
        self.set_effects(self.kwargs)

    def render_frames_deprecated(self) -> dict:
        self.init_machine()
        start = time()
        result = {}
        names = []
        im1, im2 = self.im1, self.im2
        frame_step = 1 / self.frames_count
        for i in range(self.frames_count):
            im1, im2 = self._preprocess(im1, im2, frame_index=i)
            frame_dict = self.next_frame(frame_index=i, im1=im1, im2=im2, step=frame_step)
            im1, im2 = self._postprocess(frame_dict['im1'], frame_dict['im2'], frame_index=i)
            names.append(self.save(frame_dict['to_save']))
        result['time'] = time() - start
        result['names'] = names
        return result

    def has_more(self) -> bool:
        return self.frame_index > self.frames_count

    def next_frame(self, **params) -> dict:
        params['im1'], params['im2'] = self._preprocess(
            first_image=params['im1'],
            second_image=params['im2'],
            frame_index=params['frame_index']
        )
        params = self._next_frame(**params)

        params['im1'], params['im2'] = self._postprocess(
            first_image=params['im1'],
            second_image=params['im2'],
            frame_index=params['frame_index']
        )
        return params

    def _next_frame(self, frame_index: int, im1: image_types, im2: image_types, **params) -> dict:
        print('transi next_frame')
        result = locals()

        if frame_index is None or frame_index == -1:
            frame_index = self.frame_index

        frame_index += 1
        if frame_index > self.frames_count:
            raise ValueError
        # im1, im2 = self._preprocess(im1, im2, frame_index=frame_index)
        step = params.get('step', 1 / self.frames_count)
        im1 = PIL.Image.blend(im1, im2, frame_index * step)
        # im1, im2 = self._postprocess(im1, im2, frame_index=frame_index)
        result['to_save'] = im1
        return result


class BlendingTransition(Transition):
    def init_machine(self):
        print('bt init_machine')
        self.set_effects(self.kwargs)

    def _next_frame(self, frame_index: int, im1: image_types, im2: image_types, **params) -> dict:
        print('bt next_frame')
        result = locals()

        im1 = utils.convert_image_type(im1, PillowImage)
        im2 = utils.convert_image_type(im2, PillowImage)

        im1 = PilImage.blend(im1=im1, im2=im2, alpha=frame_index/self.frames_count)
        result['to_save'] = im1

        return result
