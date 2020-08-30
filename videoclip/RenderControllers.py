# !/python38
# Python's STL
from __future__ import annotations
import os
# import time
import hashlib
import datetime
from typing import List, Type

# Third party imports
import cv2
import numpy

# RenderWarrior imports
import utils
import effects.cv_effects as cv_effects
import animations.transitions_classes as trans
import animations.ImageTravels as Travels


class ImageHandler:

    _loaded_instances = []

    def __init__(
            self,
            full_path: str,
            image_type: utils.image_types = numpy.ndarray,
            **kwargs):
        """

        :param full_path: full path to image on drive
        :param image_type: PIL.Image.Image or numpy.ndarray
        :keyword kwargs:
            :key load: loads image to RAM, pass True value for it
            :key resolution: tuple (width, height) that tells to handle rescaled image
            :key frame_indexes: list/tuple of ints describing positions on timeline where image should occur
            :key uid_setter: instance of such class with method set_next_uid() -> str
        """

        kwargs['frame_indexes'] = kwargs.get('frame_indexes', [0])
        self.init_params = locals()
        self.uid = None

        if kwargs.get('uid_setter', False):
            uid_setter = kwargs['uid_setter']
            if hasattr(uid_setter, 'set_next_uid'):
                uid_setter = getattr(uid_setter, 'set_next_uid')
                if callable(uid_setter):
                    self.uid = uid_setter()

        self.image = False if not kwargs.get('load', False) else utils.load_image(
            image_or_path=full_path,
            result_type=image_type
        )

        if 'resolution' in kwargs:
            if self.image is not False:
                self.image = utils.load_image(self.image, numpy.ndarray)
                size = utils.get_image_size(self.image)
                if size != kwargs['resolution']:
                    print('ImageHandler resize resolution to', kwargs['resolution'])
                    input('enter...')
                    self.image = cv2.resize(self.image, kwargs['resolution'])

                self.image = utils.convert_image_type(self.image, image_type)
        ImageHandler._loaded_instances.append(self)

    def __str__(self):
        return str(self.init_params)

    def get_image(self):
        if self.image is None or self.image is False:
            self.reload()
        print(f'>> StillImage.get_image() returns value of {type(self.image)}')
        return self.image

    def set_image(self, image_inst):
        curr_size = utils.get_image_size(self.image)
        next_size = utils.get_image_size(image_inst)

        if curr_size != next_size:
            image_inst = cv2.resize(utils.convert_image_type(image_inst, numpy.ndarray), curr_size)
            image_inst = utils.convert_image_type(image_inst, type(image_inst))

        self.image = image_inst

    def dismiss_image(self):
        del self.image
        self.image = None

    def reload(self, **kwargs) -> True or utils.image_types:
        """
        Reloads image from drive with other params than in __init__
        :keyword kwargs:
            :key image_type:  PIL.Image.Image or numpy.ndarray, or passed in __init__
            :key resolution:  new resolution or passed in __init__
            :key return_image: if passed and set to True, method returns reloaded image, else returns True
        :return: look above
        """
        self.image = utils.load_image(self.init_params['full_path'], self.init_params['image_type'])
        resolution = self.init_params['kwargs']['resolution']
        image_type = kwargs.get('image_type', self.init_params['image_type'])

        if kwargs.get('resolution', None) is not None:
            resolution = kwargs['resolution']

        if resolution is not None:
            print('ImageHandler reloading, resize to', resolution)
            loaded_image_size = utils.get_image_size(self.image)
            if loaded_image_size != resolution:
                self.image = utils.convert_image_type(
                    source_image=cv_effects.cut_rect(
                        image=self.image,
                        src_box=(0, 0, *loaded_image_size),
                        size=resolution
                    ),
                    dest_type=image_type
                )
        self.init_params['reload_params'] = kwargs
        if kwargs.get('return_image', None) is not None:
            return self.image
        return True

    def get_frame_indexes(self) -> List[int]:
        return sorted(self.init_params['kwargs']['frame_indexes'])

    def add_frame_indexes(self, index_or_list: int or List[int]):
        if isinstance(index_or_list, int):
            if index_or_list not in self.init_params['kwargs']['frame_indexes']:
                self.init_params['kwargs']['frame_indexes'].append(index_or_list)
        elif isinstance(index_or_list, (list, tuple)):
            no_repeated = [_ for _ in index_or_list if _ not in self.init_params['kwargs']['frame_indexes']]

            self.init_params['frame_indexes'].extend(no_repeated)
        else:
            raise ValueError

    def remove_frame_indexes(self, index_or_list: int or List[int]):
        if isinstance(index_or_list, int):
            if index_or_list in self.init_params['kwargs']['frame_indexes']:
                self.init_params['frame_indexes'].remove(index_or_list)
        elif isinstance(index_or_list, (list, tuple)):
            for index in index_or_list:
                if index in self.init_params['frame_indexes']:
                    self.init_params['frame_indexes'].remove(index)
        else:
            raise ValueError


class VideoSettings:
    youtube_hd_24fps = {
        'frame_resolution': (1280, 720),
        'fps': 24
    }

    youtube_hd_25fps = {
        'frame_resolution': (1280, 720),
        'fps': 25
    }

    youtube_fhd_25fps = {
        'frame_resolution': (1920, 1080),
        'fps': 25
    }

    sd_24fps = {
        'frame_resolution': (720, 576),
        'fps': 24
    }


class TimelineModel:
    def __init__(self, **params):
        self.init_params = params
        self.image_sources = params.get('image_sources', [])
        self.frame_resolution = params.get('frame_resolution', (1920, 1080))
        self.transitions_list = params.get('transitions_list', [])
        self.fps = params.get('fps', 25)
        self.slide_seconds = params.get('slide_seconds', 10)
        self.last_rendered_frame: utils.image_types or None = None
        self.transition_seconds = params.get('transition_seconds', 2)
        print('Timeline :: init_params:')
        for k, v in params.items():
            print(f'\t{k}: {v}')
        print('='*55)

    def get_frame_resolution(self):
        return self.frame_resolution

    def current_frame_image(self, index):
        print(f'Timeline::current_frame_index({index})')
        return utils.load_image(self.image_sources[index], numpy.ndarray)

    def render_project(self, **kwargs):
        pass

    @staticmethod
    def set_next_uid() -> str:
        sha256 = hashlib.sha256()
        sha256.update(bytearray(datetime.datetime.now().strftime("%Y-%m-%d @ %H:%M:%S")))
        return str(sha256.digest())[2:-1]  # trim prefix b' and postfix '

    @staticmethod
    def show_rendered_directory(dir_name):
        if os.path.isdir(dir_name):
            if os.name == 'nt':
                os.system('explorer ' + dir_name)
            else:
                os.system('nautilus ' + dir_name)

    @staticmethod
    def render_avi_no_sound(dir_name: str, avi_full_path: str, fps=25):

        if not os.path.isdir(dir_name):
            raise OSError

        if not avi_full_path.lower().endswith('.avi'):
            avi_full_path += '.avi'

        if dir_name[-1] not in ['\\', '/']:
            dir_name += os.sep

        frames = list((dir_name + img for img in os.listdir(dir_name) if img.split('.')[-1].lower() in ['jpg', 'png']))
        frames = sorted(frames, key=lambda x: int(x.split(os.sep)[1].split('.')[0]))

        frames_len = len(frames)
        frame_res = utils.get_image_size(frames[0])

        writer = cv2.VideoWriter(avi_full_path, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, frame_res)

        print('Frames count: ' + str(frames_len))
        j = 0
        for i in range(frames_len):
            if i > 0 and i % fps == 0:
                print(f'{i+1}/{frames_len}', end=', ')
                j += 1
                if j > 0 and j % 10 == 0:
                    print(end='\n')
            writer.write(utils.load_image(frames[i], numpy.ndarray))

        print('Rendering AVI Done!')


""" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" ""  ""'
#######################################################################################################################
"" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" ""'' """


class WizardSlideshowZoomy(TimelineModel):
    def __init__(
            self,
            image_sources: List[utils.image_types],
            transitions_list: List[Type[trans.Transition]] or None = None,
            frame_resolution=(1920, 1080),
            **params):

        """
        Wizard slideshow
        :param image_sources:
        :param transitions_list:
        :param frame_resolution:
        :keyword params:
            :key fps: 24 or 25
            :key slide_seconds: = 10
            :key frame_file_type: 'jpg' or 'png', other value or key unset gives 'jpg' by default
        """

        # Check basic integrity

        if len(image_sources) < 2:
            raise ValueError

        if len(image_sources) > 1:
            if len(transitions_list) == 0:
                transitions_list = [
                    trans.BlendingTransition
                ] * (len(image_sources) - 1) if transitions_list is None else transitions_list
        elif transitions_list == 0:
            transitions_list = []

        if len(transitions_list) >= len(image_sources):
            if len(image_sources) > 1:
                transitions_list = transitions_list[:len(image_sources) - 1]
            else:
                transitions_list = [transitions_list[0]]

        super().__init__(**{k: v for k, v in {**params, **locals()}.items() if k != 'self'})

        # Image appearing controller settings
        self.frame_resolution = frame_resolution
        self.global_frame_index = -1

        fft = params.get('frame_file_type', 'jpg')
        fft = 'jpg' if fft not in ['jpg', 'png'] else fft

        self.frame_file_type = fft
        self.image_handlers: List[ImageHandler] = []
        self.total_slide_duration = len(self.image_sources) * self.fps * (self.slide_seconds + self.transition_seconds)

        # Tell at which frame image should appear
        self.next_appear_step = self.total_slide_duration // len(self.image_sources)
        for pos, next_source in enumerate(self.image_sources):
            self.image_handlers.append(
                ImageHandler(
                    full_path=next_source,
                    image_type=numpy.ndarray,
                    resolution=frame_resolution,
                    frame_indexes=[self.next_appear_step * pos]
                )
            )

        for handler in self.image_handlers:
            print('handler:', handler)

        # Set up zooming in/out and transition
        travel_duration = (self.slide_seconds - self.transition_seconds // 2) * self.fps
        print('travel_duration', travel_duration)
        self.images_traveller = Travels.ExampleTravel()

        self.images_traveller.set_params(
            images=[],
            frames_count=travel_duration * 3,
            frame_resolution=self.frame_resolution
        )
        self.slide_transitions_timing: List[dict] = []

        for nr, handler in enumerate(self.image_handlers):
            if nr > 0:
                appear = handler.get_frame_indexes()[0]
                self.slide_transitions_timing.append({
                    'nr': nr - 1,
                    'start': appear - (self.transition_seconds * self.fps) // 2,
                    'stop': appear + (self.transition_seconds * self.fps) // 2,
                    'transition': self.transitions_list[nr - 1]
                })

        print('slide transitions timings:')
        for stt in self.slide_transitions_timing:
            print('>> timing: ', stt)

        print('\nWIZARD __init__ LOCALS()')
        for k, v in locals().items():
            print('\t' + k, ' -> ', v)
        print('WIZARD __init__ ENDS HERE\n')

    def has_more(self):
        return self.global_frame_index < self.total_slide_duration

    def pick_images_for_frame(self, frame_index: int = -1) -> List[List[utils.image_types, int]]:
        result = []
        if frame_index == -1:
            # print(f'Wizard::pick_image() for frame {frame_index} ==> (default) = {self.global_frame_index}')
            frame_index = self.global_frame_index

        for handler_id, image_handler in enumerate(self.image_handlers):
            indexes = image_handler.get_frame_indexes()

            if indexes[0] <= frame_index < indexes[0] + self.next_appear_step:
                result.append([image_handler, handler_id])

        return result

    def pick_image_for_frame(self, frame_index: int = -1) -> List[utils.image_types, int]:
        if frame_index == -1:
            # print(f'Wizard::pick_image() for frame {frame_index} ==> (default) = {self.global_frame_index}')
            frame_index = self.global_frame_index

        closest, result_handler_id = self.total_slide_duration, None
        for handler_id, image_handler in enumerate(self.image_handlers):
            indexes = image_handler.get_frame_indexes()

            if frame_index >= indexes[0]:
                dist = frame_index - indexes[0]
                if dist < closest:
                    closest, result_handler_id = dist, handler_id
        if result_handler_id is None:
            raise ValueError('No such image')
        result = self.image_handlers[result_handler_id].get_image()
        print(f'picked image.index = {result_handler_id}, with shape:', utils.get_image_size(result))
        return [
            cv2.resize(utils.convert_image_type(result, numpy.ndarray), self.frame_resolution),
            result_handler_id
        ]

    def while_transition(self) -> False or dict:
        if len(self.transitions_list) == 0:
            return False
        for next_dict in self.slide_transitions_timing:
            next_dict: dict
            if next_dict['start'] <= self.global_frame_index < next_dict['stop']:
                return next_dict
        return False

    def render_project(self, start_frame: int = 0, stop_frame: int = -1, frame_resolution=(-1, -1), **kwargs):

        start_frame = 0 if start_frame < 0 else start_frame
        stop_frame = \
            self.total_slide_duration if stop_frame == -1 or stop_frame > self.total_slide_duration else stop_frame

        file_names = []
        current_transition = None
        frame_step = 1 if start_frame < stop_frame else -1
        frame_resolution = self.frame_resolution if frame_resolution == (-1, -1) else self.frame_resolution

        for current_frame_index in range(start_frame, stop_frame, frame_step):

            print('WIZARD RENDER_PROJECT MAIN_LOOP: ', current_frame_index)
            self.global_frame_index = current_frame_index
            transition_dict = self.while_transition()

            current_frame_image, handler_index = self.pick_image_for_frame()

            if isinstance(transition_dict, dict):
                """ TRANSITION BETWEEN IMAGES """

                if current_transition is None:
                    current_transition = transition_dict['transition'](
                        frames_count=abs(transition_dict['start'] - transition_dict['stop']),
                        frame_resolution=frame_resolution
                    )

                try:
                    second = self.image_handlers[transition_dict['nr'] + 1].get_image()
                except IndexError:
                    second = current_frame_image

                traveled_images_dict = self.images_traveller.next_frame(
                    images=[
                        current_frame_image,
                        # self.image_handlers[transition_dict['nr'] + 1].get_image()
                        second
                    ]
                )

                for i in range(len(traveled_images_dict['images'])):
                    traveled_images_dict['images'][i] = utils.verify_alpha_channel(
                        traveled_images_dict['images'][i])
                    if utils.get_image_size(traveled_images_dict['images'][i]) != frame_resolution:
                        traveled_images_dict['images'][i] = cv_effects.resize_image(
                            image=utils.verify_alpha_channel(traveled_images_dict['images'][i]),
                            dest_resolution=frame_resolution
                        )

                first_image, second_image = traveled_images_dict['images']

                rendered_frame_dict = current_transition.next_frame(
                    im1=first_image,
                    im2=second_image,
                    frame_resolution=frame_resolution
                )
            else:
                """ USUAL IMAGE TRAVEL """
                current_transition = None

                traveled_images_dict = self.images_traveller.next_frame(
                    images=[
                        current_frame_image,
                    ]
                )
                rendered_frame_dict = {'to_save': traveled_images_dict['images'][0]}

            if frame_resolution != self.frame_resolution:
                rendered_frame_dict['to_save'] = cv_effects.resize_image(
                    image=rendered_frame_dict['to_save'],
                    dest_image_type=None,  # do not touch type of image, stay it
                    dest_resolution=frame_resolution
                )
            file_name = f'rendered/{self.global_frame_index}.jpg'
            utils.save_image(rendered_frame_dict['to_save'], file_name)
            file_names.append(file_name)

            print('\nWIZARD RENDER_PROJECT LOOP END\n')

        # ------------------------------------------------------------------------------------------
        # Do it after rendering frames
        print('\n\nRendering slideshow frames complete')

        if kwargs.get('avi', False):
            print('\nRendering avi file')
            TimelineModel.render_avi_no_sound('rendered', kwargs['avi'], self.fps)

            if kwargs.get('show_avi', False) is True:
                os.system('start ' + kwargs['avi'])

            if kwargs.get('del_frames', False) is True:
                frames_dir = 'rendered'
                if frames_dir[-1] not in ['\\', '/']:
                    frames_dir += os.sep

                for file in os.listdir('rendered'):
                    os.remove(frames_dir + file)

        if kwargs.get('show_dir', False) is True:
            if kwargs.get('del_frames', False) is False:
                TimelineModel.show_rendered_directory('rendered')
