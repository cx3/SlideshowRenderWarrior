# Python's STL
from __future__ import annotations
import os
import hashlib
import datetime
from typing import List, Type, Dict

# Third party imports
import cv2
import numpy

# RenderWarrior imports
from common import utils
import effects.cv_effects as cv_effects
import videoclip.ClipUtils as ClipUtils
import animations.transitions_classes as trans
import animations.ImageTravels as Travels


class ImageHandler:
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

    def get_image(self):
        if self.image is None or self.image is False:
            self.reload()
        print(f'>> StillImage.get_image() returns value of {type(self.image)}')
        return self.image

    # possible insafetyness ;>
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
            if index_or_list not in self.init_params['frame_indexes']:
                self.init_params['frame_indexes'].append(index_or_list)
        elif isinstance(index_or_list, (list, tuple)):
            no_repeated = [_ for _ in index_or_list if _ not in self.init_params['frame_indexes']]

            self.init_params['frame_indexes'].extend(no_repeated)
        else:
            raise ValueError

    def remove_frame_indexes(self, index_or_list: int or List[int]):
        if isinstance(index_or_list, int):
            if index_or_list not in self.init_params['frame_indexes']:
                self.init_params['frame_indexes'].remove(index_or_list)
        elif isinstance(index_or_list, (list, tuple)):
            for index in index_or_list:
                self.init_params['frame_indexes'].remove(index)
        else:
            raise ValueError


class ImagesManager:
    _images_handled: List[ImageHandler] = []


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
    def render_avi(dir_name: str, avi_full_path: str, fps=25):

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

"""
#######################################################################################################################
"""


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
        # :param total_slide_duration:
        :keyword params:
            :key fps: 24 or 25
            :key slide_seconds: = 10
            :key frame_file_type: 'jpg' or 'png', other value or key unset gives 'jpg' by default
        """

        '''
        [1] Check number of image sources, compare it to number of transitions
        [2] Check FPS, frame_resolution, calculate important data for rendering slides
        [3] Assign images to timeline, calculate transitions timing
        '''

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

        self.frame_resolution = frame_resolution
        self.global_frame_index = -1

        fft = params.get('frame_file_type', 'jpg')
        fft = 'jpg' if fft not in ['jpg', 'png'] else fft

        self.frame_file_type = fft
        self.image_handlers: List[ImageHandler] = []

        self.total_slide_duration = len(self.image_sources) * self.fps * (self.slide_seconds + self.transition_seconds)

        step = self.total_slide_duration // len(self.image_sources)
        for pos, next_source in enumerate(self.image_sources):
            self.image_handlers.append(
                ImageHandler(
                    full_path=next_source,
                    image_type=numpy.ndarray,
                    resolution=frame_resolution,
                    frame_indexes=[step * pos]
                )
            )

        for handler in self.image_handlers:
            print(' > handler.indexes: ', handler.get_frame_indexes())

        image_picker_cb = utils.Callback(fun_ptr=self.pick_image_for_frame, needs_params=('frame_index',))

        # TIMELINE CONTROLLER - MAIN CONTROL OF SELECTING IMAGE FOR FRAME
        self.slide_frames = ClipUtils.KeyedEffectsStack(
            ClipUtils.StillImage(
                frames_count=self.total_slide_duration,
                image=image_picker_cb
            ),
        )

        '''
         >> For each 500 frames do Zoom in & zoom out.
         >> To do so, we need some calculations and later assign it to slide frame controller
        '''

        travel_duration = (self.slide_seconds + self.transition_seconds) * self.fps
        travellers_circlelist = Travels.get_random_travellers_circlelist(travels_count=len(self.image_sources), debug=True)

        print('\t\ttraveller_circlelist: ', travellers_circlelist)

        self.images_traveller_controller: List[Dict] = []

        for next_index in range(0, self.total_slide_duration, travel_duration):
            self.images_traveller_controller.append({
                'start': next_index,
                'stop': next_index + travel_duration,
                'traveller': travellers_circlelist.next()
            })
            print('>>> images_traveller_controller: ', self.images_traveller_controller[-1])

        # position of start, end, selected transition type
        self.slide_transitions_timing: List[Dict] = []

        trans_len = len(self.transitions_list)
        self.one_trans_len = params.get('transition')  #  (params['slide_seconds'] * self.fps) // 4
        # self.one_slide_len = self.total_slide_duration // len(self.image_sources)

        for nr, next_trans in enumerate(self.transitions_list):
            start = (nr + 1) * (self.slide_seconds + self.transition_seconds) * self.fps
            stop = start + self.transition_seconds * self.fps
            self.slide_transitions_timing.append({
                'nr': nr,
                'start': start,
                'stop': start + (self.transition_seconds * self.fps),
                'transition': next_trans
            })
            # print(f'***]]] added {nr} -> {next_trans}')

        print('slide transitions timings:')
        for stt in self.slide_transitions_timing:
            print('>> timing: ', stt)

        print('\nWIZARD __init__ LOCALS()')
        for k, v in locals().items():
            print('\t' + k, ' -> ', v)
        print('WIZARD __init__ ENDS HERE\n')

    def has_more(self):
        return self.global_frame_index < self.total_slide_duration

    def pick_image_for_frame(self, frame_index: int = -1) -> List[utils.image_types, int]:
        if frame_index == -1:
            # print(f'Wizard::pick_image() for frame {frame_index} ==> (default) = {self.global_frame_index}')
            frame_index = self.global_frame_index

        closest,  result_handler_id = self.total_slide_duration, None
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

    def pick_image_traveller(self, frame_index: int = -1) -> List[dict, int] or List[None, int]:
        for index, next_dict in enumerate(self.images_traveller_controller):
            if next_dict['start'] <= frame_index < next_dict['stop']:
                return [next_dict, index]
        return [None, -1]

    def pick_next_image_traveller(self, frame_index: int = -1) -> List[dict, int] or List[None, int]:
        dict_, index = self.pick_image_traveller(frame_index)
        if isinstance(dict_, dict):
            if index < len(self.images_traveller_controller) - 1:
                return [self.images_traveller_controller[index + 1], index + 1]
            return [self.images_traveller_controller[index], index + 1]
        return [None, -1]

    def while_transition(self) -> bool or dict:
        if len(self.transitions_list) == 0:
            return False
        for next_dict in self.slide_transitions_timing:
            next_dict: dict
            if next_dict['start'] <= self.global_frame_index < next_dict['stop']:
                return next_dict
        return False

    def render_project_old_working_copy(self):
        print('\n\nWIZARD SLIDESHOW :: RENDER_PROJECT() \n')
        print('frames to render: ', self.total_slide_duration, '\t\thas_more', self.slide_frames.has_more())

        transition_frame_pos = -1

        current_image_traveller_instance = None
        current_image_traveller_index = -1
        current_transition_instance: trans.Transition or None = None

        print('total_slide_duration: ', self.total_slide_duration)

        for current_frame_index in range(self.total_slide_duration):
            self.global_frame_index = current_frame_index
            print('\n[ ' + '$$$' * 33 + ' ]')
            print('RENDER_PROJECT ITERATION global_frame_index: ', self.global_frame_index)

            image_inst, image_id = self.pick_image_for_frame(current_frame_index)
            result_rendered_frame_dict = self.slide_frames.next_frame(im1=image_inst)
            traveller_dict, traveller_index = self.pick_image_traveller(self.global_frame_index)

            if traveller_dict is not None and traveller_index != -1:
                if traveller_index != current_image_traveller_index:
                    current_image_traveller_instance = None

                if current_image_traveller_instance is None:
                    if len(result_rendered_frame_dict['to_save']) == 2:
                        result_rendered_frame_dict['to_save'] = result_rendered_frame_dict['to_save'][0]

                    print('\n\n>>>>>> render project: intantiating new image traveller')
                    current_image_traveller_instance = traveller_dict['traveller'](
                        images=[Noneresult_rendered_frame_dict['to_save'],  # self.last_rendered_frame
                        frames_count=traveller_dict['stop'] - traveller_dict['start'],
                        frame_resolution=self.frame_resolution
                    )
                    current_image_traveller_index = traveller_index

                if current_image_traveller_instance is not None:
                    print(f'\t\t\tImageTaveller next_frame.  type of "to_save"', type(result_rendered_frame_dict['to_save']))
                    result_rendered_frame_dict = current_image_traveller_instance.next_frame(
                        im1=result_rendered_frame_dict['to_save']
                    )

            # ------------------------------------------------------------------------------------------
            # Assign transition if global_frame_index is in transition timing
            selected_trans = self.while_transition()
            if isinstance(selected_trans, dict):
                print('\t\trender_project: assigning transition')

                selected_trans: dict
                transition_frame_pos += 1

                if current_transition_instance is None:
                    print('\n' + '$%$^&' * 10 + '\n\tINSTANTIATING TRANSITION', current_transition_instance)
                    # CREATE INSTANCE
                    current_transition_instance = selected_trans['transition'](
                        image1=utils.verify_alpha_channel(result_rendered_frame_dict['to_save']),
                        image2=utils.verify_alpha_channel(self.image_handlers[selected_trans['nr'] + 1].get_image()),
                        frames_count=abs(selected_trans['stop'] - selected_trans['start'])
                    )

                if hasattr(current_transition_instance, 'next_frame'):
                    # condition must be met!
                    print('\t>>>>>>>>>>>\tTRANSITING KURWA')
                    result_rendered_frame_dict = current_transition_instance.next_frame(
                        im1=utils.verify_alpha_channel(self.image_handlers[selected_trans['nr']].get_image()),
                        im2=utils.verify_alpha_channel(self.image_handlers[selected_trans['nr'] + 1].get_image()),
                        frame_resolution=self.frame_resolution
                    )
                    self.image_handlers[selected_trans['nr']].set_image(result_rendered_frame_dict['to_save'])
                    self.image_handlers[selected_trans['nr'] + 1].set_image(result_rendered_frame_dict['im2'])
                    # input('ENTER KURWA JEBANA')
                else:
                    raise AttributeError(str(type(current_transition_instance)))

            else:
                current_transition_instance = None
                transition_frame_pos = -1

            utils.save_image(result_rendered_frame_dict['to_save'], f'rendered/{self.global_frame_index}.jpg')
            # print(nf.keys())
            print('=' * 44 + ' RENDER_FRAME_ITERATION FINISHED ' + '=' * 44)
            # input('enter...')

    # ** ..(^^,).. ~^~ .. (,``).. ** ..(^^,).. ~^~ .. (,``).. **** ..(^^,).. ~^~ .. (,``).. ** ..(^^,).. ~^~ .. (,``)..
    # ..(^^,).. ~^~ .. (,``).. ** ..(^^,).. ~^~ .. (,``).. **** ..(^^,).. ~^~ .. (,``).. ** ..(^^,).. ~^~ .. (,``).. **

    def render_project(self, start_frame: int = 0, stop_frame: int = -1, frame_resolution=(-1, -1), **kwargs):

        """
        Render frames of Slideshow. Passing no arguments renders whole frames with default values

        :param start_frame: start frame index
        :param stop_frame: last frame index
        :param frame_resolution: if set to default (-1, -1) a self.frame_resolution field is used.
            Use it for i.e preview purpose

        :keyword kwargs:
            :key avi: str full path to avi file created after rendering all frames
            :key show_avi: is used only if :avi: is passed
            :key del_frames: removes all rendered frames after rendering avi, so use it with :avi:
            :key show_dir: after rendering frames, launch system files explorer with rendered frames directory, do not
                use it when :del_frames: is set to True
        :return:
        """

        start_frame = 0 if start_frame < 0 else start_frame
        stop_frame = \
            self.total_slide_duration if stop_frame == -1 or stop_frame > self.total_slide_duration else stop_frame

        frame_step = 1 if start_frame < stop_frame else -1

        current_image_traveller_instance = None
        current_image_traveller_index = -1
        current_transition_instance: trans.Transition or None = None

        transition_frame_pos = -1

        if frame_resolution == (-1, -1):
            frame_resolution = self.frame_resolution

        # MAIN RENDERING LOOP
        for current_frame_index in range(start_frame, stop_frame, frame_step):
            self.global_frame_index = current_frame_index
            print('\n[ ' + '$$$' * 33 + ' ]')
            print('RENDER_PROJECT ITERATION global_frame_index: ', self.global_frame_index)

            image_inst, image_id = self.pick_image_for_frame(current_frame_index)

            result_rendered_frame_dict = self.slide_frames.next_frame(
                im1=image_inst
            )

            # ------------------------------------------------------------------------------------------
            # Assign image travelling
            traveller_dict, traveller_index = self.pick_image_traveller(self.global_frame_index)

            if traveller_dict is not None and traveller_index != -1:
                if traveller_index != current_image_traveller_index:
                    current_image_traveller_instance = None

                if current_image_traveller_instance is None:

                    if len(result_rendered_frame_dict['to_save']) == 2:
                        result_rendered_frame_dict['to_save'] = result_rendered_frame_dict['to_save'][0]

                    print('\n\n>>>>>> render project: instantiating new image traveller')
                    current_image_traveller_instance = traveller_dict['traveller'](
                        image=result_rendered_frame_dict['to_save'],  # self.last_rendered_frame
                        frames_count=traveller_dict['stop'] - traveller_dict['start'],
                        frame_resolution=self.frame_resolution
                    )

                    current_image_traveller_index = traveller_index
                    # input('Enter...')

                if current_image_traveller_instance is not None:
                    print('\t\t\tImageTaveller next_frame')
                    result_rendered_frame_dict = current_image_traveller_instance.next_frame(
                        im1=result_rendered_frame_dict['to_save']
                    )

            # ------------------------------------------------------------------------------------------
            # Assign transition if global_frame_index is in transition timing
            selected_trans = self.while_transition()
            if isinstance(selected_trans, dict):
                print('\t\trender_project: assigning transition')

                selected_trans: dict
                transition_frame_pos += 1

                second_image_travelled = utils.verify_alpha_channel(
                    self.image_handlers[selected_trans['nr'] + 1].get_image()
                )

                if current_transition_instance is None:
                    print('\n' + '$%$^&' * 10 + '\n\tINSTANTIATING TRANSITION')
                    # CREATE INSTANCE

                    if isinstance(result_rendered_frame_dict['to_save'], list):
                        result_rendered_frame_dict['to_save'] = result_rendered_frame_dict['to_save'][0]

                    current_transition_instance = selected_trans['transition'](
                        image1=utils.verify_alpha_channel(result_rendered_frame_dict['to_save']),
                        image2=second_image_travelled,
                        frames_count=abs(selected_trans['stop'] - selected_trans['start'])
                    )

                if hasattr(current_transition_instance, 'next_frame'):
                    result_rendered_frame_dict = current_transition_instance.next_frame(
                        im1=utils.verify_alpha_channel(self.image_handlers[selected_trans['nr']].get_image()),
                        im2=utils.verify_alpha_channel(second_image_travelled),
                        frame_resolution=self.frame_resolution
                    )

                else:
                    raise AttributeError(str(type(current_transition_instance)))
            else:
                current_transition_instance = None
                transition_frame_pos = -1

            if frame_resolution != self.frame_resolution:
                result_rendered_frame_dict['to_save'] = cv_effects.resize_image(
                    image=result_rendered_frame_dict['to_save'],
                    dest_image_type=None,  # do not touch type of image, stay it
                    dest_resolution=frame_resolution
                )
            utils.save_image(result_rendered_frame_dict['to_save'], f'rendered/{self.global_frame_index}.jpg')
            print('=' * 44 + ' RENDER_FRAME_ITERATION FINISHED ' + '=' * 44)

        # ------------------------------------------------------------------------------------------
        # Do it after rendering frames
        print('\n\nRendering slideshow frames complete')

        if kwargs.get('avi', False):
            print('\nRendering avi file')
            TimelineModel.render_avi('rendered', kwargs['avi'], self.fps)

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
