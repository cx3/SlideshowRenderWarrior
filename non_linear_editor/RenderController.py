# !/python38
# Python's STL
from __future__ import annotations
# import time
from copy import deepcopy
from typing import Callable, List, Tuple, Dict

# Third party imports
import cv2
import numpy
import PIL.Image

# RenderWarrior imports
from common import utils
import effects.cv_effects as cv_effects
from animations.Transitions_v2 import NextFrame
from common.UuidController import UuidController


image_types = utils.image_types


class ImageHandler:

    _loaded_instances: List[dict] = []

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

        ImageHandler._loaded_instances.append(
            UuidController.set_uid(self)
        )

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


class OnTimingChanged:
    main_timeline_instance: TimelineController
    not_listened_methods = []

    @staticmethod
    def do_on_change():
        OnTimingChanged.main_timeline_instance.stabilize_transitions()


# https://stackoverflow.com/questions/6307761/how-to-decorate-all-functions-of-a-class-without-typing-it-over-and-over-for-eac
# https://github.com/davegallant/decorate-all-methods/blob/master/decorate_all_methods/__init__.py
# https://www.codementor.io/@sheena/advanced-use-python-decorators-class-function-du107nxsv


def on_timing_changed_listener(decorator=OnTimingChanged.do_on_change):
    def decorate(cls):
        for next_attr in cls.__dict__:
            if callable(getattr(cls, next_attr)) and next_attr not in OnTimingChanged.not_listened_methods:
                decorator: Callable
                setattr(cls, next_attr, decorator(getattr(cls, next_attr)))
        return cls
    return decorate


def do_not_listen(method: Callable):
    if callable(method):
        if method not in OnTimingChanged.not_listened_methods:
            OnTimingChanged.not_listened_methods.append(method)
    return method


class KeyedEffect(NextFrame):
    """
    Use objects of this class to change image effects strength while iterating over timeline.
    Inheriting from NextFrame, KeyedEffect should implements next_frame method for rendering
    animation frame as a part of frame based animation like gif, mp4, webm...
    """

    def __init__(self, effect_fun_ptr: Callable or utils.Callback, frames_count: int, **cb_params_key):
        """
        PIL.Image / OpenCV2 / Scikit image based effect callback handler.

        Main purpose of this class is to set argument list of  __init__ method for rendering animation at time line, at
        transition, as a part of pre-/post-process effects stack at rendering next_frame of animation that may consists
        of image sequence, video, audio, all of them or more.

        :param effect_fun_ptr: RenderWarrior any callable / Callback that's name has  'effect', takes at least one arg
            of type [PIl.Image.Image, numpy.ndarray] which must be a first argument of an passed pointer to effect
        :param frames_count:
            tells how much frames should be rendered with applied effect. Remember to use positive int numbers only
        :keyword cb_params_key:
            Keyword variables for changing effect behavior, strength while rendering next frame of animation

            :param needs: by default an empty list, otherwise when passed should be a list of str that are the names of
                parameters that MUST be passed to an effect function or TypeError will be raised.
                For testing purposes so do not be afraid to much about it's value

            The keys of param_key > should match < to arguments names of effect_fun_ptr, otherwise while rendering the
            next frame of animation the effect function will be called with default values of parameters.
            Beware that it even may guide to TypeError due to passing unexpected arguments!

            USE IT AS EXAMPLE BELOW:
                a) param_name1={'start_value': 0, 'step_value': 10},
                b) param_name2={'start_value': 50, 'step_value': -0.3, 'per_frame': 5}
                c) param_name3={'start_value': 44, 'step_value': 0}

            EXPLANATION:

                a) effect function will be run each time increasing param_name previous value by 10
                b) effect function will be run each FIVE FRAMES decreasing param_name2 by 0.3
                c) effect function will be run each time with constant value set to start_value
        """

        self.need_params = cb_params_key.get('needs', [])

        if isinstance(effect_fun_ptr, utils.Callback):
            effect_fun_ptr = effect_fun_ptr.params['fun_ptr']

        if not isinstance(effect_fun_ptr, utils.Callback) and not callable(effect_fun_ptr):
            raise TypeError('param effect_fun_ptr is not Callback nor callable')
        if not isinstance(frames_count, int):
            if frames_count is not None:
                raise TypeError('frames_count must be int > 0')
        elif frames_count < 1:
            raise TypeError('frames_count must be int > 0')

        must_have_keys = ('start_value', 'step_value',)  # 'per_frame')

        for nr, next_param in enumerate(cb_params_key):
            # print(']]]]] next_param', cb_params_key[next_param])
            next_dict = cb_params_key[next_param]

            if not isinstance(next_dict, dict):
                raise TypeError(f'param_key[{nr}] is not dict (type={type(next_dict)})  value: ({next_dict})')

            for check_key in must_have_keys:
                if check_key not in next_dict:
                    raise KeyError(f'dict cb_params_key[{nr}] has not key {check_key}')

            if 'per_frame' in next_dict:
                if not isinstance(next_dict['per_frame'], int):
                    raise ValueError(f'cb_params_key[{nr}]["per_frame"] must be int > 0 !!!')

        self.call_back = effect_fun_ptr
        key_frames: List[dict] = list()

        for next_name in cb_params_key:
            next_dict: dict = cb_params_key[next_name]
            frame = {
                'param_name': next_name,
                'values': [next_dict['start_value']] * (frames_count + 1)
            }

            step = next_dict.get('per_frame', 1)

            if step == 1:
                sv = next_dict['step_value']
                for frame_index in range(frames_count):
                    frame['values'][frame_index] = frame_index * sv
            else:
                current_value = next_dict['start_value']
                for frame_index in range(1, frames_count + 1):
                    if frame_index % step == 0:
                        current_value += step
                    frame['values'][frame_index - 1] = current_value

            frame['values'] = utils.CircleList(
                list_or_tuple=frame['values'],
                max_iterations=frames_count
            )
            key_frames.append(frame)
        self.key_frames = key_frames
        self.frames_count = frames_count
        self.turned_on = True
        self.last_frame_index = -1
        self.last_frame_rendered = None

        UuidController.set_uid(self)

    def is_turned(self) -> bool:
        return self.turned_on

    def set_turn_state(self, state: bool):
        self.turned_on = state

    def has_more(self) -> bool:  # wtf?
        # return self.last_frame_index + 1 > self.key_frames[0]['values'].has_more()
        return self.last_frame_index > self.frames_count - 1

    def next_frame(self, im1: image_types, frame_index: int = -1, **params) -> dict:

        result = locals()
        index_err = f'frame_index {frame_index} > frames to render ({self.frames_count})'

        if frame_index is None or frame_index == -1:
            self.last_frame_index += 1
            frame_index = self.last_frame_index

        self.last_frame_index = frame_index

        if frame_index > self.frames_count:
            raise IndexError(index_err)

        args_to_pass = {}
        for param_info in self.key_frames:
            args_to_pass = {
                **args_to_pass,
                param_info['param_name']: param_info['values'].get(frame_index)
            }

        result['frame_index'] = frame_index
        result['to_save'] = self.call_back(im1, **args_to_pass)
        self.last_frame_rendered = result['to_save']
        print('KeyedEffect next_frame done.')
        return result

    def last_frame(self):
        return self.last_frame_rendered

    def __str__(self) -> str:
        return f'<KeyedEffect>\n\t{self.__dict__}\n</KeyedEffect>\n'


# UNTESTED
class KeyedEffectsStack(NextFrame):
    """
    Properly the name of this class should consists of something tied with time, time line, video track, or other.
    Main purpose is to use object of it for storing more than one KeyedEffect to overlay multi effects on the frame.
    """

    def __init__(self, *keyed_effects_args: KeyedEffect or KeyedEffectsStack, **kwargs):
        """
        Stores multiple keyed effects on a time line

        :keyword keyed_effects_args:
            variable list that elements should be lists/tuples with len of two:
             - the frame start position where an effect should start
             - the KeyedEffect instance or KeyedEffectStack other instance with position that will affect on moving it
               on the time line

            If the only KeyedEffect instances are passed instead of tuples with them as a second element, by default
            start position is then set to zero.

            If KeyedEffectStack is passed, effects with their start positions are passed with original frame positions

        :keyword kwargs: unused by now
        """
        self.frame_index = -1
        self.frames_length = 0
        self.keyed_effects_stack: List[Tuple[int, KeyedEffect]] = []
        self.add(*keyed_effects_args)
        self.kwargs = kwargs
        self.last_frame_rendered = None

        UuidController.set_uid(some=self)

    def get_keyed_effects_stack(self):
        return self.keyed_effects_stack

    def add(self, *keyed_effects_args: KeyedEffect or KeyedEffectsStack):  # KeyedEffectStack <-> future __annotations__
        """
        Look at __init__ signature
        :param keyed_effects_args:
        :return: self
        """
        expanded = []
        for next_arg in keyed_effects_args:
            if isinstance(next_arg, KeyedEffectsStack):
                expanded.extend(next_arg.get_keyed_effects_stack())
            elif isinstance(next_arg, (list, tuple)):
                pos_add, effect_stack = next_arg
                if isinstance(effect_stack, KeyedEffectsStack):
                    for child_tuple in effect_stack.get_keyed_effects_stack():
                        expanded.append((child_tuple[0] + pos_add, child_tuple[1]))

        for next_arg in list(keyed_effects_args) + expanded:
            if isinstance(next_arg, (list, tuple)):
                if len(next_arg) != 2:
                    raise ValueError
                pos, effect = next_arg
                if not isinstance(pos, int) or not isinstance(effect, KeyedEffect):
                    raise TypeError
                if pos < 0:
                    raise IndexError('KeyedEffectsStack::add:  trying add KeyedEffect that has no frames')
            elif isinstance(next_arg, KeyedEffect):
                next_arg = (0, next_arg)
            else:
                raise AttributeError

            last_frame_pos = next_arg[0] + next_arg[1].frames_count
            if last_frame_pos > self.frames_length:
                self.frames_length = last_frame_pos

            self.keyed_effects_stack.append(next_arg)
            print(f'KeyedEffectsStack::add  -  added {type(next_arg)}')
        print(f'KeyedEffectsStack::add() - max_len is not {self.frames_length}')
        self.frames_length = self.get_frames_length()
        return self

    def get_used_effects_list(self, as_copy=False) -> List[KeyedEffect]:
        if as_copy:
            return [deepcopy(_[1]) for _ in self.keyed_effects_stack]
        return [_[1] for _ in self.keyed_effects_stack]

    def get_frames_length(self):
        return 0 if len(self.keyed_effects_stack) == 0 else max(
            [pos + next_effect.frames_count] for pos, next_effect in self.keyed_effects_stack
        )

    def __len__(self):
        return self.frames_length

    def effects_count(self, **kwargs):
        """
        Tells the number of children effects
        :param kwargs:
            variable keywords.
            :keyword only_turned_on:  by default set to False, otherwise the only effects that turned on are counted
            :keyword only_turned_off: by default set to False, otherwise the only effects that turned on are counted.
                setting both keywords to True return -1 because it is not possible to return at one time two values
        :return: int value
        """
        only_turned_on = kwargs.get('only_turned_on', False)
        only_turned_off = kwargs.get('only_turned_off', False)

        if only_turned_on and only_turned_off:
            # say what?? at one time? -1
            return -1
        elif not only_turned_on and not only_turned_off:
            return len(self.keyed_effects_stack)

        turned_on = len([True for _ in self.keyed_effects_stack if _[1].is_turned()])
        if only_turned_off:
            return abs(len(self.keyed_effects_stack) - turned_on)
        return turned_on

    def __str__(self):
        return f'<KeyedEffectsStack>\n{"".join(str(_) for _ in self.keyed_effects_stack)}\n</KeyedEffectsStack>'

    def has_more(self):
        print('KeyedEffectsStack.has_more():  frame_index', self.frame_index, 'frames_length', self.frames_length)
        return self.frame_index < self.frames_length

    def pick_effects(self, frame_index: int, only_turned_on=False) -> List[Tuple[int, KeyedEffect]]:
        """
        Allows to select effects assigned to frame_index
        :param frame_index: index of frame
        :param only_turned_on: if set to True, selects only effects that state is turned on
        :return: list that elements are tuple of size two: the start position and the effect
        """
        result = []
        if only_turned_on:
            for start_pos, keyed_effect in self.keyed_effects_stack:
                if start_pos < frame_index < keyed_effect.frames_count:
                    if keyed_effect.is_turned():
                        result.append((start_pos, keyed_effect))
        else:
            for start_pos, keyed_effect in self.keyed_effects_stack:
                if start_pos < frame_index < keyed_effect.frames_count:
                    result.append((start_pos, keyed_effect))
        return result

    def next_frame(self, **params) -> dict or RuntimeError or AttributeError:
        print(f'KeyedEffectsStack::next_frame params:  {params.keys()}')
        result_dict = {
            'called': 'KeyedEffectsStack::next_frame',
            'params': params,
        }
        self.frame_index = params.get('frame_index', self.frame_index) + 1
        if self.frame_index > self.frames_length:
            raise IndexError(f'KeyedEffectsStack::next_frame IndexError: frame_index={self.frame_index}')
        # frame_index: int, im1: image_types, im2: image_types,
        image = params.get('im1', params.get('image', None))
        result_dict['passed_image'] = image
        if image is None:
            image = self.pick_effects(self.frame_index)
            if image is None:
                raise RuntimeError(
                    f'KeyedEffectsStack: No parameter im1/image of type [PIL.Image.Image or numpy.ndarray] passed, ' +
                    f' ::select_effects(frame_index={self.frame_index}) returned no image!'
                )
        params['im1'] = image

        for effect_id, (start_pos, keyed_effect) in enumerate(self.pick_effects(frame_index=self.frame_index)):
            params['frame_index'] = self.frame_index - start_pos
            result_dict = keyed_effect.next_frame(**params)
            if isinstance(result_dict, dict):
                r = result_dict.get('to_save', result_dict.get('im1', result_dict.get('image1', None)))
                if r is None:
                    raise ValueError(
                        f'KeyedEffectStack::next_frame:  iterated effect returned {type(r)} (instead of image)'
                    )
                image = r
            else:
                raise TypeError

        result_dict['to_save'] = image
        result_dict['effects_used'] = [_[1].call_back.__name__ for _ in self.keyed_effects_stack]
        self.last_frame_rendered = result_dict['to_save']
        return result_dict

    def last_frame(self) -> image_types:
        return self.last_frame_rendered


class TimelineModelElement:
    ...


class TimelineElement(TimelineModelElement):
    def __init__(self, **kwargs):
        self.start_frame = kwargs.get('start_pos', 0)
        self.frames_duration = kwargs.get('frames_duration', 100)

        not_none_key_name, value = None, None
        for next_key in ('image', 'image_handler', 'image_fun_getter', 'image_cb_getter',):
            if next_key in kwargs:
                not_none_key_name, value = next_key, kwargs[next_key]
                break

        if not_none_key_name is None and value is None:
            raise TypeError

        image = None
        if not_none_key_name == 'image':
            if isinstance(value, (PIL.Image.Image, numpy.ndarray)):
                image = value
        elif not_none_key_name == 'image_handler':
            if hasattr(value, 'get_image'):
                if callable(value.get_image):
                    image = value.get_image()
        elif not_none_key_name == 'image_fun_getter':
            if callable(value):
                image = value()
                if not isinstance(image, (PIL.Image.Image, numpy.ndarray)):
                    image = None
        elif not_none_key_name == 'image_cb_getter':
            if isinstance(value, utils.Callback):
                image = value.__call__()
        else:
            raise TypeError('kwargs must-have-keys error')

        self.image = image
        self.keyed_effect_stack = None

        kes = kwargs.get('keyed_effects_stack', None)
        if kes is not None:
            if isinstance(kes, KeyedEffectsStack):
                self.keyed_effect_stack = kwargs['KeyedEffectsStack']
        else:
            def still_effect():
                return self.image

            self.keyed_effect_stack = KeyedEffectsStack(
                KeyedEffect(effect_fun_ptr=still_effect, frames_count=self.frames_duration)
            )

        _ = self.keyed_effect_stack.get_frames_length()
        if _ > self.frames_duration:
            self.frames_duration = _
        UuidController.set_uid(self)

    def __len__(self):
        return self.keyed_effect_stack.get_frames_length()


class TimelineController:
    def __init__(self, *timeline_elements: TimelineElement, **kwargs):
        self.timeline_elements: List[TimelineElement] = []
        self.fps = kwargs.get('fps', 25)
        self.frame_resolution = kwargs.get('frame_resolution', (1920, 1080))
        self.default_transition_length = kwargs.get('default_transition_length', 100)
        self.transitions_timing: List[Dict] = []
        self.stabilized_transitions = False
        self.add(*timeline_elements)

    def add(self, *timeline_elements: List[int, TimelineElement]):
        used_uids = []

        for _ in timeline_elements:
            if not isinstance(_, dict):
                raise TypeError
            if not isinstance(_, TimelineElement):
                raise TypeError

            uid = getattr(_, 'attached_uid')
            if uid in used_uids:
                raise ValueError
            used_uids.append(uid)

        self.timeline_elements = timeline_elements
        self.transitions = []
        self.stabilize_transitions()

    def __len__(self):
        return max(len(_) for _ in self.timeline_elements)

    def stabilize_transitions(self):
        if self.stabilized_transitions or len(self.timeline_elements) < 2:
            return

        timeline_elements = sorted(self.timeline_elements, key=lambda _: _.start_position)

        diff = len(timeline_elements[0]) - self.default_transition_length

        if diff < self.default_transition_length:
            min_start_pos = len(timeline_elements[0])
        else:
            min_start_pos = diff

        self.transitions_timing = []
        for i in range(1, len(self.default_transition_length)):
            timeline_elements[i].start_frame = min_start_pos

            diff = len(timeline_elements[i]) - self.default_transition_length
            len_e = len(timeline_elements[i])
            if diff < self.default_transition_length:
                min_start_pos += len_e
            else:
                min_start_pos += len_e - self.default_transition_length

            self.transitions_timing.append({
                'start': min_start_pos,
                'length': min_start_pos + self.default_transition_length
            })
        self.timeline_elements = timeline_elements
        self.stabilized_transitions = True

    def swap_elements(self, pos_or_uid_first: int or str, pos_or_uid_second: int or str):
        raise NotImplementedError
