from __future__ import annotations
from copy import deepcopy
from typing import Callable, List, Tuple, Type
from inspect import isclass

import utils
from animations.Transitions_v2 import NextFrame


image_types = utils.image_types


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
        '''if 'effect' not in effect_fun_ptr.__name__:
            raise AttributeError('effect_fun_ptr seems not to be an effect function')'''
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

    def is_turned(self) -> bool:
        return self.turned_on

    def set_turn_state(self, state: bool):
        self.turned_on = state

    def set_uid(self, uid_setter_instance) -> True or ValueError:
        if hasattr(self, '_uid'):
            raise ValueError('uid was already set')
        if hasattr(uid_setter_instance, 'set_next_uid'):
            if callable(uid_setter_instance.set_next_uid):
                setattr(self, '_uid', uid_setter_instance.set_next_uid())
                setattr(self, '_last_uid_setter', uid_setter_instance)
                return True
        raise ValueError('incorrect parameter uid_setter_instance has no function set_next_uid()')

    def has_uid(self) -> bool:
        return hasattr(self, '_uid')

    def get_uid(self) -> str or AttributeError:
        if hasattr(self, '_uid'):
            return getattr(self, '_uid')
        raise AttributeError('uid was not set')

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
    def copy(self) -> KeyedEffect:
        # getattr(self, '_last_uid_setter', uid_setter_instance)  ??
        """return type(self).__init__(
            effect_fun_ptr=self.call_back,
            frames_count=self.frames_count,
            **self.key_frames
        )"""
        return deepcopy(self)

    # UNTESTED
    def to_other_keyed_effect(
            self,
            new_effect_type: Type[KeyedEffect] or KeyedEffect,
            copy_prev_cb_params: bool = True,
            **new_params) -> KeyedEffect:

        # getattr(self, '_last_uid_setter', uid_setter_instance)  ??

        if not isclass(new_effect_type):
            new_effect_type: Type = new_effect_type.__class__

        if issubclass(new_effect_type, KeyedEffect):
            if type(new_effect_type) is type(self):
                return self.copy()
            result = type(self)(
                self.call_back,  # effect_fun_ptr=
                self.frames_count,  # frames_count=
                **new_params
            )
            if copy_prev_cb_params:
                result.key_frames.extend(deepcopy(self.key_frames))
            return result

        raise TypeError

    # @staticmethod
    # UNTESTED !!!
    def from_other(self, other: KeyedEffect) -> KeyedEffect:
        if not isinstance(other, KeyedEffect):
            raise TypeError
        '''if isinstance(other, KeyedEffect):
            for field in 'call_back,frames_count,key_frames,last_frame_index,need_params,turned_on'.split(','):
                setattr(self, field, deepcopy(getattr(other, field)))'''
        self.__dict__ = other.__dict__
        return self


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

    def __repr__(self):
        return str(self)

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


class StillImage(KeyedEffect):
    def __init__(self, frames_count: int, image: Callable or utils.Callback or utils.image_types, **cb_params_key):
        self.image = image

        def image_getter_effect(**kwargs):
            kwargs['to_save'] = kwargs['image'] = self.image
            return kwargs['image']
        super(StillImage, self).__init__(effect_fun_ptr=image_getter_effect, frames_count=frames_count, **cb_params_key)

    def next_frame(self, im1: image_types, frame_index: int = -1, **params) -> dict:
        params = locals()
        if frame_index == -1 or frame_index is None:
            frame_index = self.last_frame_index
        frame_index += 1
        self.last_frame_index = frame_index
        if frame_index > self.frames_count:
            raise IndexError()
        params['to_save'] = self.image if im1 is None else self.image
        if callable(params['to_save']) or isinstance(params['to_save'], utils.Callback):
            params['to_save'] = params['to_save']()
        self.last_frame_rendered = params['to_save']
        return params
