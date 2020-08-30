
class WizardSlideshow(TimelineModel):
    def __init__(
            self,
            image_sources: List[utils.image_types],
            transitions_list: List[Type[trans.Transition]] or None = None,
            frame_resolution=(1920, 1080),
            total_slide_duration=750,
            **params):

        """
        Wizard slideshow
        :param image_sources:
        :param transitions_list:
        :param frame_resolution:
        :param total_slide_duration:
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
                transitions_list = transitions_list[:len(image_sources)-1]
            else:
                transitions_list = [transitions_list[0]]

        super().__init__(**{k: v for k, v in locals().items() if k != 'self'})

        self.frame_resolution = frame_resolution
        self.global_frame_index = -1

        fft = params.get('frame_file_type', 'jpg')
        fft = 'jpg' if fft not in ['jpg', 'png'] else fft

        self.frame_file_type = fft
        self.image_handlers: List[ImageHandler] = []

        step = self.one_slide_len // len(self.image_sources)
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

        # TIMELINE CONTROLLER
        self.slide_frames = ClipUtils.KeyedEffectsStack(
            ClipUtils.StillImage(
                frames_count=self.total_slide_duration,
                image=image_picker_cb
            ),
        )

        # position of start, end, selected transition type
        self.slide_transitions_timing: List[Dict] = []

        trans_len = len(self.transitions_list)
        self.one_trans_len = (params['slide_seconds'] * self.fps) // 4
        self.one_slide_len = self.total_slide_duration // len(self.image_sources)

        for nr, next_trans in enumerate(self.transitions_list):
            start = (nr + 1) * (self.one_slide_len - self.one_trans_len)
            self.slide_transitions_timing.append({
                'nr': nr,
                'start': start,
                'stop': start + self.one_trans_len,
                'transition': next_trans
            })
            print(f'***]]] added {nr} -> {next_trans}')

        for stt in self.slide_transitions_timing:
            print('>> stt: ', stt)

        print('\nWIZARD __init__ LOCALS()')
        for k, v in locals().items():
            print('\t' + k, ' -> ', v)
        print('WIZARD __init__ ENDS HERE\n')

    def has_more(self):
        return self.global_frame_index < self.total_slide_duration

    def pick_image_for_frame(self, frame_index: int = -1) -> utils.image_types:
        if frame_index == -1:
            # print(f'Wizard::pick_image() for frame {frame_index} ==> (default) = {self.global_frame_index}')
            frame_index = self.global_frame_index
        closest,  result_handler_id = self.total_slide_duration, None
        for handler_id, image_handler in enumerate(self.image_handlers):
            indexes = image_handler.get_frame_indexes()

            if frame_index >= indexes[0]:
                dist = frame_index - indexes[0]
                if dist < closest:
                    closest = dist
                    result_handler_id = handler_id
        if result_handler_id is None:
            raise ValueError('No such image')
        result = self.image_handlers[result_handler_id].get_image()
        print(f'picked image.index = {result_handler_id}, with shape:', utils.get_image_size(result))
        return cv2.resize(utils.convert_image_type(result, numpy.ndarray), self.frame_resolution)

    def while_transition(self) -> bool or dict:
        if len(self.transitions_list) == 0:
            return False
        for next_dict in self.slide_transitions_timing:
            next_dict: dict
            if next_dict['start'] <= self.global_frame_index < next_dict['stop']:
                return next_dict
        return False

    def render_project(self):
        print('\n\nWIZARD SLIDESHOW :: RENDER_PROJECT() \n')
        print('frames to render: ', self.total_slide_duration, '\t\thas_more', self.slide_frames.has_more())

        transition_frame_pos = -1
        current_transition_instance: trans.Transition or None = None

        print('total_slide_duration: ', self.total_slide_duration)

        for current_frame_index in range(self.total_slide_duration):
            self.global_frame_index = current_frame_index
            print('\n[ ' + '$$$' * 33 + ' ]')
            print('RENDER_PROJECT ITERATION global_frame_index: ', self.global_frame_index)
            rendered_frame_dict = self.slide_frames.next_frame(
                im1=self.pick_image_for_frame(current_frame_index)
            )

            # Assign transition if global_frame_index is in transition timing
            selected_trans = self.while_transition()
            if isinstance(selected_trans, dict):
                selected_trans: dict
                transition_frame_pos += 1

                # current_image_index = self.current_frame_image(self.global_frame_index)
                # print('########### current_image_index', current_image_index)
                if current_transition_instance is None:
                    print('\n' + '$%$^&' * 10 + '\n\tINSTANTIATING TRANSITION', current_transition_instance)
                    current_transition_instance = selected_trans['transition'](
                        image1=utils.verify_alpha_channel(rendered_frame_dict['to_save']),
                        image2=utils.verify_alpha_channel(self.image_handlers[selected_trans['nr'] + 1].get_image()),
                        frames_count=abs(selected_trans['stop'] - selected_trans['start'])
                    )

                if hasattr(current_transition_instance, 'next_frame'):
                    print('\t>>>>>>>>>>>\tTRANSITING STARTS HERE')
                    rendered_frame_dict = current_transition_instance.next_frame(
                        im1=utils.verify_alpha_channel(self.image_handlers[selected_trans['nr']].get_image()),
                        im2=utils.verify_alpha_channel(self.image_handlers[selected_trans['nr'] + 1].get_image()),
                    )
                    self.image_handlers[selected_trans['nr']].set_image(rendered_frame_dict['to_save'])
                    print('\t>>>>>>>>>>>\tTRANSITING ENDS HERE\n')
                else:
                    raise AttributeError(str(type(current_transition_instance)))

            else:
                current_transition_instance = None
                transition_frame_pos = -1

            utils.save_image(rendered_frame_dict['to_save'], f'rendered/{self.global_frame_index}.jpg')
            print('=' * 44 + ' RENDER_FRAME_ITERATION FINISHED ' + '=' * 44)
