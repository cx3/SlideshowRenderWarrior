# Python imports
from time import time

# Third party imports

from PIL import Image, ImageEnhance, ImageFilter
from PIL.Image import Image as PillowImage

# Project imports
from utils import _run_predicates, CircleList
from effects import pil_effects


# -+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-


def _preprocess(image1, image2, **kwargs) -> tuple:
    if 'preprocess_first' in kwargs:
        image1 = _run_predicates(image1, kwargs['preprocess_first'])
    if 'preprocess_second' in kwargs:
        image2 = _run_predicates(image2, kwargs['preprocess_second'])
    print('_preprocess returns tuple')
    return image1, image2


def _while_process(image1, image2, **kwargs) -> tuple:
    # print('_while_proces')
    if 'while_process_first' in kwargs:
        image1 = _run_predicates(image1, kwargs['while_process_first'])
    if 'while_process_second' in kwargs:
        image2 = _run_predicates(image2, kwargs['while_process_second'])
    return image1, image2


def _postprocess(image1, image2, **kwargs) -> tuple:
    if 'postprocess_first' in kwargs:
        image1 = _run_predicates(image1, kwargs['postprocess_first'])
    if 'postprocess_second' in kwargs:
        image2 = _run_predicates(image2, kwargs['postprocess_second'])
    return image1, image2


# -+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-


class Transition:
    def __init__(
            self,
            image1: str or PillowImage,
            image2: str or PillowImage,
            dest_dir: str = 'rendered',
            frames_count=50,
            # prefix="AbstractTransition",
            **kwargs
    ):  # -~/*^/-_/~-~/*^/-_/~-~/*^/-_/~-~/*^/-_/~-~/*^/-_/~-~/*^/-_/~-~/*^/-_/~-~/*^/-_/~-~/*^/-_/~-~/*^/-_/~-~/*^-.

        if isinstance(image1, PillowImage):
            self.image1 = image1
        elif isinstance(image1, str):
            self.image1 = Image.open(image1)

        if isinstance(image2, PillowImage):
            self.image2 = image2
        elif isinstance(image2, str):
            self.image2 = Image.open(image2)

        if 'prefix' not in kwargs:
            class_name = str(self.__class__).split("'")[1][:-2]
            if '.' in class_name:
                self.prefix = class_name.split('.')[-1]
        else:
            self.prefix = kwargs['prefix']

        self.dest_dir = dest_dir
        self.frames_count = frames_count
        self.kwargs = kwargs
        self.name_counter = 0
        self.im1, self.im2 = self.image1, self.image2

    def rendering_process(self) -> dict:
        """Override this method in children, but do not invoke it directly. Look at method render() """
        return {
            'effect': 'Abstract Transition',
            'time': 0.,
            'names': [],
            'kwargs': self.kwargs,
        }

    def name(self, extension="jpg") -> str:
        """
        Creates next frame name using fields dest_dir, prefix, name_counter, extension
        :param extension:
        :return: path to rendered frame
        """
        self.name_counter += 1
        return f"{self.dest_dir}/{self.prefix}{self.name_counter}.{extension}"

    def render(self) -> dict:
        """Invoke this method on children for rendering frames in timeline controller"""
        start = time()
        self.im1, self.im2 = _preprocess(self.image1, self.image2, **self.kwargs)
        result_dict = self.rendering_process()
        self.im1, self.im2 = _postprocess(self.im1, result_dict['names'][-1], **self.kwargs)
        result_dict['time'] = time() - start
        return result_dict


class BlendingTransition(Transition):
    def rendering_process(self) -> dict:
        step = 1/self.frames_count
        names = []
        im1, im2 = self.im1, self.im2
        for i in range(1, self.frames_count):
            im1, im2 = _while_process(im1, im2, **self.kwargs)
            blend = Image.blend(im1, im2, i*step)
            names.append(self.name())
            blend.save(names[-1])

        '''im2 = _postprocess(im1, Image.blend(im1, im2, 1), **self.kwargs)[1]
        im2.save(self.name())'''

        return {
            'effect': 'BlendingTransition',
            'names': names,
            'kwargs': self.kwargs,
        }


class BlendingTransition2(Transition):
    def rendering_process(self) -> dict:
        # self.prefix = 'bt2_'
        step = 1/self.frames_count
        names = []
        contrast = ImageEnhance.Contrast(self.im2)
        for i in range(1, self.frames_count + 1):
            enhanced = contrast.enhance(step * i * 4)
            im1, enhanced = _while_process(self.im1, enhanced, **self.kwargs)
            blend = Image.blend(im1, enhanced, i*step)
            names.append(self.name())
            blend.save(names[-1])

        return {
            'effect': 'BlendingTransition2',
            'names': names,
            'kwargs': self.kwargs,
        }


class BlendingTransition3(Transition):
    def rendering_process(self):
        # self.prefix = 'bt3_'
        step = 1/self.frames_count
        colorize = ImageEnhance.Color(self.im2)

        y = [(2 * i * i + 4) / self.frames_count for i in range(self.frames_count//2)]

        if self.frames_count % 2 == 1:
            y = y + [y[-1]] + y[::-1]
        else:
            y = y + y[::-1]
        names = []

        for i in range(1, self.frames_count + 1):
            enhanced = colorize.enhance(step * i * y[i-1])
            im1, enhanced = _while_process(self.im1, enhanced, **self.kwargs)
            blend = Image.blend(im1, enhanced, i*step)
            names.append(self.name())
            blend.save(names[-1])

        return {
            'effect': 'BlendingTransition3',
            'names': names,
            'kwargs': self.kwargs,
        }


class BlendingContrastTransition(Transition):
    def rendering_process(self) -> dict:
        # self.prefix = 'bct_'
        step = 1/self.frames_count

        y = [(4 * i * i - 8) / self.frames_count for i in range(self.frames_count//2)]
        if self.frames_count % 2 == 1:
            y = y + [y[-1], y[-1]] + y[::-1]
        else:
            y = y + [y[-1]] + y[::-1]
        names = []

        for i in range(1, self.frames_count):
            level = 0.01 if y[i] == 0 else y[i]
            print('level', level)
            enhanced = ImageEnhance.Contrast(self.im2).enhance(level)
            im1, enhanced = _while_process(self.im1, enhanced, **self.kwargs)
            blend = Image.blend(im1, enhanced, i*step)
            names.append(self.name())
            blend.save(names[-1])

        self.im2.save(self.name())

        return {
            'effect': 'ContrastTransition',
            'names': names,
            'kwargs': self.kwargs,
        }


class BlendingSharpTransition(Transition):
    def rendering_process(self) -> dict:
        # self.prefix = 'bst_'
        step = 1 / self.frames_count

        y = [step * i * 5 for i in range(self.frames_count // 2)]
        if self.frames_count % 2 == 0:
            y = y + [y[-1]] + y[::-1]
        else:
            y = y + [y[-1], y[-1]] + y[::-1]

        names = []

        for i in range(1, self.frames_count + 1):
            sharp = ImageEnhance.Sharpness(self.im2).enhance(step * y[i])
            im1, sharp = _while_process(self.im1, sharp, **self.kwargs)
            blend = Image.blend(im1, sharp, i * step)
            names.append(self.name())
            blend.save(names[-1])

        return {
            'effect': 'blending_sharp_transition',
            'names': names,
            'kwargs': self.kwargs,
        }


class BlendingSharpTransition2(Transition):
    def rendering_process(self) -> dict:
        # self.prefix = 'bst2_'
        step = 1 / self.frames_count

        y = [step * i * 10 for i in range(self.frames_count // 2)]
        if self.frames_count % 2 == 1:
            y = y + [y[-1], y[-1]] + y[::-1]
        else:
            y = y + [y[-1]] + y[::-1]
        names = []

        for i in range(1, self.frames_count + 1):
            sharp = ImageEnhance.Sharpness(self.im2).enhance(step * y[i])
            im1, sharp = _while_process(self.im1, sharp, **self.kwargs)
            blend = Image.blend(im1, sharp, i * step)
            names.append(self.name())
            blend.save(names[-1])

        return {
            'effect': 'BlendingSharpTransition2',
            'names': names,
            'kwargs': self.kwargs,
        }


class BlendingEdgesTransition(Transition):
    def rendering_process(self) -> dict:
        # self.prefix = 'bet_'
        step = 1 / self.frames_count
        edge1 = self.im1
        names = []

        for i in range(1, self.frames_count + 1):
            if i > 0:
                if i % 4 == 0:
                    edge1 = edge1.filter(ImageFilter.EDGE_ENHANCE)
            edge1, im2 = _while_process(edge1, self.im2, **self.kwargs)
            blend = Image.blend(edge1, im2, i * step)
            names.append(self.name())
            blend.save(names[-1])

        return {
            'effect': 'blending_edges_transition',
            'names': names,
            'kwargs': self.kwargs,
        }


class BlendingZoomPanTransition(Transition):
    # NE FUNGUJE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def rendering_process(self) -> dict:
        # self.prefix = 'bzpt_'
        step = 1 / self.frames_count
        edge1 = self.im1
        names = []

        for i in range(1, self.frames_count + 1):
            if i > 0:
                if i % 4 == 0:
                    edge1 = edge1.filter(ImageFilter.EDGE_ENHANCE)
            edge1, im2 = _while_process(edge1, self.im2, **self.kwargs)
            blend = Image.blend(edge1, im2, i * step)
            names.append(self.name())
            blend.save(names[-1])

        return {
            'effect': 'blending_zoom_pan_transition',
            'names': names,
            'kwargs': self.kwargs,
        }


class CompositeBasicTransition(Transition):
    def rendering_process(self) -> dict:
        # self.prefix = 'cbt_'

        step = 1 / self.frames_count
        names = []

        im1, im2 = _while_process(self.im1, self.im2, **self.kwargs)
        names.append(self.name())
        # mask = image_manip.common_mask(im1, im2, 0)
        # Image.composite(im1, im2, mask).save(names[-1])
        im1.save(names[0])

        for i in range(2, self.frames_count-1):
            im1, im2 = _while_process(im1, im2, **self.kwargs)
            names.append(self.name())
            mask = pil_effects.common_mask(im1, im2, alpha=i * step)
            res = Image.composite(im1, im2, mask)
            res.save(names[-1])

        im1, im2 = _postprocess(im1, im2, **self.kwargs)
        names.append(self.name())
        mask = pil_effects.common_mask(im1, im2, alpha=1 * step)
        res = Image.composite(im1, im2, mask)
        res.save(names[-1])

        return {
            'effect': 'composite_basic_transition',
            'names': names,
            'kwargs': self.kwargs,
        }


class CompositeSaturationTransition(Transition):
    def rendering_process(self) -> dict:
        def predicate(image):
            return pil_effects.sharpen_saturation(image, sharp_factor=3, color_factor=3)

        # self.prefix = 'cst_'
        cbt = CompositeBasicTransition(
            self.im1, self.im2, self.dest_dir, self.frames_count, prefix=self.prefix,
            # kwargs:
            while_process_second=predicate
        )
        result_dict = cbt.render()

        return {
            'effect': 'CompositeSaturationTransition',
            'names': result_dict['names'],
            'kwargs': self.kwargs,
        }


class CompositeMaskedSaturationTransition(Transition):
    """
    pass kwargs to __init__
     - sharp_factor: int
     - color_factor: int
    """
    def rendering_process(self) -> dict:
        self.prefix = 'cmst_'
        mask_image = self.kwargs['mask_image']
        if isinstance(mask_image, str):
            mask_image: PillowImage = Image.open(mask_image)
        mask_image = mask_image.convert('L')

        sharp_factor = 1 if 'sharp_factor' not in self.kwargs else self.kwargs['sharp_factor']
        color_factor = 1 if 'color_factor' not in self.kwargs else self.kwargs['color_factor']

        im1, im2 = self.im1, self.im2
        names = []

        for i in range(1, self.frames_count):
            im2 = pil_effects.sharpen_saturation(im2, sharp_factor=sharp_factor, color_factor=color_factor)
            im2 = Image.composite(self.im1, im2, mask_image)

            im1, im2 = _while_process(im1, im2, **self.kwargs)

            names.append(self.name())
            im2.save(names[-1])

        return {
            'effect': 'CompositeSaturationTransition',
            'names': names,
            'kwargs': self.kwargs,
        }


class FirstImageZoomBlendTransition(Transition):
    def rendering_process(self) -> dict:
        # self.prefix = 'fizbt_'
        w, h = self.im1.size
        names = []

        for i in range(1, self.frames_count):
            self.im2 = pil_effects.zoom_and_crop(self.im2, 101, w, h)
            self.im1, self.im2 = _while_process(self.im1, self.im2, **self.kwargs)
            names.append(self.name())
            Image.blend(self.im1, self.im2, 0.5).save(names[-1])

        return {
            'effect': 'FirstImageZoomBlendTransition',
            'names': names,
            'kwargs': self.kwargs,
        }


class FirstImageHorizontalStretchTransition(Transition):
    def rendering_process(self) -> dict:

        def _stretch(image):
            width = 110 if 'width' not in self.kwargs else self.kwargs['width']
            return pil_effects.stretch_image_keep_prev_size(image, width, 100)

        circle_list = CircleList([_stretch])
        # self.prefix = 'fist_'  # fisting ;p
        names = []

        for task_type in ('preprocess_first', 'while_process_first', 'postprocess_first'):
            self.kwargs[task_type] = circle_list

        for i in range(1, self.frames_count + 1):
            self.im1, self.im2 = _while_process(self.im1, self.im2, **self.kwargs)
            names.append(self.name())
            self.im1, self.im2 = pil_effects.match_images_format(self.im1, self.im2, 'RGBA')
            print(self.im1.size, self.im2.size)
            b = Image.blend(self.im1, self.im2, 0.5)
            b.convert('RGB').save(names[-1])

        return {
            'effect': 'FirstImageHorizontalStretchTransition',
            'names': names,
            'kwargs': self.kwargs,
        }


class Travel:pass