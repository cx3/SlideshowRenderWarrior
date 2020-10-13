# Python imports
from time import time

# Third party imports
from PIL import Image, ImageEnhance, ImageFilter

# Project imports
from common.utils import _run_predicates
from effects import pil_effects


def signature_example(image1: str, image2: str, frames_count=50, prefix="example_", **kwargs) -> dict:
    start_time = time()
    im1, im2 = _preprocess(Image.open(image1), Image.open(image2), **kwargs)

    step = 1 / frames_count
    names = []
    for i in range(1, frames_count):
        im1, im2 = _while_process(im1, im2, **kwargs)
        names.append(f'{prefix}_signature_{i}')
        im2.save(names[-1])

    im2 = _postprocess(im1, im2, **kwargs)[1]
    names.append(f'{prefix}_signature_{frames_count}')
    im2.save(names[-1])

    return {
        'effect': 'EXAMPLE',
        'time': time() - start_time,
        'names': names,
        'kwargs': kwargs,
    }


# -+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-+*^-


def _preprocess(image1, image2, **kwargs) -> tuple:
    if 'preprocess_first' in kwargs:
        image1 = _run_predicates(image1, kwargs['preprocess_first'])
    if 'preprocess_second' in kwargs:
        image2 = _run_predicates(image2, kwargs['preprocess_second'])
    print('_preprocess returns tuple')
    return image1, image2


def _while_process(image1, image2, **kwargs) -> tuple:
    print('_while_proces')
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


def blending_transition(image1: str, image2: str, frames_count=50, prefix="blending_transition_", **kwargs) -> dict:
    start_time = time()
    im1, im2 = _preprocess(Image.open(image1), Image.open(image2), **kwargs)

    step = 1/frames_count
    names = []
    for i in range(1, frames_count):
        im1, im2 = _while_process(im1, im2, **kwargs)
        blend = Image.blend(im1, im2, i*step)
        names.append(f"rendered/{prefix}{i}.jpg")
        blend.save(names[-1])

    im2 = _postprocess(im1, Image.blend(im1, im2, 1), **kwargs)[1]
    im2.save(f"rendered/{prefix}{frames_count}.jpg")

    return {
        'effect': 'blending_transition',
        'time': time() - start_time,
        'names': names,
        'kwargs': kwargs,
    }


def blending_transition2(image1: str, image2: str, frames_count=50, prefix="blending_transition2_", **kwargs) -> dict:
    start_time = time()
    im1, im2 = _preprocess(Image.open(image1), Image.open(image2), **kwargs)
    step = 1/frames_count
    names = []
    contraster = ImageEnhance.Contrast(im2)
    for i in range(1, frames_count + 1):
        enhanced = contraster.enhance(step * i * 4)
        im1, enhanced = _while_process(im1, enhanced, **kwargs)
        blend = Image.blend(im1, enhanced, i*step)
        names.append(f"rendered/{prefix}{i}.jpg")
        blend.save(names[-1])

    return {
        'effect': 'blending_transition2',
        'time': time() - start_time,
        'names': names,
        'kwargs': kwargs,
    }


def blending_transition3(image1: str, image2: str, frames_count=50, prefix="blending_transition3_", **kwargs) -> dict:
    start_time = time()
    im1, im2 = _preprocess(Image.open(image1), Image.open(image2), **kwargs)
    step = 1/frames_count
    contraster = ImageEnhance.Color(im2)

    y = [(2 * i * i + 4) / frames_count for i in range(frames_count//2)]
    y = y + y[::-1]
    names = []

    for i in range(1, frames_count + 1):
        enhanced = contraster.enhance(step * i * y[i-1])
        im1, enhanced = _while_process(im1, enhanced, **kwargs)
        blend = Image.blend(im1, enhanced, i*step)
        names.append(f"rendered/{prefix}{i}.jpg")
        blend.save(names[-1])

    return {
        'effect': 'blending_transition3',
        'time': time() - start_time,
        'names': names,
        'kwargs': kwargs,
    }


def blending_contrast_transition(image1: str, image2: str, frames_count=50, prefix="bct_", **kwargs) -> dict:
    start_time = time()
    im1, im2 = _preprocess(Image.open(image1), Image.open(image2), **kwargs)
    step = 1/frames_count

    y = [(4 * i * i - 8) / frames_count for i in range(frames_count//2)]
    y = y + [y[-1]] + y[::-1]
    names = []

    for i in range(1, frames_count + 1):
        level = 0.01 if y[i] == 0 else y[i]
        print('level', level)
        enhanced = ImageEnhance.Contrast(im2).enhance(level)
        im1, enhanced = _while_process(im1, enhanced, **kwargs)
        blend = Image.blend(im1, enhanced, i*step)
        names.append(f"rendered/{prefix}{i}.jpg")
        blend.save(names[-1])

    return {
        'effect': 'contrast_transition',
        'time': time() - start_time,
        'names': names,
        'kwargs': kwargs,
    }


def blending_sharp_transition(image1: str, image2: str, frames_count=50, prefix="bsp_", **kwargs) -> dict:
    start_time = time()
    im1, im2 = _preprocess(Image.open(image1), Image.open(image2), **kwargs)
    step = 1 / frames_count

    y = [step * i * 5 for i in range(frames_count // 2)]
    y = y + [y[-1]] + y[::-1]
    names = []

    for i in range(1, frames_count + 1):
        sharp = ImageEnhance.Sharpness(im2).enhance(step * y[i])
        im1, sharp = _while_process(im1, sharp, **kwargs)
        blend = Image.blend(im1, sharp, i * step)
        names.append(f"rendered/{prefix}{i}.jpg")
        blend.save(names[-1])

    return {
        'effect': 'blending_sharp_transition',
        'time': time() - start_time,
        'names': names,
        'kwargs': kwargs,
    }


def blending_sharp_transition2(image1: str, image2: str, frames_count=50, prefix="bsp_", **kwargs) -> dict:
    start_time = time()
    im1, im2 = _preprocess(Image.open(image1), Image.open(image2), **kwargs)
    step = 1 / frames_count

    y = [step * i * 10 for i in range(frames_count // 2)]
    y = y + [y[-1]] + y[::-1]
    names = []

    for i in range(1, frames_count + 1):
        sharp = ImageEnhance.Sharpness(im2).enhance(step * y[i])
        im1, sharp = _while_process(im1, sharp, **kwargs)
        blend = Image.blend(im1, sharp, i * step)
        names.append(f"rendered/{prefix}{i}.jpg")
        blend.save(names[-1])

    return {
        'effect': 'blending_sharp_transition2',
        'time': time() - start_time,
        'names': names,
        'kwargs': kwargs,
    }


def blending_edges_transition(image1: str, image2: str, frames_count=50, prefix="bet_", **kwargs) -> dict:
    start_time = time()
    im1, im2 = _preprocess(Image.open(image1), Image.open(image2), **kwargs)
    step = 1 / frames_count
    edge1 = im1
    names = []

    for i in range(1, frames_count + 1):
        if i > 0:
            if i % 4 == 0:
                edge1 = edge1.filter(ImageFilter.EDGE_ENHANCE)
        edge1, im2 = _while_process(edge1, im2, **kwargs)
        blend = Image.blend(edge1, im2, i * step)
        names.append(f"rendered/{prefix}{i}.jpg")
        blend.save(names[-1])

    return {
        'effect': 'blending_edges_transition',
        'time': time() - start_time,
        'names': names,
        'kwargs': kwargs,
    }


def blending_zoom_pan_transition(image1: str, image2: str, frames_count=50, prefix="bzpt_", **kwargs) -> dict:
    start_time = time()
    im1, im2 = _preprocess(Image.open(image1), Image.open(image2), **kwargs)
    step = 1 / frames_count
    edge1 = im1
    names = []

    for i in range(1, frames_count + 1):
        if i > 0:
            if i % 4 == 0:
                edge1 = edge1.filter(ImageFilter.EDGE_ENHANCE)
        edge1, im2 = _while_process(edge1, im2, **kwargs)
        blend = Image.blend(edge1, im2, i * step)
        names.append(f"rendered/{prefix}{i}.jpg")
        blend.save(names[-1])

    return {
        'effect': 'blending_zoom_pan_transition',
        'time': time() - start_time,
        'names': names,
        'kwargs': kwargs,
    }


def composite_basic_transition(image1: str, image2: str, frames_count=50, prefix="cbt_", **kwargs) -> dict:
    start_time = time()

    im1, im2 = _preprocess(Image.open(image1), Image.open(image2), **kwargs)
    step = 1 / frames_count
    names = []

    im1, im2 = _while_process(im1, im2, **kwargs)
    names.append(f'rendered/{prefix}{1}.jpg')
    # mask = image_manip.common_mask(im1, im2, 0)
    # Image.composite(im1, im2, mask).save(names[-1])
    im1.save(names[0])

    for i in range(2, frames_count-1):
        im1, im2 = _while_process(im1, im2, **kwargs)
        names.append(f'rendered/{prefix}{i}.jpg')
        mask = pil_effects.common_mask(im1, im2, alpha=i * step)
        res = Image.composite(im1, im2, mask)
        res.save(names[-1])

    im1, im2 = _postprocess(im1, im2, **kwargs)
    names.append(f"rendered/{prefix}{frames_count}.jpg")
    mask = pil_effects.common_mask(im1, im2, 1)
    res = Image.composite(im1, im2, mask)
    res.save(names[-1])

    return {
        'effect': 'composite_basic_transition',
        'time': time() - start_time,
        'names': names,
        'kwargs': kwargs,
    }


def composite_saturation_transition(image1: str, image2: str, frames_count=50, prefix="cst_", **kwargs) -> dict:
    start_time = time()
    im1, im2 = _preprocess(Image.open(image1), Image.open(image2), **kwargs)

    def predicate(image):
        return pil_effects.sharpen_saturation(image, sharp_factor=3, color_factor=3)

    result_dict = composite_basic_transition(
        im1, im2, frames_count, prefix,
        # kwargs:
        while_process_second=predicate
    )

    return {
        'effect': 'composite_saturation_transition',
        'time': time() - start_time,
        'names': result_dict['names'],
        'kwargs': kwargs,
    }
