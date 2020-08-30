# Python imports
from time import time

# Third party imports
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


PillowImage = Image.Image


def load_image(image_or_path) -> PillowImage:
    if isinstance(image_or_path, str):
        return Image.open(image_or_path)
    if isinstance(image_or_path, PillowImage):
        return image_or_path
    raise TypeError


filters_available = {
    'blur': ImageFilter.BLUR,
    'contour': ImageFilter.CONTOUR,
    'detail': ImageFilter.DETAIL,
    'edge_enhance': ImageFilter.EDGE_ENHANCE,
    'edge_enhance_more': ImageFilter.EDGE_ENHANCE_MORE,
    'emboss': ImageFilter.EMBOSS,
    'find_edges': ImageFilter.FIND_EDGES,
    'smooth': ImageFilter.SMOOTH,
    'smooth_more': ImageFilter.SMOOTH_MORE,
    'sharpen': ImageFilter.SHARPEN
}


class SafeZonedImage:
    def __init__(self, image: str or PillowImage):
        if isinstance(image, str):
            image = Image.open(image)
        if not isinstance(image, PillowImage):
            raise TypeError
        self.image = image
        width, height = self.image.size

        self.big_square = Image.new('RGB', (width * 2, height * 2))
        for coord in [(0, 0), (width, 0), (0, height), (width, height)]:
            self.big_square.paste(self.image, coord)

    def size(self) -> tuple:
        return self.image.size

    def safe_crop(self, x1, y1) -> PillowImage:
        width, height = [abs(_) for _ in self.image.size]

        if x1 > width:
            x1 /= width
        if y1 > height:
            y1 /= height
        return self.big_square.crop((x1, y1, x1 + width, y1 + height))

    def unsafe_crop(self, box) -> PillowImage:
        return self.big_square.crop(box)


def zoom_and_crop(image: PillowImage or str, percent=101, width=None, height=None) -> PillowImage:
    image = load_image(image)
    prev_x, prev_y = image.size
    new_x, new_y = int(prev_x * percent / 100), int(prev_y * percent / 100)
    avg_diff_x, avg_diff_y = abs(prev_x - new_x) // 2, abs(prev_y - new_y) // 2
    resized = ImageOps.fit(image, (new_x, new_y), Image.ANTIALIAS)
    resized = resized.crop((avg_diff_x, avg_diff_y, new_x - avg_diff_x, new_y - avg_diff_y))
    # print('res size', resized.size)
    if width is None and height is None:
        return resized
    if isinstance(width, int) and isinstance(height, int):
        return resized.resize((width, height), Image.ANTIALIAS)
    raise AttributeError


def common_mask(image1: str or PillowImage, image2: str or PillowImage, **kwargs) -> PillowImage:
    image1 = load_image(image1).convert('1').convert('RGB')
    image2 = load_image(image2).convert('1').convert('RGB')
    alpha = 0.5 if 'alpha' not in kwargs else kwargs['alpha']

    if 'preprocess_first' in kwargs:
        for func in kwargs['preprocess_first']:
            image1 = func(image1)

    if 'preprocess_second' in kwargs:
        for func in kwargs['preprocess_second']:
            image2 = func(image2)

    result = Image.blend(image1, image2, alpha)
    return result.convert('1')


def match_images_format(im1: PillowImage, im2: PillowImage, image_format='RGB') -> tuple:
    if not isinstance(im1, PillowImage) or not isinstance(im2, PillowImage):
        raise TypeError()
    return im1.convert(image_format), im2.convert(image_format)


def sharpen_saturation(image: str or PillowImage, **kwargs) -> PillowImage:
    """
    :param image: image to process on
    :param kwargs: should contain sharp_factor: int, color_factor: int
    :return:
    """
    if isinstance(image, str):
        print('sharpen saturation isinstance')
        image = Image.open(image).convert('RGB')
    print('sharpen saturation rest ofcode')

    sharp_factor = 3 if 'sharp_factor' not in kwargs else kwargs['sharp_factor']
    color_factor = 3 if 'color_factor' not in kwargs else kwargs['color_factor']

    image = ImageEnhance.Sharpness(image).enhance(sharp_factor)
    image = ImageEnhance.Color(image).enhance(color_factor)

    return image


def bold_contour_edges(image: str or PillowImage, **kwargs) -> PillowImage:
    """
    :param image: path to image or instance of PillowImage
    :param kwargs: as_mask=True gives black-white image
    :return: PillowImage instance
    """
    if isinstance(image, str):
        image: PillowImage = Image.open(image).convert('RGB')

    edge_enhance = 1 if 'edge_enhance' not in kwargs else kwargs['edge_enhance']

    for i in range(edge_enhance):
        image = image.filter(ImageFilter.EDGE_ENHANCE)
    if 'as_mask' in kwargs:
        if kwargs['as_mask'] is True:
            return image.convert('L', colors=2)
    return image


def use_image_filter(image: str or PillowImage, pillow_filter_name: str):
    """
    :param image:
    :param pillow_filter_name: blur, contour, detail, edge_enhance, edge_enhance_more, emboss, find_edges, smooth,
        smooth_more, sharpen
    :return: image with assigned filter
    """

    pillow_filter_name = pillow_filter_name.lower().strip()
    image = load_image(image)

    if pillow_filter_name not in filters_available:
        raise ValueError(f"Unknown filter type {pillow_filter_name}")
    return image.filter(filters_available[pillow_filter_name])


def use_image_filters(image: str or PillowImage, **kwargs) -> PillowImage:
    if isinstance(image, str):
        image: PillowImage = Image.open(image)

    for k, v in kwargs.items():
        for i in range(v):
            image = use_image_filter(image, k.lower().strip())
    return image


def stretch_image_keep_prev_size(image: str or PillowImage, width_precent=105, height_precent=105) -> PillowImage:
    image = load_image(image)

    prev_x, prev_y = image.size
    new_x, new_y = int(prev_x * width_precent / 100), int(prev_y * height_precent / 100)

    avg_diff_x, avg_diff_y = abs(prev_x - new_x) // 2, abs(prev_y - new_y) // 2
    resized = ImageOps.fit(image, (new_x, new_y), Image.ANTIALIAS)
    resized = resized.crop((avg_diff_x, avg_diff_y, new_x - avg_diff_x, new_y - avg_diff_y))

    return resized.resize((prev_x, prev_y), Image.ANTIALIAS)


def posterize(image: str or PillowImage, bits=7) -> PillowImage:
    """
    Posterize image
    :param image: path or Image.Image instance
    :param bits: number 1..8
    :return:
    """
    return ImageOps.posterize(load_image(image), bits)
    