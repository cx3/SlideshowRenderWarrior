import PIL
import numpy as np

from common import utils
from effects import pil_effects
import effects.cv_effects as eff

def show(img):
    cv2.imshow('test', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


print(eff.cv2_get_color_spaces())
exit()

im = utils.load_image('source_images/1.jpg', np.ndarray)
# im = colors.invert_channel(im, 'r')

im = eff.cut_irregular_shape(im, np.array([[(10, 10), (300, 300), (10, 300)]], dtype=np.int32))
show(im)
exit()

im = utils.load_image('source_images/1.jpg', np.ndarray)
im = eff.invert_channel_effect(im, 'r')

exit()

im = eff.apply_color_overlay_effect('source_images/1.jpg', intensity=0.3, red=10, green=100, blue=0)
import cv2

cv2.imshow('name', im)
cv2.waitKey()
cv2.destroyAllWindows()
exit()

import numpy

im = utils.load_image('source_images/1.jpg', PIL.Image.Image)
print('type after loading ', type(im))
print('type after conv    ', type(utils.convert_image_type(im, numpy.ndarray)))  #
exit()

from effects.elasticdeform_effect import elastic_transform

elastic_transform(r'C:\proj\py\rw\source_images\3.jpg', 0.1, 20).show()
exit()

from effects.pixelate_effect import pixelate

pixelate("source_images/1.jpg", 256).show()
exit()

from effects.sketch_effect import sketch_v4

sk = sketch_v4("source_images/1.jpg")
print(sk.show())
exit()

second = utils.CircleList([
    pil_effects.bold_contour_edges,
    pil_effects.sharpen_saturation,
    pil_effects.zoom_and_crop
])

import animations.transitions_classes_OLD as tc

tc.FirstImageHorizontalStretchTransition(
    image1="source_images/1.jpg",
    image2="source_images/3.jpg",
    frames_count=50,
    while_process_second=second
).render()

edges_callback = utils.Callback(
    fun_ptr=pil_effects.bold_contour_edges,
)

circle_list = utils.CircleList(
    list_or_tuple=[
        pil_effects.sharpen_saturation,
        edges_callback,
    ],
    max_iterations=25
)

tc.FirstImageZoomBlendTransition(
    image1="source_images/1.jpg",
    image2="source_images/3.jpg",
    frames_count=25,
    while_process_second=circle_list
).render()

bt = tc.CompositeMaskedSaturationTransition(
    image1="source_images/1.jpg",
    image2="source_images/3.jpg",
    frames_count=25,
    # kwargs:
    sharp_factor=2,
    color_factor=2,
    mask_image="source_images/2.jpg"
)
print(bt.render())





'''travel_error = 'ImageTravel param travel_mode must be None or tuple of 2: (directions[?]) * 2'
travel_ok = True

if travel_mode is not None and len(travel_mode) == 2:
    if travel_mode[0] == travel_mode[1] or \ 
    travel_mode[0] not in directions \
    or travel_mode[1] not in directions:
        travel_ok = False
elif len(travel_mode) != 2:
    travel_ok = False

if not travel_ok:
    raise AttributeError(travel_error)

if travel_mode is None:
    travel_mode = ['lb', 'ru']'''