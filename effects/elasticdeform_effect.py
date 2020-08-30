import elasticdeform
import numpy, imageio
import PIL.Image as Image

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def elastic_transform(image: str or Image.Image, alpha, sigma, random_state=None) -> Image.Image:
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    if isinstance(image, str):
        image = Image.open(image)

    image = numpy.array(image)

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    print(x.shape)
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return Image.fromarray(distored_image.reshape(image.shape))


def test2():
    X = numpy.array(Image.open(r'C:\proj\py\rw\source_images\3.jpg'))
    X[::10, ::10] = 1

    # apply deformation with a random 3 x 3 grid
    X_deformed = elasticdeform.deform_random_grid(X, sigma=25, points=3)

    Image.fromarray(X_deformed).show()

