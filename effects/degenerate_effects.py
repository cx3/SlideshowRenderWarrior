import numpy as np
import matplotlib.pylab as plt
from PIL import Image


def do_normalise(im):
    return -np.log(1 / ((1 + im) / 257) - 1)


def undo_normalise(im):
    return (1 + 1 / (np.exp(-im) + 1) * 257).astype("uint8")


def rotation_matrix(theta):
    """
    3D rotation matrix around the X-axis by angle theta
    """
    return np.c_[
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ]


def plti(im, h=8, **kwargs):
    """
    Helper function to plot an image.
    """
    y = im.shape[0]
    x = im.shape[1]
    w = (y/x) * h
    plt.figure(figsize=(w,h))
    plt.imshow(im, interpolation="none", **kwargs)
    plt.axis('off')


def test1(path):
    im1 = np.asarray(Image.open(path))
    im_normed = do_normalise(im1)
    im_rotated = np.einsum("ijk,lk->ijl", im_normed, rotation_matrix(np.pi))
    im2 = undo_normalise(im_rotated)

    plti(im2)
    Image.fromarray(im2).show()


def test2(path):
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(5, 8))

    im = np.asarray(Image.open(path))

    for i in range(20):
        im_normed = do_normalise(im)
        im_rotated = np.einsum("ijk,lk->ijl", im_normed, rotation_matrix(i * np.pi / 10))
        im2 = undo_normalise(im_rotated)

        ax.imshow(im2)
        ax.set_title("Angle: {}*pi/10".format(i), fontsize=20)
        ax.set_axis_off()

        Image.fromarray(im2).save(f'rendered/{i}.jpg')

    '''anim = FuncAnimation(fig, update, frames=np.arange(0, 20), interval=50)
    anim.save('colour_rotation.gif', dpi=80, writer='imagemagick')
    plt.close()'''