from typing import Tuple
from xmlrpc.client import boolean
import cv2
import numpy as np
from matplotlib import pyplot as plt


def initialize_preview(width, height, preview_size_factor):
    WIDTH = width
    HEIGHT = height
    PREVIEW_SIZE_FACTOR = preview_size_factor

    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    plt.rcParams["figure.figsize"] = (WIDTH * PREVIEW_SIZE_FACTOR * px, HEIGHT * PREVIEW_SIZE_FACTOR * px)


def load_image(image_path, size=None):
    image = cv2.imread(image_path)
    if size: image = cv2.resize(image, size)
    return image


def show(image, title='Figure', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)


def grid_plot(images: Tuple[Tuple[cv2.Mat, str, str]], rows: int = 1, vertical: boolean = False):

    def _grid_plot(image_tuple, slot, title="Figure", cmap="gray"):
        if len(image_tuple) >= 2: title = image_tuple[1]
        if len(image_tuple) == 3: cmap = image_tuple[2]
        slot.imshow(image_tuple[0], cmap)
        slot.title.set_text(title)

    i = 0
    cols = int(len(images) / rows)
    _, image_axes = plt.subplots(rows if not vertical else cols, cols if not vertical else rows)

    for axis in image_axes:

        if not (rows == 1 or cols == 1):
            for cell in axis:
                _grid_plot(images[i], cell)
                i += 1

        else:
            _grid_plot(images[i], axis)
            i += 1


def load_stream(video_path, size=None):
    stream = cv2.VideoCapture(video_path)

    frame = stream.read()
    frames = []

    while(frame[0]):
        try: frames.append(cv2.resize(frame[1], size) if size else frame[1])
        except Exception as e: print(e)
        frame = stream.read()

    return frames
