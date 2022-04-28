from typing import Tuple
import cv2
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


def grid_plot(images: Tuple[Tuple[cv2.Mat, str, str]], rows: int = 1, vertical: bool = False):

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


# def load_stream(video_path, frame_size=None):
    
#     stream = cv2.VideoCapture(video_path)

#     frame = stream.read()
#     frames = []

#     while(frame[0]):
#         try: frames.append(cv2.resize(frame[1], frame_size) if frame_size else frame[1])
#         except Exception as e: print(e)
#         frame = stream.read()
        
#     stream.release()

#     return frames


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)

    # Print New Line on Complete
    if iteration == total: print()
