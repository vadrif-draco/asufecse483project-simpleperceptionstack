"""
Lane Lines Detection pipeline

Usage:
    main.py [--verbose] INPUT_PATH OUTPUT_PATH 

Options:

-h --help                               show this screen
--verbose                               show perspective transform, binary image, additional debug info
"""

from LaneDetector import *

from moviepy.editor import VideoFileClip
from IPython.display import HTML

from docopt import docopt

def main():
    args = docopt(__doc__)
    input = args['INPUT_PATH']
    output = args['OUTPUT_PATH']


    lanedet = LaneDetector()
    Verbose = False

    if args['--verbose']:
        Verbose = True

    process_frame = lambda frame: lanedet.pipeline(frame,Verbose = Verbose, diagnostics=False)

    video_input = VideoFileClip(input)                          
    processed_video = video_input.fl_image(process_frame)
    processed_video.write_videofile(output, audio=False)


if __name__ == "__main__":
    main()