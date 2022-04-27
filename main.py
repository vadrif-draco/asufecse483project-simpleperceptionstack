"""
Usage:
    main.py [--verbose] [--debug] INPUT_PATH OUTPUT_PATH 

Options:

-h --help                               show this screen
--verbose                            show perspective transform, binary image
--debug                            Enable debugging mode
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
    debugging = False

    if args['--verbose']:
        Verbose = True

    if args['--debug']:
        debugging = True

    process_frame = lambda frame: lanedet.pipeline(frame,Verbose = Verbose, debugging=debugging)

    video_input = VideoFileClip(input)                          
    processed_video = video_input.fl_image(process_frame)
    if debugging:
        processed_video.write_videofile(output, audio=False, logger=None)
    else:
        processed_video.write_videofile(output, audio=False)


if __name__ == "__main__":
    main()