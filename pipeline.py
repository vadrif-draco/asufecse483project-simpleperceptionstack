import os
import cv2
import numpy as np
import datetime as dt

import macros
import pipeline_phase1
import pipeline_phase2


def run_pipeline(downsize_ratio, input_file_path, output_folder_path, frames_only, video_only, debugging, verbose):

    if not os.path.exists(input_file_path):
        print(f"Sorry, the provided input path ({input_file_path}) does not exist.")
        return

    video_input = cv2.VideoCapture(input_file_path)
    ret, first_frame_BGR = video_input.read()
    if not ret:
        print("Something went wrong reading the stream... Exiting...")
        return

    print(f"Reading from ({os.path.abspath(input_file_path)})...")

    FRAME_HEIGHT = np.shape(first_frame_BGR)[0] // downsize_ratio
    FRAME_WIDTH = np.shape(first_frame_BGR)[1] // downsize_ratio
    FRAME_COUNT = int(video_input.get(cv2.CAP_PROP_FRAME_COUNT))
    FRAMES_PER_SECOND = int(video_input.get(cv2.CAP_PROP_FPS))

    first_frame_BGR = cv2.resize(first_frame_BGR, (FRAME_WIDTH, FRAME_HEIGHT))

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    if not video_only:
        if not os.path.exists(f"{output_folder_path}/frames"):
            os.makedirs(f"{output_folder_path}/frames")

    print(f"Saving into ({os.path.abspath(output_folder_path)})...")

    initial_lane_data, HOODCUTOFF, SKYCUTOFF = pipeline_phase1.get_initial_lane(first_frame_BGR)
    initial_lane_canvas = initial_lane_data.draw(first_frame_BGR, SKYCUTOFF, HOODCUTOFF, debugging)
    initial_lane_detection_frame = cv2.vconcat(

        [
            first_frame_BGR[:SKYCUTOFF],
            cv2.addWeighted(initial_lane_canvas, 1, first_frame_BGR[SKYCUTOFF:HOODCUTOFF], 1, 0),
            first_frame_BGR[HOODCUTOFF:]
        ]

    )
    
    bounding_boxes_to_draw, bounding_boxes_confidences, bounding_boxes_nms_indices =\
        pipeline_phase2.detect_car(first_frame_BGR)
        
    for i in bounding_boxes_nms_indices:
        box = bounding_boxes_to_draw[i]
        cv2.rectangle(initial_lane_detection_frame, box[:2], box[2:4], (0, 0, 255), 4)
        cv2.putText(initial_lane_detection_frame, f'Car @{int(10000*bounding_boxes_confidences[i])/100}%', (box[0], box[1] - 10),\
            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    if debugging:

        # want to draw rectangle on both sides of height same as image
        # and width as width * height / ((hoodcutoff-skycutoff) * num)
        # of color 63 gray, to be rgb
        debug_mode_placeholder = cv2.cvtColor(np.full(

            shape=(FRAME_HEIGHT,
                   int(FRAME_WIDTH * FRAME_HEIGHT / ((HOODCUTOFF - SKYCUTOFF) * pipeline_phase1.NUM_OF_PREPROCESSES_USED))),
            fill_value=63,
            dtype=np.uint8

        ), cv2.COLOR_GRAY2BGR)

        initial_lane_detection_frame = cv2.hconcat([debug_mode_placeholder,
                                                    initial_lane_detection_frame,
                                                    debug_mode_placeholder])

    if not frames_only:
        video_output = cv2.VideoWriter(

            f'{output_folder_path}/video.avi',
            cv2.VideoWriter_fourcc(*"XVID"),
            FRAMES_PER_SECOND,
            (np.shape(initial_lane_detection_frame)[1], np.shape(initial_lane_detection_frame)[0])

        )
        video_output.write(initial_lane_detection_frame)

    if not video_only:
        cv2.imwrite(

            f'{output_folder_path}/frames/0001.jpg',
            initial_lane_detection_frame

        )

    print('\n You may press CTRL+C at any time to save and terminate the pipeline.\n')
    macros.printProgressBar(1, FRAME_COUNT, suffix=f' (0001 / {FRAME_COUNT})')

    index = 1
    ret, frame_BGR = video_input.read()
    previous_lane_data = initial_lane_data
    try:
        while(ret):

            index += 1
            frame_BGR = cv2.resize(frame_BGR, (FRAME_WIDTH, FRAME_HEIGHT))

            current_lane_data, current_lane_canvas, candidates_preview =\
                pipeline_phase1.get_next_lane(previous_lane_data, frame_BGR, SKYCUTOFF, HOODCUTOFF, debugging)

            current_lane_detection_frame = cv2.vconcat(

                [
                    frame_BGR[:SKYCUTOFF],
                    cv2.addWeighted(current_lane_canvas, 1, frame_BGR[SKYCUTOFF:HOODCUTOFF], 1, 0),
                    frame_BGR[HOODCUTOFF:]
                ]

            )

            current_lane_detection_frame = cv2.putText(

                img=current_lane_detection_frame,
                text=f'{str(index).zfill(4)} / {FRAME_COUNT}',
                bottomLeftOrigin=False,
                org=(4, FRAME_HEIGHT - 8),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(255, 255, 255),  # BGR
                lineType=cv2.LINE_AA

            )
    
            bounding_boxes_to_draw, bounding_boxes_confidences, bounding_boxes_nms_indices =\
                pipeline_phase2.detect_car(frame_BGR)
                
            for i in bounding_boxes_nms_indices:
                box = bounding_boxes_to_draw[i]
                cv2.rectangle(current_lane_detection_frame, box[:2], box[2:4], (0, 0, 255), 4)
                cv2.putText(current_lane_detection_frame, f'Car @{int(10000*bounding_boxes_confidences[i])/100}%', (box[0], box[1] - 10),\
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

            if debugging:

                current_lane_detection_frame = cv2.hconcat(

                    [np.full_like(candidates_preview[0], 63), current_lane_detection_frame, candidates_preview[0]]
                    if len(candidates_preview) == 1 else
                    [candidates_preview[1], current_lane_detection_frame, candidates_preview[0]]

                )

                cv2.imshow("Debug Mode Preview",
                           cv2.resize(current_lane_detection_frame,
                                      [int(np.shape(current_lane_detection_frame)[1] * downsize_ratio * 0.8),
                                       int(np.shape(current_lane_detection_frame)[0] * downsize_ratio * 0.8)]))

                cv2.waitKey(1)

            if not video_only: cv2.imwrite(f'{output_folder_path}/frames/{str(index).zfill(4)}.jpg',
                                           current_lane_detection_frame)

            if not frames_only: video_output.write(current_lane_detection_frame)

            macros.printProgressBar(index, FRAME_COUNT, suffix=f' ({str(index).zfill(4)} / {FRAME_COUNT})')

            previous_lane_data = current_lane_data
            ret, frame_BGR = video_input.read()

        print("\n\n Pipeline finished processing the video successfully; saving...\n")

    except KeyboardInterrupt: print("\n\n Abrupt termination requested; saving progress so far...\n")
    # except Exception as e: print(e if verbose else "\n\n An error has occured! Saving what can be saved...\n")
    finally:
        video_input.release()
        if not frames_only: video_output.release()


if __name__ == "__main__":

    print() # Just a new line for readability's sake
    import argparse
    argparser = argparse.ArgumentParser(description='configure/run the pipeline for this lane perception stack.')

    argparser.add_argument("src_file",
                           help="path to the video FILE that you want to run the pipeline on.")

    argparser.add_argument("dest_folder",
                           nargs="?",
                           default=f"./out/{dt.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}",
                           help="(optional) path to the output FOLDER to save pipeline output in; if left empty defaults\
                           to a subfolder named with the run date/time in a folder 'out' within this file's directory")

    argparser.add_argument("-r", metavar='downsize_ratio',
                           choices=range(1, 5),
                           default=1,
                           type=int,
                           help="ratio by which input is to be downsized for the sake of output rendering performance")

    argparser.add_argument("-d", "--debug",
                           action="store_true",
                           help="show debugging window and add some debugging info. (marked lanes) to the output frames/video")

    argparser.add_argument("-v", "--verbose",
                           action="store_true",
                           help="print diagnostics and other useful information in a log file (will be put in output folder) [to be implemented]")

    outputstyle = argparser.add_mutually_exclusive_group()

    outputstyle.add_argument("-F", "--frames_only",
                             action="store_true",
                             help="save output frames only (i.e., do not save video)")

    outputstyle.add_argument("-V", "--video_only",
                             action="store_true",
                             help="save output video only (i.e., no frame-by-frame output)")

    args = argparser.parse_args()

    run_pipeline(

        downsize_ratio=args.r,
        input_file_path=args.src_file,
        output_folder_path=args.dest_folder,
        frames_only=args.frames_only,
        video_only=args.video_only,
        debugging=args.debug,
        verbose=args.verbose
        
    )
