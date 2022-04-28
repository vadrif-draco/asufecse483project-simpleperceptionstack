import os
import cv2
import numpy as np
import datetime as dt

import macros
import salama_preprocessing_techniques as prep
import salama_lane_detection_algorithm as lane
import argparse


def __pipeline_preprocesses(frame_BGR):

    frame_GRAY = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2GRAY)
    frame_HLS_S = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HLS)[:, :, 2]
    frame_HLS_L = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HLS)[:, :, 1]
    frame_HSV_V = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HLS)[:, :, 1]
    frame_LS_hybrid = cv2.addWeighted(frame_HLS_L, 1.5, frame_HLS_S, 2.5, -255)
    frame_VS_hybrid = cv2.addWeighted(frame_HSV_V, 1.5, frame_HLS_S, 2.5, -255)

    HLS_color_thresholded_edges, _, _ = prep.color_thresholded_edges_pre(frame_BGR, prep.HLS_COLORSPACE_THRESH)
    LAB_color_thresholded_edges, _, _ = prep.color_thresholded_edges_pre(frame_BGR, prep.LAB_COLORSPACE_THRESH)
    # HLS_S_thresholded_adaptive = prep.adaptive_thresholding_pre(frame_HLS_S)
    LS_hybrid_thresholded_adaptive = prep.adaptive_thresholding_pre(frame_LS_hybrid)
    VS_hybrid_thresholded_adaptive = prep.adaptive_thresholding_pre(frame_VS_hybrid)
    HLS_L_sobel_thresh, _ = prep.sobel_pre(frame_HLS_S, (1, 3), (50, 150))
    LS_hybrid_sobel_thresh, _ = prep.sobel_pre(frame_LS_hybrid, (1, 5), (50, 150))
    VS_hybrid_sobel_thresh, _ = prep.sobel_pre(frame_VS_hybrid, (1, 5), (50, 150))
    TOPHAT_thresholded, _ = prep.morph_top_hat_pre(frame_BGR)
    image_inv_BLACKHAT_thresholded, _, _ = prep.morph_inv_black_hat_pre(frame_BGR)
    raw_canny = prep.canny_raw_pre(frame_GRAY)

    # HLS_S_thresholded_adaptive,\
    return\
        HLS_color_thresholded_edges,\
        LAB_color_thresholded_edges,\
        LS_hybrid_thresholded_adaptive,\
        VS_hybrid_thresholded_adaptive,\
        HLS_L_sobel_thresh,\
        LS_hybrid_sobel_thresh,\
        VS_hybrid_sobel_thresh,\
        TOPHAT_thresholded,\
        image_inv_BLACKHAT_thresholded,\
        raw_canny,


# def __pipeline_score(previous_lane_data: lane.lane_data, next_lane_data: lane.lane_data):

#     union = 0
#     intersection = 0
#     for y in range(0, next_lane_data.lane_height):
#         for x in range(0, next_lane_data.lane_width):
#             match_next = (

#                 x > next_lane_data.left_lane_polyfit_pts[y]
#                 and
#                 x < next_lane_data.right_lane_polyfit_pts[y]

#             )
#             match_prev = (

#                 x > previous_lane_data.left_lane_polyfit_pts[y]
#                 and
#                 x < previous_lane_data.right_lane_polyfit_pts[y]

#             )
#             union += (match_next or match_prev)
#             intersection += (match_next and match_prev)

#     return ((intersection / union) if union != 0 and intersection != 0 else inf, next_lane_data)


##################################################################################################


# def __pipeline_score(previous_lane_data: lane.lane_data, next_lane_data: lane.lane_data):


#     score = [0, 0]
#     for y in range(0, next_lane_data.lane_height):
#         score[0] += abs(previous_lane_data.left_lane_polyfit_pts[y] - next_lane_data.left_lane_polyfit_pts[y])
#         score[1] += abs(previous_lane_data.right_lane_polyfit_pts[y] - next_lane_data.right_lane_polyfit_pts[y])

#     return score


##################################################################################################


def __pipeline_lanes_score(previous_lane_data: lane.lane_data, next_lane_candidates: list[cv2.Mat],
                           left_lane_mask, right_lane_mask):

    candidates_scored = []
    for candidate in next_lane_candidates:

        # FIXME: Hough is too harsh, but I need its guidance nevertheless
        _, candidate_edges_canvas = prep.hough_transform_raw_pre(candidate)
        # candidate_edges_canvas = candidate

        candidate_edges_canvas_left_lane = cv2.bitwise_and(candidate_edges_canvas, left_lane_mask)
        candidate_edges_canvas_left_lane_pts = np.argwhere(candidate_edges_canvas_left_lane != 0)
        if candidate_edges_canvas_left_lane_pts.size == 0: continue
            # placeholder_canvas = np.zeros_like(candidate_edges_canvas)
            # placeholder_canvas[:] = 127
            # return [99999, None, placeholder_canvas]

        candidate_edges_canvas_right_lane = cv2.bitwise_and(candidate_edges_canvas, right_lane_mask)
        candidate_edges_canvas_right_lane_pts = np.argwhere(candidate_edges_canvas_right_lane != 0)
        if candidate_edges_canvas_right_lane_pts.size == 0: continue
            # placeholder_canvas = np.zeros_like(candidate_edges_canvas)
            # placeholder_canvas[:] = 127
            # return [99999, None, placeholder_canvas]

        polyfit_range = np.linspace(0, previous_lane_data.lane_height - 1, previous_lane_data.lane_height)

        candidate_data = lane.lane_data(

            lane_height=previous_lane_data.lane_height,
            lane_width=previous_lane_data.lane_width,
            l_pts=candidate_edges_canvas_left_lane_pts,
            r_pts=candidate_edges_canvas_right_lane_pts,
            range_poly=polyfit_range,
            l_poly=lane.polyfit_lane_pts(

                lane_pts=candidate_edges_canvas_left_lane_pts,
                polyfit_range=polyfit_range,
                polyfit_rank=1

            ),
            r_poly=lane.polyfit_lane_pts(

                lane_pts=candidate_edges_canvas_right_lane_pts,
                polyfit_range=polyfit_range,
                polyfit_rank=1

            ),
        )

        score = [0, 0]
        for y in range(0, previous_lane_data.lane_height):
            score[0] += abs(previous_lane_data.left_lane_polyfit_pts[y] - candidate_data.left_lane_polyfit_pts[y])
            score[1] += abs(previous_lane_data.right_lane_polyfit_pts[y] - candidate_data.right_lane_polyfit_pts[y])

        candidates_scored.append([score, candidate_data, candidate_edges_canvas])

    return candidates_scored


def get_next_lane(previous_lane_data: lane.lane_data, frame_BGR: cv2.Mat, skycutoff, hoodcutoff, debugging):

    left_ROI, right_ROI = previous_lane_data.extract_ROIs()
    left_lane_mask = previous_lane_data.extract_mask(left_ROI)
    right_lane_mask = previous_lane_data.extract_mask(right_ROI)
    lane_masks_intersection = cv2.bitwise_and(left_lane_mask, right_lane_mask)
    left_lane_mask = cv2.bitwise_xor(left_lane_mask, lane_masks_intersection)
    right_lane_mask = cv2.bitwise_xor(right_lane_mask, lane_masks_intersection)

    next_lane_candidates = __pipeline_preprocesses(frame_BGR[skycutoff:hoodcutoff])

    candidates_scored = __pipeline_lanes_score(previous_lane_data, next_lane_candidates,
                                               left_lane_mask, right_lane_mask)

    best_candidate_score_left = min(candidates_scored, key=lambda c: c[0][0])
    best_candidate_score_right = min(candidates_scored, key=lambda c: c[0][1])

    try:

        get_next_lane.best_candidate_score_left_memory = get_next_lane.best_candidate_score_left_memory[1:]
        get_next_lane.best_candidate_score_left_memory.append(best_candidate_score_left[0][0])

        get_next_lane.best_candidate_score_right_memory = get_next_lane.best_candidate_score_right_memory[1:]
        get_next_lane.best_candidate_score_right_memory.append(best_candidate_score_right[0][1])

    except AttributeError:

        get_next_lane.best_candidate_score_left_memory = [1000] * 5
        get_next_lane.best_candidate_score_right_memory = [1000] * 5

    # TODO: If best score is below some threshold (1.25x w.r.t memory average sounds good)...
    # ...go ahead with ensemble, in increasing voting threshold order
    # If something gets a better score (that is below threshold), take it obviously lol
    # However if not, just re-use previous lane data
    
    # use_previous = False

    # if best_candidate_score_left[0][0] > 1.25 * np.mean(get_next_lane.best_candidate_score_left_memory) or \
    #         best_candidate_score_right[0][1] > 1.25 * np.mean(get_next_lane.best_candidate_score_right_memory):

    #     voting_ensemble_candidates = []
    #     for i in range(len(next_lane_candidates)):
    #         voting_ensemble_candidates.append(prep.edge_voting_ensemble(next_lane_candidates, i))

    #     voting_ensemble_candidates_scored = __pipeline_lanes_score(previous_lane_data, next_lane_candidates,
    #                                                                left_lane_mask, right_lane_mask)

    #     best_voting_ensemble_candidate_score_left = min(voting_ensemble_candidates_scored, key=lambda c: c[0][0])
    #     best_voting_ensemble_candidate_score_right = min(voting_ensemble_candidates_scored, key=lambda c: c[0][1])

    #     best_candidate_score_left = min(best_candidate_score_left, best_voting_ensemble_candidate_score_left, key=lambda c: c[0][0])
    #     best_candidate_score_right = min(best_candidate_score_right, best_voting_ensemble_candidate_score_right, key=lambda c: c[0][0])

    #     if best_candidate_score_left[0][0] > 1.25 * np.mean(get_next_lane.best_candidate_score_left_memory) or \
    #             best_candidate_score_right[0][1] > 1.25 * np.mean(get_next_lane.best_candidate_score_right_memory):

    #         next_lane_data = previous_lane_data
    #         use_previous = True
            
    # if not use_previous:

    next_lane_data = lane.lane_data(

        lane_height=best_candidate_score_left[1].lane_height,  # arbitrary choice, either of them work
        lane_width=best_candidate_score_left[1].lane_width,  # same ^
        range_poly=best_candidate_score_left[1].polyfit_range,  # same ^
        l_poly=best_candidate_score_left[1].left_lane_polyfit_pts,
        r_poly=best_candidate_score_right[1].right_lane_polyfit_pts,
        l_pts=best_candidate_score_left[1].left_lane_pts,
        r_pts=best_candidate_score_right[1].right_lane_pts,

    )

    next_lane_canvas = next_lane_data.draw(frame_BGR, skycutoff, hoodcutoff, debugging)

    if debugging:

        candidates_marked_RGB = []
        for candidate in candidates_scored:

            best_left = (candidate[1] == best_candidate_score_left[1])
            best_right = (candidate[1] == best_candidate_score_right[1])

            candidate_left = cv2.bitwise_and(candidate[2], left_lane_mask)
            candidate_right = cv2.bitwise_and(candidate[2], right_lane_mask)

            candidate_left_RGB = cv2.cvtColor(candidate_left, cv2.COLOR_GRAY2RGB)
            candidate_right_RGB = cv2.cvtColor(candidate_right, cv2.COLOR_GRAY2RGB)

            left_lane_mask_RGB = cv2.cvtColor(left_lane_mask, cv2.COLOR_GRAY2RGB)
            right_lane_mask_RGB = cv2.cvtColor(right_lane_mask, cv2.COLOR_GRAY2RGB)

            if best_left:
                candidate_left_RGB[:, :, 0] = 0
                candidate_left_RGB[:, :, 1] = 0
                left_lane_mask_RGB[:, :, 0] = 0
                left_lane_mask_RGB[:, :, 1] = 0

            if best_right:
                candidate_right_RGB[:, :, 1] = 0
                candidate_right_RGB[:, :, 2] = 0
                right_lane_mask_RGB[:, :, 1] = 0
                right_lane_mask_RGB[:, :, 2] = 0

            candidate_RGB = cv2.bitwise_or(candidate_left_RGB, candidate_right_RGB)
            lane_masks_RGB = cv2.bitwise_or(left_lane_mask_RGB, right_lane_mask_RGB)
            candidate_RGB = cv2.addWeighted(candidate_RGB, 1, lane_masks_RGB, 0.2, 0)

            candidates_marked_RGB.append(candidate_RGB)

        # Left score
        frame_BGR = cv2.putText(

            img=frame_BGR,
            text=f'{best_candidate_score_left[0][0]} ({np.mean(get_next_lane.best_candidate_score_left_memory)})',
            bottomLeftOrigin=False,  # when false, it is at the top-left corner.
            org=(4, 16 * (np.shape(frame_BGR)[1] // 640)),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=np.shape(frame_BGR)[1] // 640,
            color=(0, 0, 255),  # BGR
            lineType=cv2.LINE_AA

        )

        # Right score
        frame_BGR = cv2.putText(

            img=frame_BGR,
            text=f'{best_candidate_score_right[0][1]} ({np.mean(get_next_lane.best_candidate_score_right_memory)})',
            bottomLeftOrigin=False,  # when false, it is at the top-left corner.
            org=(4, 32 * (np.shape(frame_BGR)[1] // 640)),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=np.shape(frame_BGR)[1] // 640,
            color=(255, 0, 0),  # BGR
            lineType=cv2.LINE_AA

        )

        candidates_preview = cv2.vconcat(candidates_marked_RGB)
        # cv2.imshow(

        #     "Debug Mode Preview",
        #     cv2.hconcat(

        #         [
        #             frame_BGR,
        #             cv2.resize(

        #                 candidates_preview,
        #                 (int(np.shape(candidates_preview)[1] * (np.shape(frame_BGR)[0] / np.shape(candidates_preview)[0])),
        #                  np.shape(frame_BGR)[0])

        #             )
        #         ]

        #     )

        # )
        # cv2.waitKey(1000)

    return next_lane_data, next_lane_canvas, candidates_preview if debugging else None


def __pipeline_initial_detection(first_frame_BGR):

    HOODCUTOFF, _ = prep.get_hood_cutoff(first_frame_BGR)
    SKYCUTOFF, _, _ = prep.get_sky_cutoff(first_frame_BGR[:HOODCUTOFF], (1, 3))

    _, detection_ready_frame =\
        prep.hough_transform_raw_pre(
            prep.edge_voting_ensemble(
                __pipeline_preprocesses(
                    first_frame_BGR[SKYCUTOFF:HOODCUTOFF]
                ),
                voting_threshold=4
            )
        )

    detected_lane_data = lane.peeking_center_detect(detection_ready_frame, 1)

    return detected_lane_data, HOODCUTOFF, SKYCUTOFF


def run_pipeline(width, height, preview_factor, input_file_path, output_folder_path, debugging):

    if not os.path.exists(input_file_path): print(f"Sorry, the provided input path ({input_file_path}) does not exist."); return

    video_input = cv2.VideoCapture(input_file_path)
    FRAME_COUNT = int(video_input.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = int(video_input.get(cv2.CAP_PROP_FPS))
    ret, first_frame_BGR = video_input.read()
    first_frame_BGR = cv2.resize(first_frame_BGR, (width, height))

    if not ret: print("Something went wrong reading the stream... Exiting..."); return

    print(f"Reading from ({os.path.abspath(input_file_path)})...")
    if not os.path.exists(output_folder_path): os.makedirs(output_folder_path)
    if not os.path.exists(f"{output_folder_path}/frames"): os.makedirs(f"{output_folder_path}/frames")
    print(f"Saving into ({os.path.abspath(output_folder_path)})...")

    video_output = cv2.VideoWriter(f'{output_folder_path}/video.avi',
                                   cv2.VideoWriter_fourcc(*"XVID"), FPS, (width, height))

    initial_lane_data, HOODCUTOFF, SKYCUTOFF = __pipeline_initial_detection(first_frame_BGR)
    initial_lane_canvas = initial_lane_data.draw(first_frame_BGR, SKYCUTOFF, HOODCUTOFF, debugging)
    initial_lane_detection_frame = cv2.vconcat(

        [
            first_frame_BGR[:SKYCUTOFF],
            cv2.addWeighted(initial_lane_canvas, 1, first_frame_BGR[SKYCUTOFF:HOODCUTOFF], 1, 0),
            first_frame_BGR[HOODCUTOFF:]
        ]

    )

    cv2.imwrite(f'{output_folder_path}/frames/0001.jpg', initial_lane_detection_frame)
    video_output.write(initial_lane_detection_frame)
    macros.printProgressBar(1, FRAME_COUNT, suffix=f' (0001 / {FRAME_COUNT})')

    index = 1
    ret, frame_BGR = video_input.read()
    previous_lane_data = initial_lane_data
    try:
        while(ret):

            index += 1
            frame_BGR = cv2.resize(frame_BGR, (width, height))

            current_lane_data, current_lane_canvas, candidates_preview =\
                get_next_lane(previous_lane_data, frame_BGR, SKYCUTOFF, HOODCUTOFF, debugging)

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
                org=(4, height - 8 * (height // 360)),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=height // 360,
                color=(255, 255, 255),  # BGR
                lineType=cv2.LINE_AA

            )
            if debugging:

                debug_view = cv2.hconcat(

                    [
                        current_lane_detection_frame,
                        cv2.resize(

                            candidates_preview,
                            (int(np.shape(candidates_preview)[1] * (np.shape(frame_BGR)[0] / np.shape(candidates_preview)[0])),
                             np.shape(frame_BGR)[0])

                        )
                    ]

                )

                cv2.imshow(

                    "Debug Mode Preview",
                    cv2.resize(debug_view,
                               [
                                   np.shape(debug_view)[1] * preview_factor,
                                   np.shape(debug_view)[0] * preview_factor
                               ])

                )
                cv2.waitKey(1000)

            cv2.imwrite(f'{output_folder_path}/frames/{str(index).zfill(4)}.jpg', current_lane_detection_frame)
            video_output.write(current_lane_detection_frame)

            macros.printProgressBar(index, FRAME_COUNT, suffix=f' ({str(index).zfill(4)} / {FRAME_COUNT})')

            previous_lane_data = current_lane_data
            ret, frame_BGR = video_input.read()

    except KeyboardInterrupt: print("\n\nAbrupt termination requested; saving progress so far...")
    # except Exception as e: print(f"\n\nAn error has occured! Saving what can be saved...\n\n{e.__traceback__}")
    finally: print("\n\nExiting...\n"); video_input.release(); video_output.release()


if __name__ == "__main__":

    run_pipeline(

        width=1280 // 2,
        height=720 // 2,
        preview_factor=2,
        input_file_path="./assets/harder_challenge_video.mp4",
        output_folder_path=f"./out/{dt.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}",
        debugging=True

    )

    # alt_pipeline(INPUT_PATH, OUTPUT_PATH)


# def alt_pipeline(INPUT_PATH, OUTPUT_PATH):

#     from moviepy.editor import VideoFileClip

#     video_input = VideoFileClip(INPUT_PATH)

#     HOODCUTOFF, _ = pre.get_hood_cutoff(video_input.get_frame(0))
#     SKYCUTOFF, _, _ = pre.get_sky_cutoff(video_input.get_frame(0)[:HOODCUTOFF], (1, 3))

#     def process_frame(frame_BGR):

#         preprocessed_frame, _, _ = pre.color_thresholding_preprocess(frame_BGR[SKYCUTOFF:HOODCUTOFF], [
#             # H, L, S
#             [(18, 22), (0, 255), (15, 255)],  # Yellows
#             [(0, 255), (191, 255), (127, 255)],  # Whites
#         ])

#         left_lane_polyfit_pts, right_lane_polyfit_pts, polyfit_range, \
#             left_lane_pts_x, left_lane_pts_y, left_peeking_center_trace_x, left_peeking_center_trace_y, \
#             right_lane_pts_x, right_lane_pts_y, right_peeking_center_trace_x, right_peeking_center_trace_y = \
#             salama_lane_detection_algorithm.peeking_center_detect(preprocessed_frame)

#         lane = np.zeros_like(frame_BGR[:HOODCUTOFF])
#         for y in range(0, np.shape(preprocessed_frame)[0], 1):
#             for x in range(int(left_lane_polyfit_pts[y]) + 1, int(right_lane_polyfit_pts[y])):
#                 if x > 0 and x < np.shape(frame_BGR)[1]:
#                     lane[SKYCUTOFF + y][x] = [0, 31 + 224 / (np.shape(preprocessed_frame)[0] - y), 0]

#         return cv2.addWeighted(lane, 1, frame_BGR[:HOODCUTOFF], 1, 0)

#     processed_video = video_input.fl_image(process_frame)
#     processed_video.write_videofile(f"project_video_output.mp4", audio=False)
