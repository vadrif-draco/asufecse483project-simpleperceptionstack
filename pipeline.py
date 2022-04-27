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
    HLS_S_thresholded_adaptive = prep.adaptive_thresholding_pre(frame_HLS_S)
    LS_hybrid_thresholded_adaptive = prep.adaptive_thresholding_pre(frame_LS_hybrid)
    VS_hybrid_thresholded_adaptive = prep.adaptive_thresholding_pre(frame_VS_hybrid)
    HLS_L_sobel_thresh, _ = prep.sobel_pre(frame_HLS_S, (1, 3), (50, 150))
    LS_hybrid_sobel_thresh, _ = prep.sobel_pre(frame_LS_hybrid, (1, 5), (50, 150))
    VS_hybrid_sobel_thresh, _ = prep.sobel_pre(frame_VS_hybrid, (1, 5), (50, 150))
    TOPHAT_thresholded, _ = prep.morph_top_hat_pre(frame_BGR)
    image_inv_BLACKHAT_thresholded, _, _ = prep.morph_inv_black_hat_pre(frame_BGR)
    raw_canny = prep.canny_raw_pre(frame_GRAY)

    return\
        HLS_color_thresholded_edges,\
        LAB_color_thresholded_edges,\
        HLS_S_thresholded_adaptive,\
        LS_hybrid_thresholded_adaptive,\
        VS_hybrid_thresholded_adaptive,\
        HLS_L_sobel_thresh,\
        LS_hybrid_sobel_thresh,\
        VS_hybrid_sobel_thresh,\
        TOPHAT_thresholded,\
        image_inv_BLACKHAT_thresholded,\
        raw_canny,


def __pipeline_voting_ensemble(preprocesses, voting_threshold: int = 4):

    voting_ensemble_result = prep.edge_voting_ensemble(preprocesses, voting_threshold)
    voting_ensemble_edges, voting_ensemble_edges_canvas = prep.hough_transform_raw_pre(voting_ensemble_result)
    try:
        for edge in voting_ensemble_edges:
            cv2.line(voting_ensemble_edges_canvas,
                     (edge[0][0], edge[0][1]),
                     (edge[0][2], edge[0][3]),
                     255,
                     2)
        return voting_ensemble_edges_canvas
    except: return None


def __pipeline_score(previous_lane_data: lane.lane_data, next_lane_data: lane.lane_data):

    union = 0
    intersection = 0
    for y in range(0, next_lane_data.lane_height):
        for x in range(0, next_lane_data.lane_width):
            match_next = (

                x > next_lane_data.left_lane_polyfit_pts[y]
                and
                x < next_lane_data.right_lane_polyfit_pts[y]

            )
            match_prev = (

                x > previous_lane_data.left_lane_polyfit_pts[y]
                and
                x < previous_lane_data.right_lane_polyfit_pts[y]

            )
            union += (match_next or match_prev)
            intersection += (match_next and match_prev)

    return ((intersection / union), next_lane_data)


def get_next_lane(previous_lane_data: lane.lane_data, frame_BGR: cv2.Mat, skycutoff, hoodcutoff, debugging=False):

    edges_canvas_width = np.shape(frame_BGR[skycutoff:hoodcutoff])[1]
    edges_canvas_height = np.shape(frame_BGR[skycutoff:hoodcutoff])[0]

    left_ROI, right_ROI = lane.extract_separate_ROIs(previous_lane_data)
    next_left_lane_mask = np.zeros((previous_lane_data.lane_height, previous_lane_data.lane_width), dtype=np.uint8)
    next_right_lane_mask = np.zeros((previous_lane_data.lane_height, previous_lane_data.lane_width), dtype=np.uint8)

    for y, pair in enumerate(left_ROI):
        for x in range(pair[0], pair[1]):
            next_left_lane_mask[y][x] = 255

    for y, pair in enumerate(right_ROI):
        for x in range(pair[0], pair[1]):
            next_right_lane_mask[y][x] = 255

    next_lane_candidates = __pipeline_preprocesses(frame_BGR[skycutoff:hoodcutoff])

    candidate_lane_data_overlap_scores = []
    for candidate in next_lane_candidates:

        _, candidate_edges_canvas = prep.hough_transform_raw_pre(candidate)

        candidate_edges_canvas_left_lane = cv2.bitwise_and(candidate_edges_canvas, next_left_lane_mask)
        candidate_edges_canvas_left_lane_pts = np.argwhere(candidate_edges_canvas_left_lane == 255)

        candidate_edges_canvas_right_lane = cv2.bitwise_and(candidate_edges_canvas, next_right_lane_mask)
        candidate_edges_canvas_right_lane_pts = np.argwhere(candidate_edges_canvas_right_lane == 255)

        polyfit_range = np.linspace(0, edges_canvas_height - 1, edges_canvas_height)

        next_lane_data = lane.lane_data(

            lane_height=edges_canvas_height,
            lane_width=edges_canvas_width,
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

        candidate_lane_data_overlap_scores.append(__pipeline_score(previous_lane_data, next_lane_data))

    # TODO: Get max score, if higher than some arbitrary ratio, go on with it, if not use ensemble
    best_candidate = max(candidate_lane_data_overlap_scores, key=lambda c: c[0])[1]

    next_lane_canvas = best_candidate.draw(frame_BGR, skycutoff, hoodcutoff, debugging)

    return next_lane_data, next_lane_canvas


def run_pipeline(width, height, input_path, output_path, debugging=False):

    if not os.path.exists(input_path): print(f"Sorry, the provided input path ({input_path}) does not exist."); return
    print(f"Loading input stream frames from ({os.path.abspath(input_path)}), please wait...")

    frames_BGR = macros.load_stream(input_path, frame_size=(width, height))

    if not os.path.exists(output_path): os.makedirs(output_path)
    print(f"Saving output stream frames into ({os.path.abspath(output_path)})...")

    # No need to recalculate cutoffs every frame, just do it once for the first
    HOODCUTOFF, _ = prep.get_hood_cutoff(frames_BGR[0])
    SKYCUTOFF, _, _ = prep.get_sky_cutoff(frames_BGR[0][:HOODCUTOFF], (1, 3))

    detection_ready_frame = __pipeline_voting_ensemble(__pipeline_preprocesses(frames_BGR[0][SKYCUTOFF:HOODCUTOFF]))

    detected_lane_data = lane.peeking_center_detect(detection_ready_frame, 1)

    lane_canvas = detected_lane_data.draw(frames_BGR[0], SKYCUTOFF, HOODCUTOFF, debugging)

    initial_detected_lane_frame = cv2.addWeighted(lane_canvas, 1, frames_BGR[0][:HOODCUTOFF], 1, 0)

    OUTPUT_WIDTH = np.shape(initial_detected_lane_frame)[1]
    OUTPUT_HEIGHT = np.shape(initial_detected_lane_frame)[0]

    video_output = cv2.VideoWriter(f'{output_path}/video.avi',
                                   cv2.VideoWriter_fourcc(*"XVID"),
                                   30, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

    cv2.imwrite(f'{output_path}/1.jpg', cv2.vconcat([initial_detected_lane_frame, frames_BGR[0][HOODCUTOFF:]]))
    video_output.write(cv2.vconcat([initial_detected_lane_frame, frames_BGR[0][HOODCUTOFF:]]))

    previous_lane_data = detected_lane_data

    for index, frame_BGR in enumerate(frames_BGR[1:]):

        current_lane_data, current_lane_canvas =\
            get_next_lane(previous_lane_data, frame_BGR, SKYCUTOFF, HOODCUTOFF, debugging)

        detected_lane_frame = cv2.addWeighted(current_lane_canvas, 1, frame_BGR[:HOODCUTOFF], 1, 0)

        cv2.imwrite(f'{output_path}/{index + 2}.jpg', cv2.vconcat([detected_lane_frame, frame_BGR[HOODCUTOFF:]]))
        video_output.write(cv2.vconcat([detected_lane_frame, frame_BGR[HOODCUTOFF:]]))

        macros.printProgressBar(index + 2, len(frames_BGR), suffix=f' ({index + 2} / {len(frames_BGR)})')

        previous_lane_data = current_lane_data

    video_output.release()
    print("\n")


if __name__ == "__main__":

    WIDTH = 1280 // 2
    HEIGHT = 720 // 2
    INPUT_PATH = "./assets/project_video.mp4"
    OUTPUT_PATH = f"./out/{dt.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"
    DEBUGGING_MODE = False

    run_pipeline(WIDTH, HEIGHT, INPUT_PATH, OUTPUT_PATH, DEBUGGING_MODE)
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
