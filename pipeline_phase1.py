import cv2
import numpy as np
import salama_preprocessing_techniques as prep
import salama_lane_detection_algorithm as lane

NUM_OF_PREPROCESSES_USED = None


def __pipeline_preprocesses(frame_BGR):

    frame_GRAY = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2GRAY)
    frame_HLS_S = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HLS)[:, :, 2]
    frame_HLS_L = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HLS)[:, :, 1]
    frame_HSV_V = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HLS)[:, :, 1]
    frame_LS_hybrid = cv2.addWeighted(frame_HLS_L, 1.5, frame_HLS_S, 2.5, -255)
    frame_VS_hybrid = cv2.addWeighted(frame_HSV_V, 1.5, frame_HLS_S, 2.5, -255)

    preprocessed_images = []

    preprocessed_images.append(prep.color_thresholded_edges_pre(frame_BGR, prep.HLS_COLORSPACE_THRESH)[0])
    preprocessed_images.append(prep.color_thresholded_edges_pre(frame_BGR, prep.LAB_COLORSPACE_THRESH)[0])
    # preprocessed_images.append(prep.adaptive_thresholding_pre(frame_HLS_S))
    preprocessed_images.append(prep.adaptive_thresholding_pre(frame_LS_hybrid))
    preprocessed_images.append(prep.adaptive_thresholding_pre(frame_VS_hybrid))
    preprocessed_images.append(prep.sobel_pre(frame_HLS_S, (1, 3))[0])
    preprocessed_images.append(prep.sobel_pre(frame_LS_hybrid, (1, 5))[0])
    preprocessed_images.append(prep.sobel_pre(frame_VS_hybrid, (1, 5))[0])
    preprocessed_images.append(prep.morph_top_hat_pre(frame_BGR)[0])
    preprocessed_images.append(prep.morph_inv_black_hat_pre(frame_BGR)[0])
    preprocessed_images.append(prep.canny_raw_pre(frame_GRAY))

    global NUM_OF_PREPROCESSES_USED
    NUM_OF_PREPROCESSES_USED = len(preprocessed_images)

    return preprocessed_images


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


def __pipeline_get_lanes_scores(previous_lane_data: lane.lane_data,
                                next_lane_candidates: list[cv2.Mat],
                                left_lane_mask, right_lane_mask):

    candidates_scored = []
    for candidate in next_lane_candidates:

        # FIXME: Hough is too harsh, but I need its guidance nevertheless
        _, candidate_edges_canvas = prep.hough_transform_raw_pre(candidate)
        # candidate_edges_canvas = candidate

        candidate_edges_canvas_left_lane = cv2.bitwise_and(candidate_edges_canvas, left_lane_mask)
        candidate_edges_canvas_left_lane_pts = np.argwhere(candidate_edges_canvas_left_lane != 0)
        # if candidate_edges_canvas_left_lane_pts.size == 0: continue

        candidate_edges_canvas_right_lane = cv2.bitwise_and(candidate_edges_canvas, right_lane_mask)
        candidate_edges_canvas_right_lane_pts = np.argwhere(candidate_edges_canvas_right_lane != 0)
        # if candidate_edges_canvas_right_lane_pts.size == 0: continue

        if candidate_edges_canvas_left_lane_pts.size == 0 or candidate_edges_canvas_right_lane_pts.size == 0:
            candidates_scored.append([None, None, np.full_like(candidate_edges_canvas, 63)])
            continue

        candidate_lane_data = lane.lane_data(

            lane_height=previous_lane_data.lane_height,
            lane_width=previous_lane_data.lane_width,

            l_pts=candidate_edges_canvas_left_lane_pts,
            r_pts=candidate_edges_canvas_right_lane_pts,

            lane_polyfit_rank=1

        )

        # FIXME: This scoring method is too sensitive to the shaky camera
        # Since the algorithm depends on the difference in running scores
        # This sensitivity messes everything up
        # score = [0, 0]
        # for y in range(0, previous_lane_data.lane_height):
        #     score[0] += abs(previous_lane_data.left_lane_polyfit_pts[y] -
        #                     candidate_lane_data.left_lane_polyfit_pts[y])
        #     score[1] += abs(previous_lane_data.right_lane_polyfit_pts[y] -
        #                     candidate_lane_data.right_lane_polyfit_pts[y])

        ################################################################################

        approx_error_left = 0
        left_lane_coeffs = list(zip(previous_lane_data.left_lane_polyfit_coeffs,
                                    candidate_lane_data.left_lane_polyfit_coeffs))

        for previous_coeff, candidate_coeff in left_lane_coeffs:
            approx_error_left += 10000 * abs((previous_coeff - candidate_coeff) / previous_coeff)

        approx_error_right = 0
        right_lane_coeffs = list(zip(previous_lane_data.right_lane_polyfit_coeffs,
                                     candidate_lane_data.right_lane_polyfit_coeffs))

        for previous_coeff, candidate_coeff in right_lane_coeffs:
            approx_error_right += 10000 * abs((previous_coeff - candidate_coeff) / previous_coeff)

        score = [approx_error_left, approx_error_right]

        candidates_scored.append([score, candidate_lane_data, candidate_edges_canvas])

    return candidates_scored


def get_next_lane(previous_lane_data: lane.lane_data, frame_BGR: cv2.Mat, skycutoff, hoodcutoff, debugging):

    try:

        left_ROI = previous_lane_data.extract_left_ROI(ROI_width=min(
            np.shape(frame_BGR)[1] // 32 + get_next_lane.previous_lanes_average_score_left // 160,
            np.shape(frame_BGR)[1] // 16)
        )

        # left_ROI = previous_lane_data.extract_left_ROI(ROI_width=max(
        #     np.shape(frame_BGR)[1] // 32 - get_next_lane.previous_lanes_average_score_left // 160,
        #     np.shape(frame_BGR)[1] // 16)
        # )

        right_ROI = previous_lane_data.extract_right_ROI(ROI_width=max(
            np.shape(frame_BGR)[1] // 16 - get_next_lane.previous_lanes_average_score_right // 160,
            np.shape(frame_BGR)[1] // 32)
        )

    except AttributeError:

        left_ROI = previous_lane_data.extract_left_ROI(ROI_width=np.shape(frame_BGR)[1] // 16)
        right_ROI = previous_lane_data.extract_right_ROI(ROI_width=np.shape(frame_BGR)[1] // 16)

    left_lane_mask = previous_lane_data.extract_mask(left_ROI)
    right_lane_mask = previous_lane_data.extract_mask(right_ROI)

    # FIXME: The perpendiculars implementation messed this up
    # left_lane_mask_topleft_area = cv2.fillPoly(np.zeros_like(left_lane_mask),
    #                                            pts=[np.array([[0, 0],
    #                                                           left_ROI[0],
    #                                                           left_ROI[3]])],
    #                                            color=255)
    # right_lane_mask_topright_area = cv2.fillPoly(np.zeros_like(right_lane_mask),
    #                                              pts=[np.array([[np.shape(right_lane_mask)[1], 0],
    #                                                             right_ROI[1],
    #                                                             right_ROI[2]])],
    #                                              color=255)
    # lane_masks_intersection = cv2.bitwise_and(left_lane_mask, right_lane_mask)
    # left_lane_mask = cv2.subtract(left_lane_mask, right_lane_mask_topright_area)
    # left_lane_mask = cv2.subtract(left_lane_mask, lane_masks_intersection)
    # right_lane_mask = cv2.subtract(right_lane_mask, left_lane_mask_topleft_area)
    # right_lane_mask = cv2.subtract(right_lane_mask, lane_masks_intersection)

    y_cutoff = 0
    for y in range(previous_lane_data.lane_height - 1, -1, -1):
        if cv2.countNonZero(cv2.bitwise_and(left_lane_mask[y], right_lane_mask[y])) > 1:
            y_cutoff = y
            break

    for y in range(y_cutoff, -1, -1):
        left_lane_mask[y] = 0
        right_lane_mask[y] = 0

    next_lane_candidates = __pipeline_preprocesses(frame_BGR[skycutoff:hoodcutoff])

    scored_candidates = __pipeline_get_lanes_scores(previous_lane_data, next_lane_candidates,
                                                    left_lane_mask, right_lane_mask)

    best_candidate_left = min(filter(lambda _: _[0] != None, scored_candidates), key=lambda c: c[0][0])
    best_candidate_right = min(filter(lambda _: _[0] != None, scored_candidates), key=lambda c: c[0][1])

    get_next_lane.previous_lanes_average_score_left = 0
    get_next_lane.previous_lanes_average_score_right = 0
    for candidate in scored_candidates:
        if candidate[0] != None:
            get_next_lane.previous_lanes_average_score_left += candidate[0][0]
            get_next_lane.previous_lanes_average_score_right += candidate[0][1]

    get_next_lane.previous_lanes_average_score_left /= len(scored_candidates)
    get_next_lane.previous_lanes_average_score_right /= len(scored_candidates)

    try:

        get_next_lane.best_candidate_score_left_memory = get_next_lane.best_candidate_score_left_memory[1:]
        get_next_lane.best_candidate_score_left_memory.append(best_candidate_left[0][0])

        get_next_lane.best_candidate_score_right_memory = get_next_lane.best_candidate_score_right_memory[1:]
        get_next_lane.best_candidate_score_right_memory.append(best_candidate_right[0][1])

    except AttributeError:

        get_next_lane.best_candidate_score_left_memory = [1000] * 5
        get_next_lane.best_candidate_score_right_memory = [1000] * 5

    # TODO: If best score is below some threshold (1.25x w.r.t memory average sounds good)...
    # ...go ahead with ensemble, in increasing voting threshold order
    # If something gets a better score (that is below a slightly larger threshold), take it obviously lol
    # However if not, just re-use previous lane data

    use_previous = False
    voting_ensemble_candidates = []

    if best_candidate_left[0][0] > 1.25 * np.mean(get_next_lane.best_candidate_score_left_memory)\
            or best_candidate_right[0][1] > 1.25 * np.mean(get_next_lane.best_candidate_score_right_memory):

        for i in range(1, len(next_lane_candidates) + 1):
            voting_ensemble_candidates.append(prep.edge_voting_ensemble(next_lane_candidates, i))

        # Reminder that __pipeline_lanes_score may or may not apply Hough Transform based on my mood hehe
        scored_voting_ensemble_candidates = __pipeline_get_lanes_scores(previous_lane_data, voting_ensemble_candidates,
                                                                        left_lane_mask, right_lane_mask)

        best_voting_ensemble_candidate_left =\
            min(filter(lambda _: _[0] != None, scored_voting_ensemble_candidates),
                key=lambda c: c[0][0])

        best_voting_ensemble_candidate_right =\
            min(filter(lambda _: _[0] != None, scored_voting_ensemble_candidates),
                key=lambda c: c[0][1])

        best_candidate_left = min(best_candidate_left, best_voting_ensemble_candidate_left, key=lambda c: c[0][0])

        best_candidate_right = min(best_candidate_right, best_voting_ensemble_candidate_right, key=lambda c: c[0][1])

        if best_candidate_left[0][0] > 1.10 * np.mean(get_next_lane.best_candidate_score_left_memory) or \
                best_candidate_right[0][1] > 1.10 * np.mean(get_next_lane.best_candidate_score_right_memory):

            use_previous = True
            pass

    next_lane_data = previous_lane_data if use_previous else lane.lane_data(

        lane_height=best_candidate_left[1].lane_height,  # arbitrary choice, either of them work
        lane_width=best_candidate_left[1].lane_width,  # same ^
        l_pts=best_candidate_left[1].left_lane_pts,
        r_pts=best_candidate_right[1].right_lane_pts,
        lane_polyfit_rank=1

    )

    next_lane_canvas = next_lane_data.draw(frame_BGR, skycutoff, hoodcutoff, debugging)

    if debugging:

        all_candidates_scored = [scored_candidates]
        if len(voting_ensemble_candidates) != 0: all_candidates_scored.append(scored_voting_ensemble_candidates)

        all_candidates_marked_RGB = []
        for scored_candidates in all_candidates_scored:

            candidates_marked_RGB = []
            for candidate in scored_candidates:

                if candidate[0] == None:
                    candidate_RGB = cv2.cvtColor(candidate[2], cv2.COLOR_GRAY2RGB)

                else:

                    best_left = (candidate[1] == best_candidate_left[1])
                    best_right = (candidate[1] == best_candidate_right[1])

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
                    candidate_RGB = cv2.addWeighted(candidate_RGB, 1, lane_masks_RGB, 0.15, 8)

                candidates_marked_RGB.append(candidate_RGB)
            all_candidates_marked_RGB.append(candidates_marked_RGB)

        # Left score
        frame_BGR = cv2.putText(

            img=frame_BGR,
            text=f'B: {str(int(best_candidate_left[0][0])).zfill(5)}, '
            + f'A: {str(int(np.mean(get_next_lane.previous_lanes_average_score_left))).zfill(5)}, '
            + f'M(B): {str(int(np.mean(get_next_lane.best_candidate_score_left_memory))).zfill(5)}',
            bottomLeftOrigin=False,  # when false, it is at the top-left corner.
            org=(4, 16),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            color=(0, 0, 255),  # BGR
            lineType=cv2.LINE_AA

        )

        # Right score
        frame_BGR = cv2.putText(

            img=frame_BGR,
            text=f'B: {str(int(best_candidate_right[0][1])).zfill(5)}, '
            + f'A: {str(int(np.mean(get_next_lane.previous_lanes_average_score_right))).zfill(5)}, '
            + f'M(B): {str(int(np.mean(get_next_lane.best_candidate_score_right_memory))).zfill(5)}',
            bottomLeftOrigin=False,  # when false, it is at the top-left corner.
            org=(4, 32),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            color=(255, 0, 0),  # BGR
            lineType=cv2.LINE_AA

        )

        candidates_preview = []
        for candidates_marked_RGB in all_candidates_marked_RGB:
            candidates_marked_RGB = cv2.vconcat(candidates_marked_RGB)
            candidates_marked_RGB = cv2.resize(

                candidates_marked_RGB,
                (int(np.shape(candidates_marked_RGB)[1] * (np.shape(frame_BGR)[0] / np.shape(candidates_marked_RGB)[0])),
                 np.shape(frame_BGR)[0])

            )
            candidates_preview.append(candidates_marked_RGB)
        # Preview no longer done at this stage
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


def get_initial_lane(first_frame_BGR):

    HOODCUTOFF, _ = prep.get_hood_cutoff(first_frame_BGR)
    SKYCUTOFF, _, _ = prep.get_sky_cutoff(first_frame_BGR[:HOODCUTOFF], (1, 3))

    _, detection_ready_frame =\
        prep.hough_transform_raw_pre(
            prep.edge_voting_ensemble(
                __pipeline_preprocesses(
                    first_frame_BGR[SKYCUTOFF:HOODCUTOFF]
                ),
                voting_threshold=6
            )
        )

    detected_lane_data = lane.peeking_center_detect(detection_ready_frame, 1)

    return detected_lane_data, HOODCUTOFF, SKYCUTOFF


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
