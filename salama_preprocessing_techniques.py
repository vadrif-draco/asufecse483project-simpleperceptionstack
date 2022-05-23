from typing import Tuple
import numpy as np
import cv2


def __get_edgiest_row(edgy_image: cv2.Mat):

    # edge_threshold = np.average(edgy_image)
    # consecutive_hits_threshold = np.std(edgy_image) / 3.5
    # print(consecutive_hits_threshold)

    # consecutive_hits = 0
    # for i, row in enumerate(edgy_image):
    #     if np.average(row) > edge_threshold:
    #         consecutive_hits += 1
    #     else:
    #         consecutive_hits = 0
    #     if consecutive_hits >= consecutive_hits_threshold: return i

    # return np.shape(edgy_image)[0] / 2

    rows_averages = [

        np.mean(row) for row in  # Only look in the middle six eigths of the image
        edgy_image[(np.shape(edgy_image)[0] // 8):(7 * np.shape(edgy_image)[0] // 8)]

    ]
    return(np.shape(edgy_image)[0] // 8 + rows_averages.index(np.max(rows_averages)))


def get_hood_cutoff(image_BGR: cv2.Mat):

    width = np.shape(image_BGR)[1]
    height = np.shape(image_BGR)[0]

    image_step1 = image_BGR[height - int(height / 10):]
    image_step2 = cv2.cvtColor(image_step1, cv2.COLOR_BGR2GRAY)
    image_step3 = cv2.GaussianBlur(image_step2, (5, 5), 0)

    CANNY_THRESHOLD = (np.mean(image_step3) - np.std(image_step3), np.mean(image_step3) + np.std(image_step3))
    # CANNY_THRESHOLD = (85, 255)
    image_step4 = cv2.Canny(image_step3, CANNY_THRESHOLD[0], CANNY_THRESHOLD[1], L2gradient=True)

    # cum_i = 0; n_i = 0
    # for i, row in enumerate(image_step4):
    #     for cell in row:
    #         if cell == 255:
    #             cum_i += i
    #             n_i += 1

    # hood_cutoff = round(cum_i / n_i) if n_i else height / 20

    hood_cutoff = min(np.shape(image_step4)[0] // 2, __get_edgiest_row(image_step4))

    image_step5 = np.copy(image_step1)
    cv2.line(image_step5, (0, hood_cutoff), (width - 1, hood_cutoff), (0, 0, 255), 1, cv2.LINE_AA)

    hood_cutoff = height - int(height / 10) + hood_cutoff

    return hood_cutoff, [

        [cv2.cvtColor(image_step1, cv2.COLOR_BGR2RGB), "Step 1: Original, roughly cropped to show hood portion only"],
        [image_step2, "Step 2: Grayed"],
        [image_step3, "Step 3: Blurred"],
        [image_step4, "Step 4: Canny applied with dynamic edge threshold"],
        [cv2.cvtColor(image_step5, cv2.COLOR_BGR2RGB), "Step 5: Hood cut-off estimate drawn"],

    ]


def get_sky_cutoff(image_BGR: cv2.Mat, sobel_blur_ksize: Tuple = None, post_sobel_threshold=85):

    if (sobel_blur_ksize == None):
        sobel_blur_ksize = [

            1 + 2 * (np.shape(image_BGR)[1] // 80),
            1 + 2 * (np.shape(image_BGR)[0] // 40),

        ]

    image_HLS = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HLS)
    image_HSV = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)

    image_HLS_L = image_HLS[:, :, 1]

    image_HLS_S = image_HLS[:, :, 2]
    image_LS_hybrid = cv2.addWeighted(image_HLS_L, 1.5, image_HLS_S, 2.5, -255)

    image_HSV_V = image_HSV[:, :, 2]
    image_VS_hybrid = cv2.addWeighted(image_HSV_V, 1.5, image_HLS_S, 2.5, -255)

    image_HLS_S_blur_sobely_thresholded, image_HLS_S_blur_sobely =\
        sobel_pre(image_HLS_S, sobel_blur_ksize, post_sobel_threshold)

    image_LS_hybrid_blur_sobely_thresholded, image_LS_hybrid_blur_sobely =\
        sobel_pre(image_LS_hybrid, sobel_blur_ksize, post_sobel_threshold)

    image_VS_hybrid_blur_sobely_thresholded, image_VS_hybrid_blur_sobely =\
        sobel_pre(image_VS_hybrid, sobel_blur_ksize, post_sobel_threshold)

    potential_sky_cutoffs = [

        __get_edgiest_row(image_HLS_S_blur_sobely_thresholded),
        __get_edgiest_row(image_LS_hybrid_blur_sobely_thresholded),
        __get_edgiest_row(image_VS_hybrid_blur_sobely_thresholded)

    ]

    sky_cutoff = max((np.shape(image_BGR)[0] // 5) * 3, round(np.average(potential_sky_cutoffs)))

    return sky_cutoff, potential_sky_cutoffs, [

        [image_HLS_S, "S Channel"],
        [image_HLS_S_blur_sobely, "S Channel Sobel Edges"],
        [image_HLS_S_blur_sobely_thresholded, "S Channel Sobel Edges - Thresholded"],

        [image_LS_hybrid, "LS Hybrid"],
        [image_LS_hybrid_blur_sobely, "LS Hybrid Sobel Edges"],
        [image_LS_hybrid_blur_sobely_thresholded, "LS Hybrid Sobel Edges - Thresholded"],

        [image_VS_hybrid, "VS Hybrid"],
        [image_VS_hybrid_blur_sobely, "VS Hybrid Sobel Edges"],
        [image_VS_hybrid_blur_sobely_thresholded, "VS Hybrid Sobel Edges - Thresholded"],

    ]


def __apply_thresholds(image_in_target_colorspace: cv2.Mat, thresholds: Tuple[Tuple[Tuple]]):

    # A list of thresholds
    # Each threshold carries three channels
    # Each channel has a start and end range
    # Hence the 3D Tuple
    intermediate_outputs = []
    for threshold in thresholds:

        # New method

        intermediate_output_mask = cv2.inRange(image_in_target_colorspace, threshold[0], threshold[1])

        intermediate_outputs.append(cv2.bitwise_and(image_in_target_colorspace,
                                                    image_in_target_colorspace,
                                                    mask=intermediate_output_mask))

        # Old method

        # intermediate_output = np.copy(image_in_target_colorspace)

        # for i, row in enumerate(intermediate_output):
        #     for j, pixel in enumerate(row):
        #         if not(pixel[0] >= threshold[0][0] and pixel[0] <= threshold[0][1] and
        #                pixel[1] >= threshold[1][0] and pixel[1] <= threshold[1][1] and
        #                pixel[2] >= threshold[2][0] and pixel[2] <= threshold[2][1]): intermediate_output[i, j] = [0, 0, 0]

        # intermediate_outputs.append(intermediate_output)

    output = np.zeros_like(image_in_target_colorspace)
    for intermediate_output in intermediate_outputs: output = cv2.bitwise_or(output, intermediate_output)
    return output, intermediate_outputs


class __lane_thresholds_colorspace:
    def __init__(self, thresholds: Tuple[Tuple[Tuple]], from_BGR: int, to_BGR: int):
        self.thresholds = thresholds
        self.from_BGR = from_BGR
        self.to_BGR = to_BGR


HLS_COLORSPACE_THRESH = __lane_thresholds_colorspace(thresholds=[

    # Old method format
    # [(16, 26), (0, 255), (127, 255)],  # Yellows
    # [(0, 255), (224, 255), (0, 255)],  # Whites

    # New method format
    [np.array([16, 0, 127], dtype=np.uint8), np.array([26, 255, 255], dtype=np.uint8)],  # Yellows
    [np.array([0, 224, 0], dtype=np.uint8), np.array([255, 255, 255], dtype=np.uint8)],  # White

], from_BGR=cv2.COLOR_BGR2HLS, to_BGR=cv2.COLOR_HLS2BGR)


LAB_COLORSPACE_THRESH = __lane_thresholds_colorspace(thresholds=[

    # Old range old method format
    # [(80, 255), (127, 191), (127, 255)], # Yellows
    # [(223, 255), (31, 223), (31, 223)], # Whites

    # New range old method format
    # [(20, 255), (110, 159), (159, 215)],  # Yellows
    # [(239, 255), (96, 159), (96, 159)],  # Whites

    # New range new method format
    [np.array([20, 110, 159], dtype=np.uint8), np.array([255, 159, 215], dtype=np.uint8)],  # Yellows
    [np.array([239, 96, 96], dtype=np.uint8), np.array([255, 159, 159], dtype=np.uint8)]  # Whites

], from_BGR=cv2.COLOR_BGR2LAB, to_BGR=cv2.COLOR_LAB2BGR)


def color_thresholded_edges_pre(image_BGR: cv2.Mat, colorspace: __lane_thresholds_colorspace):

    image_in_target_colorspace = cv2.cvtColor(image_BGR, colorspace.from_BGR)

    thresholded_frame, intermediate_threshold_steps =\
        __apply_thresholds(image_in_target_colorspace, colorspace.thresholds)

    return \
        cv2.threshold(

            cv2.Sobel(
                # cv2.Canny(

                cv2.medianBlur(

                    cv2.GaussianBlur(

                        cv2.cvtColor(

                            cv2.cvtColor(

                                thresholded_frame,
                                colorspace.to_BGR

                            ), cv2.COLOR_BGR2GRAY

                        ), (1 + 2 * (np.shape(thresholded_frame)[1] // 640), 1 + 2 * (np.shape(thresholded_frame)[0] // 80)), 0

                    ), 1 + 2 * (np.shape(thresholded_frame)[1] // 640)

                ), cv2.CV_8U, 0, 1
                # ), 85, 255

            ), 0, 255, cv2.THRESH_BINARY

        )[1], thresholded_frame, intermediate_threshold_steps


def adaptive_thresholding_pre(image_mono: cv2.Mat, threshold_block_size: int = None, post_blur: int = 3):

    if threshold_block_size == None: threshold_block_size = 3 + 2 * (np.shape(image_mono)[1] // 80)
    return cv2.medianBlur(

        cv2.adaptiveThreshold(

            image_mono, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, threshold_block_size, -(np.average(image_mono) // 40)

        ), post_blur)


def sobel_pre(image_mono: cv2.Mat, gaussian_blur_ksize: Tuple = (5, 5), binarization_threshold=50):

    image_blur = cv2.GaussianBlur(image_mono, ksize=gaussian_blur_ksize, sigmaX=0, sigmaY=0)
    image_sobel_raw = cv2.Sobel(image_blur, cv2.CV_8U, 0, 1)
    _, image_sobel_binary = cv2.threshold(image_sobel_raw,
                                          binarization_threshold,
                                          255,
                                          cv2.THRESH_BINARY)

    return image_sobel_binary, image_sobel_raw


def morph_top_hat_pre(image_BGR: cv2.Mat):

    image_GRAY = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)
    image_TOPHAT = cv2.morphologyEx(

        image_GRAY,
        cv2.MORPH_TOPHAT,
        # cv2.getStructuringElement(cv2.MORPH_RECT, (37, 37))
        cv2.getStructuringElement(cv2.MORPH_RECT, (1 + 2 * (np.shape(image_GRAY)[1] // 160),
                                                   1 + 2 * (np.shape(image_GRAY)[0] // 80)))

    )

    _, image_TOPHAT_thresholded = cv2.threshold(image_TOPHAT, 31, 255, cv2.THRESH_BINARY)

    return image_TOPHAT_thresholded, image_TOPHAT


def morph_inv_black_hat_pre(image_BGR: cv2.Mat):

    image_GRAY = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)
    image_BLACKHAT = cv2.morphologyEx(

        image_GRAY,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1 + 2 * (np.shape(image_GRAY)[1] // 4),
                                                   1 + 2 * (np.shape(image_GRAY)[0] // 4)))

    )

    _, image_BLACKHAT_thresholded = cv2.threshold(image_BLACKHAT, 15, 255, cv2.THRESH_BINARY)
    image_inv_BLACKHAT_thresholded = cv2.bitwise_not(image_BLACKHAT_thresholded)

    return image_inv_BLACKHAT_thresholded, image_BLACKHAT_thresholded, image_BLACKHAT


def canny_raw_pre(image_mono: cv2.Mat, gaussian_blur_ksize: Tuple = (5, 5), canny_hysteresis_thresh: Tuple = (80, 240)):

    image_mono_blurred = cv2.GaussianBlur(image_mono, gaussian_blur_ksize, 0)
    image_mono_blurred_canny = cv2.Canny(image_mono_blurred, canny_hysteresis_thresh[0], canny_hysteresis_thresh[1])
    return image_mono_blurred_canny


def hough_transform_raw_pre(edge_image: cv2.Mat):

    edges_detected = cv2.HoughLinesP(

        image=edge_image,  # input, should be of just edges
        rho=np.shape(edge_image)[1] // 160,  # grid rho increment
        theta=np.pi / 180,  # grid theta increment... no need to make it dynamic, it's an angle
        # it basically means the increment between every other angle to be checked
        # for example, theta=np.pi=180deg means check every 180deg (i.e., vertical lines only)
        # theta=np.pi/2=90deg means every 90deg (so, vertical and horizontal)
        # so on... so the current setting np.pi/180=1deg means every degree
        threshold=np.shape(edge_image)[1] // 80,  # voting threshold
        maxLineGap=np.shape(edge_image)[1] // 320,  # maximum length of gap
        minLineLength=np.shape(edge_image)[1] // 160,  # minimum length of edge

    )

    edges_canvas = np.zeros_like(edge_image)
    try:
        for edge in edges_detected:
            cv2.line(edges_canvas, (edge[0][0], edge[0][1]), (edge[0][2], edge[0][3]), 255, 2)
    except:
        print("No edges found... Returning black canvas")

    return edges_detected, edges_canvas


def edge_voting_ensemble(edge_images: Tuple[cv2.Mat], voting_threshold: int):

    # Assert that voting threshold is less than number of images (otherwise, it will always fail, obviously)
    assert(voting_threshold <= len(edge_images))
    
    # Assert that voting threshold is not 0 (because that gives a trivial result of an all-white canvas)
    assert(voting_threshold != 0)

    canvas = np.zeros_like(edge_images[0])
    for edge_image in edge_images:
        canvas = np.add(canvas, np.divide(edge_image, 255))

    # for i in range(np.shape(edge_images[0][1])):
    #     for j in range(np.shape(edge_images[0][0])):
    #         for edge_image in edge_images:
    #             canvas[i][j] += (edge_image[i][j] == 255)

    # for i, row in enumerate(canvas):
    #     for j, pixel in enumerate(row):
    #         canvas[i][j] = 255 if pixel >= voting_threshold else 0

    canvas = cv2.inRange(canvas, voting_threshold, 255)
    return canvas
