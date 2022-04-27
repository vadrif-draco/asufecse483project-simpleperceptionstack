import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore', category=np.RankWarning)


class __lane_data:
    def __init__(self, edges_canvas,
                 l_poly, r_poly, range_poly,
                 l_pts_x, l_pts_y, r_pts_x, r_pts_y,
                 l_center_trace_x: None, l_center_trace_y: None, r_center_trace_x: None, r_center_trace_y: None):

        self.edges_canvas = edges_canvas

        self.left_lane_polyfit_pts = l_poly
        self.right_lane_polyfit_pts = r_poly
        self.polyfit_range = range_poly

        self.left_lane_pts_x = l_pts_x
        self.left_lane_pts_y = l_pts_y
        self.right_lane_pts_x = r_pts_x
        self.right_lane_pts_y = r_pts_y

        self.left_peeking_center_trace_x = l_center_trace_x
        self.left_peeking_center_trace_y = l_center_trace_y
        self.right_peeking_center_trace_x = r_center_trace_x
        self.right_peeking_center_trace_y = r_center_trace_y

    def draw(self, background_image, skycutoff, hoodcutoff, debugging=False):

        color_intensity = 0
        lane = np.zeros_like(background_image[:hoodcutoff])
        for y in range(0, np.shape(self.edges_canvas)[0], 1):
            color_intensity = (2 * y * 255 / np.shape(lane)[0])
            for x in range(int(self.left_lane_polyfit_pts[y]) + 1, int(self.right_lane_polyfit_pts[y])):
                if x > 0 and x < np.shape(background_image)[1]:
                    lane[skycutoff + y][x] = [0, max(0, color_intensity), 0]

        if (debugging):

            # Left lane lines
            lane = cv2.polylines(

                img=lane,
                pts=[np.array(list(zip(
                    self.left_peeking_center_trace_x, [skycutoff + a for a in self.left_peeking_center_trace_y]
                ))).reshape(-1, 1, 2)],
                color=(255, 0, 0), thickness=1, isClosed=False, lineType=cv2.LINE_AA

            )

            lane = cv2.polylines(

                img=lane,
                pts=[np.array(list(zip(
                    self.left_lane_polyfit_pts, [skycutoff + a for a in self.polyfit_range]
                )), dtype=np.int32).reshape(-1, 1, 2)],
                color=(255, 0, 0), thickness=4, isClosed=False, lineType=cv2.LINE_AA

            )

            lane = cv2.polylines(

                img=lane,
                pts=[np.array(list(zip(
                    self.left_lane_pts_x, [skycutoff + a for a in self.left_lane_pts_y]
                ))).reshape(-1, 1, 2)],
                color=(255, 255, 0), thickness=2, isClosed=False

            )

            # Right lane lines
            lane = cv2.polylines(

                img=lane,
                pts=[np.array(list(zip(
                    self.right_peeking_center_trace_x, [skycutoff + a for a in self.right_peeking_center_trace_y]
                ))).reshape(-1, 1, 2)],
                color=(0, 0, 255), thickness=1, isClosed=False, lineType=cv2.LINE_AA

            )

            lane = cv2.polylines(

                img=lane,
                pts=[np.array(list(zip(
                    self.right_lane_polyfit_pts, [skycutoff + a for a in self.polyfit_range]
                )), dtype=np.int32).reshape(-1, 1, 2)],
                color=(0, 0, 255), thickness=4, isClosed=False, lineType=cv2.LINE_AA

            )

            lane = cv2.polylines(

                img=lane,
                pts=[np.array(list(zip(
                    self.right_lane_pts_x, [skycutoff + a for a in self.right_lane_pts_y]
                ))).reshape(-1, 1, 2)],
                color=(0, 255, 255), thickness=2, isClosed=False

            )

        return lane


def peeking_center_detect(preprocessed_image: cv2.Mat, polyfit_rank: int = 2) -> __lane_data:

    width = np.shape(preprocessed_image)[1]
    height = np.shape(preprocessed_image)[0]
    HIT_THRESHOLD = height // 5
    CONSECUTIVE_BLANKS_THRESHOLD = height // 10

    # Left lane
    previous_hit = 0
    peeking_center = width // 2
    consecutive_blanks_counter = 0
    left_lane_pts = []
    left_peeking_center_trace_y = []
    left_peeking_center_trace_x = []
    for y in range(height - 1, 2, -1):  # y-axis, bottom-up

        while peeking_center > 0 and peeking_center < width - 1 and (

            preprocessed_image[y][peeking_center] == 255 or
            preprocessed_image[y - 1][peeking_center - 1] == 255 or
            preprocessed_image[y - 2][peeking_center - 1] == 255 or
            preprocessed_image[y - 3][peeking_center - 1] == 255 or
            preprocessed_image[y - 1][peeking_center] == 255 or
            preprocessed_image[y - 2][peeking_center] == 255 or
            preprocessed_image[y - 3][peeking_center] == 255 or
            preprocessed_image[y - 1][peeking_center + 1] == 255 or
            preprocessed_image[y - 2][peeking_center + 1] == 255 or
            preprocessed_image[y - 3][peeking_center + 1] == 255

        ):
            
            peeking_center -= 2
            if (peeking_center < 0): peeking_center = 0; break
            left_peeking_center_trace_x.append(peeking_center)
            left_peeking_center_trace_y.append(y)

        left_peeking_center_trace_x.append(peeking_center)
        left_peeking_center_trace_y.append(y)

        for x in range(peeking_center, -1, -1):  # x-axis, center-to-left

            if (x > previous_hit) or (previous_hit == 0) or (abs(x - previous_hit) < width // 160):
                
                if preprocessed_image[y][x] == 255:
                    consecutive_blanks_counter = 0
                    left_lane_pts.append([y, x])
                    previous_hit = x
                    break

                else: preprocessed_image[y][x] = 127

        consecutive_blanks_counter += 1
        if (consecutive_blanks_counter > CONSECUTIVE_BLANKS_THRESHOLD
            and len(left_lane_pts) > HIT_THRESHOLD): break

    # Right lane
    previous_hit = width - 1
    peeking_center = width // 2
    consecutive_blanks_counter = 0
    right_lane_pts = []
    right_peeking_center_trace_y = []
    right_peeking_center_trace_x = []
    for y in range(height - 1, 2, -1):  # y-axis, bottom-up

        while peeking_center > 0 and peeking_center < width - 1 and (

            preprocessed_image[y][peeking_center] == 255 or
            preprocessed_image[y - 1][peeking_center - 1] == 255 or
            preprocessed_image[y - 2][peeking_center - 1] == 255 or
            preprocessed_image[y - 3][peeking_center - 1] == 255 or
            preprocessed_image[y - 1][peeking_center] == 255 or
            preprocessed_image[y - 2][peeking_center] == 255 or
            preprocessed_image[y - 3][peeking_center] == 255 or
            preprocessed_image[y - 1][peeking_center + 1] == 255 or
            preprocessed_image[y - 2][peeking_center + 1] == 255 or
            preprocessed_image[y - 3][peeking_center + 1] == 255

        ):

            peeking_center += 2
            if (peeking_center >= width): peeking_center = width - 1; break
            right_peeking_center_trace_x.append(peeking_center)
            right_peeking_center_trace_y.append(y)

        right_peeking_center_trace_x.append(peeking_center)
        right_peeking_center_trace_y.append(y)

        for x in range(peeking_center, width, 1):  # x-axis, center-to-right

            if (x < previous_hit) or (previous_hit == width - 1) or (abs(x - previous_hit) < width // 160):
                
                if preprocessed_image[y][x] == 255:
                    consecutive_blanks_counter = 0
                    right_lane_pts.append([y, x])
                    previous_hit = x
                    break

                else: preprocessed_image[y][x] = 127

        consecutive_blanks_counter += 1
        if (consecutive_blanks_counter > CONSECUTIVE_BLANKS_THRESHOLD
            and len(right_lane_pts) > HIT_THRESHOLD): break

    polyfit_range = np.linspace(0, np.shape(preprocessed_image)[0] - 1, np.shape(preprocessed_image)[0])

    if (len(left_lane_pts) <= 1): left_lane_pts = [[0, 0], [height - 1, 0]]
    left_lane_pts_y = list(zip(*left_lane_pts))[0]
    left_lane_pts_x = list(zip(*left_lane_pts))[1]
    left_lane_coeffs = np.polyfit(left_lane_pts_y, left_lane_pts_x, polyfit_rank)

    if (len(right_lane_pts) <= 1): right_lane_pts = [[0, width - 1], [height - 1, width - 1]]
    right_lane_pts_y = list(zip(*right_lane_pts))[0]
    right_lane_pts_x = list(zip(*right_lane_pts))[1]
    right_lane_coeffs = np.polyfit(right_lane_pts_y, right_lane_pts_x, polyfit_rank)

    left_lane_polyfit_pts = 0
    right_lane_polyfit_pts = 0
    for i in range(polyfit_rank + 1):
        left_lane_polyfit_pts += left_lane_coeffs[i] * polyfit_range**(polyfit_rank - i)
        right_lane_polyfit_pts += right_lane_coeffs[i] * polyfit_range**(polyfit_rank - i)

    # left_lane_coeffs = np.polyfit(left_lane_pts_y, left_lane_pts_x, 1)
    # left_lane_coeffs = np.polyfit(left_lane_pts_y, left_lane_pts_x, 2)
    # left_lane_coeffs = np.polyfit(left_lane_pts_y, left_lane_pts_x, 3)

    # left_lane_polyfit_pts =\
    #     left_lane_coeffs[0] * polyfit_range**1 +\
    #     left_lane_coeffs[1] * polyfit_range**0
    # left_lane_polyfit_pts =\
    #     left_lane_coeffs[0] * polyfit_range**2 +\
    #     left_lane_coeffs[1] * polyfit_range**1 +\
    #     left_lane_coeffs[2] * polyfit_range**0
    # left_lane_polyfit_pts =\
    #     left_lane_coeffs[0] * polyfit_range**3 +\
    #     left_lane_coeffs[1] * polyfit_range**2 +\
    #     left_lane_coeffs[2] * polyfit_range**1 +\
    #     left_lane_coeffs[3] * polyfit_range**0\

    # right_lane_coeffs = np.polyfit(right_lane_pts_y, right_lane_pts_x, 1)
    # right_lane_coeffs = np.polyfit(right_lane_pts_y, right_lane_pts_x, 2)
    # right_lane_coeffs = np.polyfit(right_lane_pts_y, right_lane_pts_x, 3)

    # right_lane_polyfit_pts =\
    #     right_lane_coeffs[0] * polyfit_range**1 +\
    #     right_lane_coeffs[1] * polyfit_range**0
    # right_lane_polyfit_pts =\
    #     right_lane_coeffs[0] * polyfit_range**2 +\
    #     right_lane_coeffs[1] * polyfit_range**1 +\
    #     right_lane_coeffs[2] * polyfit_range**0
    # right_lane_polyfit_pts =\
    #     right_lane_coeffs[0] * polyfit_range**3 +\
    #     right_lane_coeffs[1] * polyfit_range**2 +\
    #     right_lane_coeffs[2] * polyfit_range**1 +\
    #     right_lane_coeffs[3] * polyfit_range**0

    return __lane_data(

        edges_canvas=preprocessed_image,

        l_poly=left_lane_polyfit_pts, r_poly=right_lane_polyfit_pts, range_poly=polyfit_range,

        l_pts_x=left_lane_pts_x, l_pts_y=left_lane_pts_y, r_pts_x=right_lane_pts_x, r_pts_y=right_lane_pts_y,

        l_center_trace_x=left_peeking_center_trace_x, l_center_trace_y=left_peeking_center_trace_y,
        r_center_trace_x=right_peeking_center_trace_x, r_center_trace_y=right_peeking_center_trace_y

    )
