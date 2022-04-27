import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore', category=np.RankWarning)


class lane_data:

    def __init__(self, lane_height, lane_width, range_poly, l_poly, r_poly, l_pts, r_pts, l_center_pts=None, r_center_pts=None):

        self.lane_width = lane_width
        self.lane_height = lane_height

        self.polyfit_range = range_poly
        self.left_lane_polyfit_pts = l_poly
        self.right_lane_polyfit_pts = r_poly

        self.left_lane_pts_x = list(zip(*l_pts))[1]
        self.left_lane_pts_y = list(zip(*l_pts))[0]
        self.right_lane_pts_x = list(zip(*r_pts))[1]
        self.right_lane_pts_y = list(zip(*r_pts))[0]

        self.left_peeking_center_pts_x = list(zip(*l_center_pts))[1] if l_center_pts else None
        self.left_peeking_center_pts_y = list(zip(*l_center_pts))[0] if l_center_pts else None
        self.right_peeking_center_pts_x = list(zip(*r_center_pts))[1] if r_center_pts else None
        self.right_peeking_center_pts_y = list(zip(*r_center_pts))[0] if r_center_pts else None

    def draw(self, background_image, skycutoff, hoodcutoff, debugging=False):

        color_intensity = 0
        lane = np.zeros_like(background_image[:hoodcutoff])
        for y in range(0, self.lane_height):
            color_intensity = (y / self.lane_height) * 127
            for x in range(int(self.left_lane_polyfit_pts[y]), int(self.right_lane_polyfit_pts[y] + 1)):
                if x >= 0 and x < self.lane_width:
                    lane[skycutoff + y][x] = [0, max(0, color_intensity), 0]

        if (debugging):

            # Left lane lines

            if (self.left_peeking_center_pts_x):
                lane = cv2.polylines(

                    img=lane,
                    pts=[np.array(list(zip(
                        self.left_peeking_center_pts_x, [skycutoff + a for a in self.left_peeking_center_pts_y]
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

            if (self.right_peeking_center_pts_x):
                lane = cv2.polylines(

                    img=lane,
                    pts=[np.array(list(zip(
                        self.right_peeking_center_pts_x, [skycutoff + a for a in self.right_peeking_center_pts_y]
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


def peeking_center_detect(preprocessed_image: cv2.Mat, polyfit_rank: int = 2) -> lane_data:

    width = np.shape(preprocessed_image)[1]
    height = np.shape(preprocessed_image)[0]
    HIT_THRESHOLD = height // 5
    CONSECUTIVE_BLANKS_THRESHOLD = height // 10

    # Left lane
    previous_hit = 0
    peeking_center = width // 2
    consecutive_blanks_counter = 0
    left_lane_pts = []
    left_peeking_center_pts = []
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
            left_peeking_center_pts.append([y, peeking_center])

        left_peeking_center_pts.append([y, peeking_center])

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
    right_peeking_center_pts = []
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
            right_peeking_center_pts.append([y, peeking_center])

        right_peeking_center_pts.append([y, peeking_center])

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

    if (len(left_lane_pts) <= 1): left_lane_pts = [[0, 0], [height - 1, 0]]
    if (len(right_lane_pts) <= 1): right_lane_pts = [[0, width - 1], [height - 1, width - 1]]

    polyfit_range = np.linspace(0, height - 1, height)
    left_lane_polyfit_pts = polyfit_lane_pts(left_lane_pts, polyfit_range, polyfit_rank)
    right_lane_polyfit_pts = polyfit_lane_pts(right_lane_pts, polyfit_range, polyfit_rank)

    return lane_data(

        lane_width=width,
        lane_height=height,

        range_poly=polyfit_range,
        l_poly=left_lane_polyfit_pts,
        r_poly=right_lane_polyfit_pts,

        l_pts=left_lane_pts,
        r_pts=right_lane_pts,

        l_center_pts=left_peeking_center_pts,
        r_center_pts=right_peeking_center_pts

    )


def polyfit_lane_pts(lane_pts: list[list], polyfit_range: np.ndarray, polyfit_rank: int):

    try:
        lane_pts_y = list(zip(*lane_pts))[0]
        lane_pts_x = list(zip(*lane_pts))[1]
        lane_coeffs = np.polyfit(lane_pts_y, lane_pts_x, polyfit_rank)
        lane_polyfit_pts = 0
        for i in range(polyfit_rank + 1): lane_polyfit_pts += lane_coeffs[i] * polyfit_range**(polyfit_rank - i)
        return lane_polyfit_pts
    except: print(lane_pts); return []


def extract_separate_ROIs(lane: lane_data):

    def __ROI_extractor(lane_pts):

        canvas_width = lane.lane_width
        lane_ROI = []
        for pt in lane_pts:
            if pt >= 0 and pt < canvas_width:
                lane_ROI.append([

                    round(max(0, pt - canvas_width // 16)),
                    round(min(canvas_width, pt + canvas_width // 16))

                ])
        return lane_ROI

    # Reminder that the polyfit points are 1-dimensional, they are the X's to the lane.polyfit_range
    return __ROI_extractor(lane.left_lane_polyfit_pts), __ROI_extractor(lane.right_lane_polyfit_pts)


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
