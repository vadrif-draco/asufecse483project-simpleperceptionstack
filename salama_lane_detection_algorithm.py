import math
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore', category=np.RankWarning)


class lane_data:

    def __init__(self,
                 lane_width,
                 lane_height,
                 l_pts, r_pts,
                 lane_polyfit_rank,
                 l_center_pts=None, r_center_pts=None):

        self.lane_width = lane_width
        self.lane_height = lane_height

        self.left_lane_pts = l_pts
        self.left_lane_pts_y = list(zip(*l_pts))[0]
        self.left_lane_pts_x = list(zip(*l_pts))[1]

        self.right_lane_pts = r_pts
        self.right_lane_pts_y = list(zip(*r_pts))[0]
        self.right_lane_pts_x = list(zip(*r_pts))[1]

        self.polyfit_range = np.linspace(0, self.lane_height - 1, self.lane_height)
        
        self.left_lane_polyfit_pts, self.left_lane_polyfit_coeffs,\
            = polyfit_lane_pts(self.left_lane_pts, self.polyfit_range, lane_polyfit_rank)
        
        self.right_lane_polyfit_pts, self.right_lane_polyfit_coeffs,\
            = polyfit_lane_pts(self.right_lane_pts, self.polyfit_range, lane_polyfit_rank)

        self.left_peeking_center_pts = l_center_pts
        if (self.left_peeking_center_pts != None):
            self.left_peeking_center_pts_x = list(zip(*l_center_pts))[1]
            self.left_peeking_center_pts_y = list(zip(*l_center_pts))[0]

        self.right_peeking_center_pts = r_center_pts
        if (self.right_peeking_center_pts != None):
            self.right_peeking_center_pts_x = list(zip(*r_center_pts))[1]
            self.right_peeking_center_pts_y = list(zip(*r_center_pts))[0]

    def __ROI_extractor(self, lane_pts, ROI_width):

        # lane_ROI = []
        # for pt in lane_pts:
        #     if pt >= 0 and pt < self.lane_width:
        #         lane_ROI.append([

        #             # round(max(0, pt - (ROI_width // 2))),
        #             # round(min(self.lane_width, pt + (ROI_width // 2)))

        #         ])

        # return lane_ROI

        #####################################################################################################

        # return np.array([

        #     # Notice: XY tuples, not YX!
        #     [round(max(0, lane_pts[0] - (ROI_width / 2))), 0],  # TOPLEFT
        #     [round(lane_pts[0] + (ROI_width / 2)), 0],  # TOPRIGHT
        #     [round(min(self.lane_width, lane_pts[-1] + (ROI_width / 2))), self.lane_height - 1],  # BOTTRIGHT
        #     [round(lane_pts[-1] - (ROI_width / 2)), self.lane_height - 1],  # BOTTLEFT

        # ])

        #####################################################################################################

        lane_ROI = []
        for i in range(1, len(lane_pts) - 1):
            if lane_pts[i] >= 0 and lane_pts[i] < self.lane_width:

                # Get perpendicular to line at this point
                # To do so, need the slope at this line
                # Can get it by polyfitting this point and its neighbors

                slope, _ = np.polyfit(x=list(range(i - 1, i + 1 + 1)),
                                      y=lane_pts[i - 1:i + 1 + 1],
                                      deg=1)

                # Then use the perpendicular to their slope at the current point
                # To get two equidistant points from it, with displacement ROI_width

                lane_ROI.append([

                    (
                        lane_pts[i] + (ROI_width / 2) * math.sin(math.atan(-1 / slope)),
                        i + (ROI_width / 2) * math.cos(math.atan(-1 / slope)),
                    ),
                    (
                        lane_pts[i] - (ROI_width / 2) * math.sin(math.atan(-1 / slope)),
                        i - (ROI_width / 2) * math.cos(math.atan(-1 / slope)),
                    )

                ])

        return lane_ROI

    def extract_left_ROI(self, ROI_width):

        # Reminder that the polyfit points are 1-dimensional, they are the X's to the lane.polyfit_range
        return self.__ROI_extractor(self.left_lane_polyfit_pts, ROI_width)

    def extract_right_ROI(self, ROI_width):

        # Reminder that the polyfit points are 1-dimensional, they are the X's to the lane.polyfit_range
        return self.__ROI_extractor(self.right_lane_polyfit_pts, ROI_width)

    def extract_mask(self, lane_ROI):

        # lane_mask = np.zeros((self.lane_height, self.lane_width), dtype=np.uint8)

        # for y, pair in enumerate(lane_ROI):
        #     for x in range(pair[0], pair[1]):
        #         lane_mask[y][x] = 255

        #####################################################################################################

        # cv2.fillPoly(lane_mask, pts=[lane_ROI], color=255)

        #####################################################################################################

        # cv2.polyline of thickness 2 pixels
        lane_mask = np.zeros((self.lane_height, self.lane_width), dtype=np.uint8)
        for pair in lane_ROI: lane_mask = cv2.polylines(img=lane_mask,
                                                        pts=[np.array(pair, dtype=np.int32)],
                                                        isClosed=False,
                                                        thickness=2,
                                                        color=255)

        return lane_mask

    def draw(self, background_image, skycutoff, hoodcutoff, debugging):

        color_intensity = 0
        lane = np.zeros_like(background_image[skycutoff:hoodcutoff])
        for y in range(0, self.lane_height):
            color_intensity = (y / self.lane_height) * 127
            for x in range(int(self.left_lane_polyfit_pts[y]), int(self.right_lane_polyfit_pts[y] + 1)):
                if x >= 0 and x < self.lane_width:
                    lane[y][x] = [0, max(0, color_intensity), 0]

        if (debugging):

            # Left lane lines

            if (self.left_peeking_center_pts):
                lane = cv2.polylines(

                    img=lane,
                    # pts=[np.array(self.left_peeking_center_pts).reshape(-1, 1, 2)],
                    pts=[np.array(list(zip(
                        self.left_peeking_center_pts_x, [skycutoff + a for a in self.left_peeking_center_pts_y]
                    ))).reshape(-1, 1, 2)],
                    color=(0, 0, 255), thickness=1, isClosed=False, lineType=cv2.LINE_AA

                )

            lane = cv2.polylines(

                img=lane,
                pts=[np.array(list(zip(
                    self.left_lane_polyfit_pts, self.polyfit_range
                )), dtype=np.int32).reshape(-1, 1, 2)],
                color=(0, 0, 255), thickness=4, isClosed=False, lineType=cv2.LINE_AA

            )

            lane = cv2.polylines(

                img=lane,
                # pts=[np.array(self.left_lane_pts).reshape(-1, 1, 2)],
                pts=[np.array(list(zip(
                    self.left_lane_pts_x, self.left_lane_pts_y
                ))).reshape(-1, 1, 2)],
                color=(0, 255, 255), thickness=1, isClosed=False

            )

            # Right lane lines

            if (self.right_peeking_center_pts):
                lane = cv2.polylines(

                    img=lane,
                    # pts=[np.array(self.right_peeking_center_pts).reshape(-1, 1, 2)],
                    pts=[np.array(list(zip(
                        self.right_peeking_center_pts_x, self.right_peeking_center_pts_y
                    ))).reshape(-1, 1, 2)],
                    color=(255, 0, 0), thickness=1, isClosed=False, lineType=cv2.LINE_AA

                )

            lane = cv2.polylines(

                img=lane,
                pts=[np.array(list(zip(self.right_lane_polyfit_pts, self.polyfit_range)),
                              dtype=np.int32).reshape(-1, 1, 2)],
                color=(255, 0, 0), thickness=4, isClosed=False, lineType=cv2.LINE_AA

            )

            lane = cv2.polylines(

                img=lane,
                # pts=[np.array(self.right_lane_pts).reshape(-1, 1, 2)],
                pts=[np.array(list(zip(
                    self.right_lane_pts_x, self.right_lane_pts_y
                ))).reshape(-1, 1, 2)],
                color=(255, 255, 0), thickness=1, isClosed=False

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

    return lane_data(

        lane_width=width,
        lane_height=height,

        l_pts=left_lane_pts,
        r_pts=right_lane_pts,

        lane_polyfit_rank=polyfit_rank,

        l_center_pts=left_peeking_center_pts,
        r_center_pts=right_peeking_center_pts

    )


def polyfit_lane_pts(lane_pts: list[list], polyfit_range: np.ndarray, polyfit_rank: int):

    try:
        lane_pts_y = list(zip(*lane_pts))[0]
        lane_pts_x = list(zip(*lane_pts))[1]
        lane_polyfit_coeffs = np.polyfit(lane_pts_y, lane_pts_x, polyfit_rank)
        lane_polyfit_pts = 0

        for i in range(polyfit_rank + 1):
            lane_polyfit_pts += lane_polyfit_coeffs[i] * polyfit_range**(polyfit_rank - i)

        return lane_polyfit_pts, lane_polyfit_coeffs

    except:
        print(lane_pts)
        return []


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
