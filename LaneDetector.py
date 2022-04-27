from Utils import *



class LaneDetector:

    def __init__(self):

        self.IMG_SHAPE = (720, 1280)

        self.cache = np.array([])
        self.attempts = 0
        self.reset = True

        self.x_mppx = 0.0064
        self.y_mppx = 0.0291

        self.poly_param = None
    

    def get_poly_points(self, left_fit, right_fit):
        '''
        Get the points for the left lane/ right lane defined by the polynomial coeff's 'left_fit'
        and 'right_fit'
        :param left_fit (ndarray): Coefficients for the polynomial that defines the left lane line
        :param right_fit (ndarray): Coefficients for the polynomial that defines the right lane line
        : return (Tuple(ndarray, ndarray, ndarray, ndarray)): x-y coordinates for the left and right lane lines
        '''
        height, width = self.IMG_SHAPE
        
        # Get the points for the entire height of the image
        plot_y = np.linspace(0, height-1, height)
        plot_xleft = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
        plot_xright = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]
        
        # But keep only those points that lie within the image
        plot_xleft = plot_xleft[(plot_xleft >= 0) & (plot_xleft <= width - 1)]
        plot_xright = plot_xright[(plot_xright >= 0) & (plot_xright <= width - 1)]
        plot_yleft = np.linspace(height - len(plot_xleft), height - 1, len(plot_xleft))
        plot_yright = np.linspace(height - len(plot_xright), height - 1, len(plot_xright))
        
        return plot_xleft.astype(int), plot_yleft.astype(int), plot_xright.astype(int), plot_yright.astype(int)
                    
    def check_validity(self, left_fit, right_fit, debugging=False):
        '''
        Determine the validity of lane lines represented by a set of second order polynomial coefficients 
        :param left_fit (ndarray): Coefficients for the 2nd order polynomial that defines the left lane line
        :param right_fit (ndarray): Coefficients for the 2nd order polynomial that defines the right lane line
        :param debugging (boolean): Boolean flag for logging
        : return (boolean)
        '''
        
        if left_fit is None or right_fit is None:
            return False
        
        plot_xleft, plot_yleft, plot_xright, plot_yright = self.get_poly_points(left_fit, right_fit)

        # Check whether the two lines lie within a plausible distance from one another for three distinct y-values

        y1 = self.IMG_SHAPE[0] - 1 # Bottom
        y2 = self.IMG_SHAPE[0] - int(min(len(plot_yleft), len(plot_yright)) * 0.35) # For the 2nd and 3rd, take values between y1 and the top-most available value.
        y3 = self.IMG_SHAPE[0] - int(min(len(plot_yleft), len(plot_yright)) * 0.75)

        # Compute the respective x-values for both lines
        x1l = left_fit[0]  * (y1**2) + left_fit[1]  * y1 + left_fit[2]
        x2l = left_fit[0]  * (y2**2) + left_fit[1]  * y2 + left_fit[2]
        x3l = left_fit[0]  * (y3**2) + left_fit[1]  * y3 + left_fit[2]

        x1r = right_fit[0] * (y1**2) + right_fit[1] * y1 + right_fit[2]
        x2r = right_fit[0] * (y2**2) + right_fit[1] * y2 + right_fit[2]
        x3r = right_fit[0] * (y3**2) + right_fit[1] * y3 + right_fit[2]

        # Compute the L1 norms
        x1_diff = abs(x1l - x1r)
        x2_diff = abs(x2l - x2r)
        x3_diff = abs(x3l - x3r)

        # Define the threshold values for each of the three points
        min_dist_y1 = 480 # 510 # 530 
        max_dist_y1 = 730 # 750 # 660
        min_dist_y2 = 280
        max_dist_y2 = 730 # 660
        min_dist_y3 = 140
        max_dist_y3 = 730 # 660
        
        if (x1_diff < min_dist_y1) | (x1_diff > max_dist_y1) | \
            (x2_diff < min_dist_y2) | (x2_diff > max_dist_y2) | \
            (x3_diff < min_dist_y3) | (x3_diff > max_dist_y3):
            if debugging:
                print("Violated distance criterion: " +
                    "x1_diff == {:.2f}, x2_diff == {:.2f}, x3_diff == {:.2f}".format(x1_diff, x2_diff, x3_diff))
            return False

        # Check whether the line slopes are similar for two distinct y-values
        # x = Ay**2 + By + C
        # dx/dy = 2Ay + B
        
        y1left_dx  = 2 * left_fit[0]  * y1 + left_fit[1]
        y3left_dx  = 2 * left_fit[0]  * y3 + left_fit[1]
        y1right_dx = 2 * right_fit[0] * y1 + right_fit[1]
        y3right_dx = 2 * right_fit[0] * y3 + right_fit[1]

        # Compute the L1-norm
        norm1 = abs(y1left_dx - y1right_dx)
        norm2 = abs(y3left_dx - y3right_dx)
        
        # if debugging: print( norm1, norm2)

        # Define the L1 norm threshold
        thresh = 0.6 #0.58 
        if (norm1 >= thresh) | (norm2 >= thresh):
            if debugging:
                print("Violated tangent criterion: " +
                    "norm1 == {:.3f}, norm2 == {:.3f} (thresh == {}).".format(norm1, norm2, thresh))
                return False
        
        return True

    def polyfit_sliding_window(self, binary, lane_width_px=578, plot_result=False, debugging=False):
        '''
        Detect lane lines in a thresholded binary image using the sliding window technique
        :param binary (ndarray): Thresholded binary image
        :param lane_width_px (int): Average lane line width (in px) for the warped image 
        computed empirically
        :param plot_result (boolean): Boolean flag for visualisation
        :param diagnositics (boolean): Boolean flag for logging
        '''
        
        ret = True

        # Sanity check
        if binary.max() <= 0:
            return False, np.array([]), np.array([]), np.array([])
        
        # Step 1: Compute the histogram along all the columns in the lower half of the image. 
        # The two most prominent peaks in this histogram will be good indicators of the
        # x-position of the base of the lane lines
        histogram = None
        cutoffs = [int(binary.shape[0] / 2), 0]
        
        for cutoff in cutoffs:
            histogram = np.sum(binary[cutoff:, :], axis=0)
            
            if histogram.max() > 0:
                break

        if histogram.max() == 0:
            print('Unable to detect lane lines in this frame. Trying another frame!')
            return False, np.array([]), np.array([])
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        if plot_result:
            plotting([(binary, 'Histogram')])
            plt.plot(histogram, 'b', linewidth=4.0)

        out = np.dstack((binary, binary, binary)) * 255

        nb_windows = 12 # number of sliding windows
        margin = 100 # width of the windows +/- margin
        minpix = 50 # min number of pixels needed to recenter the window
        window_height = int(self.IMG_SHAPE[0] / nb_windows)
        min_lane_pts = 10  # min number of 'hot' pixels needed to fit a 2nd order polynomial as a 
                        # lane line
        
        # Identify the x-y positions of all nonzero pixels in the image
        # Note: the indices here are equivalent to the coordinate locations of the
        # pixel
        nonzero = binary.nonzero()
        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nb_windows):
            # Identify window boundaries in x and y (and left and right)
            win_y_low = self.IMG_SHAPE[0] - (1 + window) * window_height
            win_y_high = self.IMG_SHAPE[0] - window * window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin

            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw windows for visualisation
            cv2.rectangle(out, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),\
                        (0, 255, 0), 2)
            cv2.rectangle(out, (win_xright_low, win_y_low), (win_xright_high, win_y_high),\
                        (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)
                            & (nonzerox >= win_xleft_low) & (nonzerox <= win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)
                            & (nonzerox >= win_xright_low) & (nonzerox <= win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) >  minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))

            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract pixel positions for the left and right lane lines
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        left_fit, right_fit = None, None
        
        # Sanity check; Fit a 2nd order polynomial for each lane line pixels
        if len(leftx) >= min_lane_pts and len(rightx) >= min_lane_pts:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        
        # Validate detected lane lines
        valid = self.check_validity(left_fit, right_fit, debugging=debugging)
    
        if not valid:
            # If the detected lane lines are NOT valid:
            # 1. Compute the lane lines as an average of the previously detected lines
            # from the cache and flag this detection cycle as a failure by setting ret=False
            # 2. Else, if cache is empty, return 
            
            if len(self.cache) == 0:
                if debugging: print('WARNING: Unable to detect lane lines in this frame.')
                return False, np.array([]), np.array([])
            
            avg_params = np.mean(self.cache, axis=0)
            left_fit, right_fit = avg_params[0], avg_params[1]
            ret = False
            
        plot_xleft, plot_yleft, plot_xright, plot_yright = self.get_poly_points(left_fit, right_fit)

        # Color the detected pixels for each lane line
        out[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 255, 0]
        out[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 255, 255]

        left_poly_pts = np.array([np.transpose(np.vstack([plot_xleft, plot_yleft]))])
        right_poly_pts = np.array([np.transpose(np.vstack([plot_xright, plot_yright]))])

        # Plot the fitted polynomial
        cv2.polylines(out, np.int32([left_poly_pts]), isClosed=False, color=(200,255,155), thickness=4)
        cv2.polylines(out, np.int32([right_poly_pts]), isClosed=False, color=(200,255,155), thickness=4)

        if plot_result:
            plotting([(out, 'Sliding Window')], figsize=(20, 40))
            
        return ret, out, np.array([left_fit, right_fit])

















    def polyfit_adapt_search(self, img, prev_poly_param, plot_result=False, debugging=False):
        '''
        Function that: 
        1. Uses the sliding window technique to perform incremental localised adaptive threhsolding
        over the previosuly detected lane line trajectory to develop a threhsolded binary image. Then,
        2. Uses this generated binary image to detect and fit lane lines in a margin around the previous fit rather 
        than performing a blind search
        :param img (ndarray): Warped image
        :param prev_poly_param (ndarray): Polynomial coefficients of the previously detected lane lines
        :param plot_result (boolean): Boolean flag for visualisation
        :param debugging (boolean): Boolean flag for logging
        : return (ndarray, ndarray): 3 channel image with the newly detected lane lines, current polynomial coefficients
        '''
        
        # Cache of the previosuly detected lane line coefficients
        # Number of retries before the pipeline is RESET to detect lines via the smoothing window aproach
        
        # Sanity check
        assert(len(img.shape) == 3)
        
        # Setup
        nb_windows = 10 # Number of windows over which to perform the localised color thresholding  
        bin_margin = 80 # Width of the windows +/- margin for localised thresholding
        margin = 60 # Width around previous line positions +/- margin around which to search for the new lines
        window_height = int(img.shape[0] / nb_windows)
        smoothing_window = 5 # Number of frames over which to compute the Moving Average
        min_lane_pts = 10
        
        binary = np.zeros_like(img[:,:,0]) # Placeholder for the thresholded binary image
        img_plot = np.copy(img)
            
        left_fit, right_fit = prev_poly_param[0], prev_poly_param[1]
        plot_xleft, plot_yleft, plot_xright, plot_yright = self.get_poly_points(left_fit, right_fit)
        
        leftx_current = int(plot_xleft[-1])
        rightx_current = int(plot_xright[-1])
        
        # Iterate over the windows, perform localised color thresholding and generate the binary image
        for window in range(nb_windows):
            # Identify window boundaries in x and y (and left and right)
            win_y_low = self.IMG_SHAPE[0] - (window + 1) * window_height
            win_y_high = self.IMG_SHAPE[0] - window * window_height
            win_xleft_low = min(max(0, leftx_current - bin_margin), 1280)
            win_xleft_high = min(max(0, leftx_current + bin_margin), 1280)
            win_xright_low = min(max(0, rightx_current - bin_margin), 1280)
            win_xright_high = min(max(0, rightx_current + bin_margin), 1280)

            img_win_left = img[win_y_low:win_y_high, win_xleft_low:win_xleft_high,:]
            binary[win_y_low:win_y_high, win_xleft_low:win_xleft_high] = \
                get_binary_image(img_win_left, plot_result=False)

            img_win_right = img[win_y_low:win_y_high, win_xright_low:win_xright_high, :]
            binary[win_y_low:win_y_high, win_xright_low:win_xright_high] = \
                get_binary_image(img_win_right, plot_result=False)

            # Given that we only keep the points/values for a line that lie within the image
            # (see 'self.get_poly_points'), the overall length and consequently number of points (i.e. x-values
            # and y-values) can be < the length of the image. As a result, we check for the presence
            # of the current window's lower y-value i.e 'win_y_low' as a valid point within the previously detected line
            # If, a point associated with this y-value exists, we update the x-position of the next window with
            # the corresponding x-value.
            # Else, we keep the x-position of the subsequent windows the same and move up the image
            idxs = np.where(plot_yleft == win_y_low)[0]
            if len(idxs) != 0:
                leftx_current = int(plot_xleft[idxs[0]])
                
            idxs = np.where(plot_yright == win_y_low)[0]
            if len(idxs) != 0:
                rightx_current = int(plot_xright[idxs[0]])

            if plot_result:
                left_pts = np.array([np.transpose(np.vstack([plot_xleft, plot_yleft]))])
                right_pts = np.array([np.transpose(np.vstack([plot_xright, plot_yright]))])
                
                # Plot the previously detected lane lines
                cv2.polylines(img_plot, np.int32([left_pts]), isClosed=False, color=(255, 20, 147), thickness=4)
                cv2.polylines(img_plot, np.int32([right_pts]), isClosed=False, color=(255, 20, 147), thickness=4)    
                
                bin_win_left = binary[win_y_low:win_y_high, win_xleft_low:win_xleft_high]
                bin_win_left = np.dstack((bin_win_left, np.zeros_like(bin_win_left), np.zeros_like(bin_win_left))) * 255

                bin_win_right = binary[win_y_low:win_y_high, win_xright_low:win_xright_high]
                bin_win_right = np.dstack([np.zeros_like(bin_win_right), np.zeros_like(bin_win_right), bin_win_right]) * 255
                
                # Blend the localised image window with its corresponding thresholded binary version
                win_left = cv2.addWeighted(bin_win_left, 0.5, img_win_left, 0.7, 0)
                win_right = cv2.addWeighted(bin_win_right, 0.5, img_win_right, 0.7, 0)
                
                # Draw the binary search window
                cv2.rectangle(img_plot, (win_xleft_low,win_y_low), (win_xleft_high,win_y_high), (0,255,0), 5)
                cv2.rectangle(img_plot, (win_xright_low,win_y_low), (win_xright_high,win_y_high), (0,255,0), 5)
                
                f, _ = plt.subplots(1, 2, figsize=(13,5))

                plt.subplot(121)
                plt.axis('off')
                plt.imshow(binary, cmap='gray')

                plt.subplot(122)
                plt.axis('off')
                plt.imshow(img_plot)

                plt.subplots_adjust(top=0.98, bottom=0.0, left=0.0, right=1.0, hspace=0.1, wspace=0.05)
                plt.savefig('./gif_images/window1{:02}.png'.format(window))
                
                # The blended Binary window and Image window is added later for better visualisation
                img_plot[win_y_low:win_y_high, win_xleft_low:win_xleft_high] = win_left
                img_plot[win_y_low:win_y_high, win_xright_low:win_xright_high] = win_right
            
        # Identify the x-y coordinates of all the non-zero pixels from the binary image
        # generated above
        nonzero = binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Extract all the 
        left_lane_inds = \
            ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) &
            (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 

        right_lane_inds = \
            ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) &
            (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Sanity checks
        if len(leftx) > min_lane_pts:
            left_fit = np.polyfit(lefty, leftx, 2)
        else:
            if debugging: print('WARNING: Less than {} pts detected for the left lane. {}'.format(min_lane_pts, len(leftx)))

        if len(rightx) > min_lane_pts:
            right_fit = np.polyfit(righty, rightx, 2)
        else:
            if debugging: print('WARNING: Less than {} pts detected for the right lane. {}'.format(min_lane_pts, len(rightx)))
            
        valid = self.check_validity(left_fit, right_fit, debugging=debugging)

        # Perform smoothing via moving average
        if valid:
            if len(self.cache) < smoothing_window:
                self.cache = np.concatenate((self.cache, [np.array([left_fit, right_fit])]), axis=0)
            elif len(self.cache) >= smoothing_window:
                self.cache[:-1] = self.cache[1:]
                self.cache[-1] = np.array([left_fit, right_fit])
    
            avg_params = np.mean(self.cache, axis=0)
            left_fit, right_fit = avg_params[0], avg_params[1]
            plot_xleft, plot_yleft, plot_xright, plot_yright = self.get_poly_points(left_fit, right_fit)
            curr_poly_param = np.array([left_fit, right_fit])
        else:
            self.attempts += 1
            curr_poly_param = prev_poly_param
        
        out = np.dstack([binary, binary, binary]) * 255
        win_img = np.zeros_like(out)

        # Color the lane line pixels
        out[lefty, leftx] = [255, 0, 0]
        out[righty, rightx] = [255, 255, 255]

        left_window1 = np.array([np.transpose(np.vstack([plot_xleft - margin, plot_yleft]))])
        left_window2 = np.array([np.flipud(np.transpose(np.vstack([plot_xleft + margin, plot_yleft])))])
        left_pts = np.hstack([left_window1, left_window2])

        right_window1 = np.array([np.transpose(np.vstack([plot_xright - margin, plot_yright]))])
        right_window2 = np.array([np.flipud(np.transpose(np.vstack([plot_xright + margin, plot_yright])))])
        right_pts = np.hstack([right_window1, right_window2])

        # Draw the search boundary
        cv2.fillPoly(win_img, np.int_([left_pts]), (0, 255, 0))
        cv2.fillPoly(win_img, np.int_([right_pts]), (0, 255, 0))

        out = cv2.addWeighted(out, 1, win_img, 0.25, 0)

        left_poly_pts = np.array([np.transpose(np.vstack([plot_xleft, plot_yleft]))])
        right_poly_pts = np.array([np.transpose(np.vstack([plot_xright, plot_yright]))])

        # Draw the fit lane lines
        cv2.polylines(out, np.int32([left_poly_pts]), isClosed=False, color=(200,255,155), thickness=4)
        cv2.polylines(out, np.int32([right_poly_pts]), isClosed=False, color=(200,255,155), thickness=4)

        return out, curr_poly_param















    def compute_offset_from_center(self, poly_param, x_mppx):
        '''
        Computes the offset of the car from the center of the detected lane lines
        :param poly_param (ndarray): Set of 2nd order polynomial coefficients that represent the detected lane lines
        :param x_mppx (float32): metres/pixel in the x-direction
        :return (float32): Offset 
        '''
        plot_xleft, plot_yleft, plot_xright, plot_yright = self.get_poly_points(poly_param[0], poly_param[1])
        
        lane_center = (plot_xright[-1] + plot_xleft[-1]) / 2
        car_center = self.IMG_SHAPE[1] / 2
        
        offset = (lane_center - car_center) * x_mppx
        return offset

    def compute_curvature(self, poly_param, y_mppx, x_mppx):
        '''
        Computes the curvature of the lane lines (in metres)
        :param poly_param (ndarray): Set of 2nd order polynomial coefficients that represent the detected lane lines
        :param y_mppx (float32): metres/pixel in the y-direction
        :param x_mppx (float32): metres/pixel in the x-direction
        :return (float32): Curvature (in metres) 
        '''
        plot_xleft, plot_yleft, plot_xright, plot_yright = self.get_poly_points(poly_param[0], poly_param[1])
        
        y_eval = np.max(plot_yleft)

        left_fit_cr = np.polyfit(plot_yleft * y_mppx, plot_xleft * x_mppx, 2)
        right_fit_cr = np.polyfit(plot_yright * y_mppx, plot_xright * x_mppx, 2)
        
        left_curverad = ((1 + (2*left_fit_cr[0]* y_eval*y_mppx + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*y_mppx + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        return left_curverad, right_curverad

    def draw(self, img, warped, invM, poly_param, curve_rad, offset):
        '''
        Utility function to draw the lane boundaries and numerical estimation of lane curvature and vehicle position.
        :param img (ndarray): Original image
        :param warped (ndarray): Warped image
        :param invM (ndarray): Inverse Perpsective Transformation matrix
        :param poly_param (ndarray): Set of 2nd order polynomial coefficients that represent the detected lane lines
        :param curve_rad (float32): Lane line curvature
        :param offset (float32): Car offset
        :return (ndarray): Image with visual display
        '''
        
        warp_zero = np.zeros_like(warped[:,:,0]).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        left_fit = poly_param[0]
        right_fit = poly_param[1]
        plot_xleft, plot_yleft, plot_xright, plot_yright = self.get_poly_points(left_fit, right_fit)
        
        pts_left = np.array([np.transpose(np.vstack([plot_xleft, plot_yleft]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([plot_xright, plot_yright])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Color the road
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 220, 110))
                        
        cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False,
                    color=(255, 255, 255), thickness=10)
        cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False,
                    color=(255, 255, 255), thickness= 10)
        
        # Unwarp and merge with the original image
        unwarped = cv2.warpPerspective(color_warp, invM, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        out = cv2.addWeighted(img, 1, unwarped, 0.4, 0)
        

        text = 'Radius of curvature = {:04.2f} m'.format(curve_rad)

        cv2.putText(out, text, (50, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        direction = ''
        if offset > 0:
            direction = 'left'
        elif offset < 0:
            direction = 'right'
        
        text = 'Vehicle is {:0.1f} cm {} of center'.format(abs(offset) * 100, direction)
        cv2.putText(out, text, (50, 110), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        return out

    def triple_split_view(self, images):
        '''
        Utility function to create triple split view for display in video
        :param images ([ndarray]): List of images
        :returm (ndarray): Single RGB image 
        '''

        scale_factor = 2
        
        # Sizes/shapes are in (x,y) format for convenience use with cv2
        img_shape = self.IMG_SHAPE[::-1]
        scaled_size = (round(img_shape[0] / scale_factor), round(img_shape[1] / scale_factor))
        x_max, y_max = img_shape[0], img_shape[1] + scaled_size[1] # x, y + y'

        # Top-left corner positions for each of the three windows
        positions = [(0,0), (0, img_shape[1]), (round(0.5 * img_shape[0]), img_shape[1])]
        sizes = [img_shape, scaled_size, scaled_size] 
        
        out = np.zeros((y_max, x_max, 3), dtype=np.uint8)

        for idx, img in enumerate(images):
            # Resize the image
            if img.shape[0] != sizes[idx][1] | img.shape[1] != sizes[idx][0]:
                img = cv2.resize(img, dsize=sizes[idx])

            # Place the resized image onto the final output image
            x, y = positions[idx]
            w, h = sizes[idx]
            out[y:min(y + h, y_max), x:min(x + w, x_max), :] = img[:min(h, y_max - y), :min(w, x_max - x)]

        return out

    def pipeline(self, img, Verbose=True, plot_result=False, debugging=False):

        # global poly_param # Important for successive calls to the pipeline

        max_attempts = 5
        
        result = np.copy(img)
        warped, (M, invM) = preprocess_image(img)
        title = ''
    
        if self.reset == True:
            title = 'Sliding window'
            if debugging: print(title)

            binary = get_binary_image(warped)
            ret, img_poly, self.poly_param = self.polyfit_sliding_window(binary, debugging=debugging)
            if ret:
                if debugging: print('Success!')
                self.reset = False
                self.cache = np.array([self.poly_param])
            else:
                if len(img_poly) == 0:
                    print('Sliding window failed!')
                    return img

        else:
            title = 'Adaptive Search'
            if debugging: print(title)

            img_poly, self.poly_param = self.polyfit_adapt_search(warped, self.poly_param, debugging=debugging)
            if self.attempts == max_attempts:
                if debugging: print('Resetting...')
                self.reset = True
                self.attempts = 0

        left_curverad, right_curverad = self.compute_curvature(self.poly_param, self.y_mppx, self.x_mppx)
        offset = self.compute_offset_from_center(self.poly_param, self.x_mppx)
        result = self.draw(img, warped, invM, self.poly_param, (left_curverad + right_curverad) / 2, offset)

        blended_warped_poly = cv2.addWeighted(img_poly, 0.6, warped, 1, 0)
        ret2 = np.hstack([img_poly, blended_warped_poly])
        ret3 = np.hstack([result, warped])
        #ret3 = triple_split_view([result, img_poly, blended_warped_poly])
        ret3 = np.vstack([ret3, ret2])
        if plot_result:
            plt.figure(figsize=(20, 12))
            plt.title(title)
            plt.imshow(ret3)
        if Verbose:
            return ret3
        else:
            return result
