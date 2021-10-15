import matplotlib.pyplot as plt
import math
import numpy as np
import cv2

class error_meas:
    def __init__ (self):
        self.w, self.h = 0, 0
        self.lane_width_base_real = 50 + 1.9 + 1.9
        self.lane_width_base = 696 - (-18)
        self.lane_width_mid = 573 - 73
        self.perspective_ratio = self.lane_width_mid / self.lane_width_base

        self.hsv_image = 0
        self.mask_image = 0
        self.opening_image = 0
        self.cropped_image = 0
        self.window_image = 0

        self.histogram = 0
        self.leftx_base = 0
        self.rightx_base = 0
        self.left_lane_inds = 0
        self.right_lane_inds = 0
        self.left_fit = 0
        self.right_fit = 0
        self.left_fitx = 0
        self.right_fitx = 0
        self.ploty = 0
        self.visualization_data = 0

        self.lane_avg = 0
        self.lane_mid_pos = 0
        self.start_point = [0, 0]
        self.move_point = 0

        self.desired_angle = 0
        self.error_pos = 0

    def imread(self, path):
        image = cv2.imread(path)
        return image

    def region_of_interest(self, img, ratio=0.5):
        height, width = img.shape
        mask = np.zeros_like(img)

        # only focus bottom half of the screen
        polygon = np.array([[
            (0, height * ratio),
            (width, height * ratio),
            (width, height),
            (0, height),
        ]], np.int32)

        cv2.fillPoly(mask, polygon, 255)
        cropped_image = cv2.bitwise_and(img, mask)
        return cropped_image

    def lane_base(self, img, ratio_horizon_line=0.3, nsection=6):
        pos_horizon_line = 1 - ratio_horizon_line
        section = [int(img.shape[1]*(k/nsection)) for k in range(1, nsection+1)]
        histogram = np.sum(img[int(img.shape[0]*pos_horizon_line):,:], axis=0)
        midpoint = np.int(histogram.shape[0]//2)
        quarter_point = np.int(midpoint//2)
        leftx_base, rightx_base = section[1], section[4]
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint+quarter_point:]) + midpoint + quarter_point
        return histogram, leftx_base, rightx_base
    
    def sliding_window_1st_deg(self, img, histogram, leftx_base, rightx_base, deg=1, nwindow=10):
        window_height = int(img.shape[0]/nwindow)
        # Get all the x and y positions of nonzero pixels
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        # Set the width of the window +/- margin
        margin = 80

        # Set minimum number of pixels found for recenter window
        minpix = 5

        self.left_lane_inds = []
        self.right_lane_inds = []

        rectangle_data = []

        for n in range(nwindow):
            # Identify window boundaries
            window_top = img.shape[0] - (n+1)*window_height
            window_bot = img.shape[0] - (n)*window_height
            window_xleft_low = leftx_current - margin
            window_xleft_high = leftx_current + margin
            window_xright_low = rightx_current - margin
            window_xright_high = rightx_current + margin

            rectangle_data.append((window_top, window_bot, window_xleft_low, window_xleft_high, window_xright_low, window_xright_high))

            # Identify the nonzero pixels in window
            one_left_inds = ((nonzeroy >= window_top) & (nonzeroy < window_bot) & (nonzerox >= window_xleft_low) & (nonzerox < window_xleft_high)).nonzero()[0]
            one_right_inds = ((nonzeroy >= window_top) & (nonzeroy < window_bot) & (nonzerox >= window_xright_low) & (nonzerox < window_xright_high)).nonzero()[0]

            self.left_lane_inds.append(one_left_inds)
            self.right_lane_inds.append(one_right_inds)
            
            # Determine whether recenter or not
            if len(one_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[one_left_inds]))
            if len(one_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[one_right_inds]))

        self.left_lane_inds = np.concatenate(self.left_lane_inds)
        self.right_lane_inds = np.concatenate(self.right_lane_inds)

        leftx = nonzerox[self.left_lane_inds]
        lefty = nonzeroy[self.left_lane_inds] 
        rightx = nonzerox[self.right_lane_inds]
        righty = nonzeroy[self.right_lane_inds] 

        left_fit, right_fit = (None, None)
        # Fit a second order polynomial to each
        if len(leftx) != 0:
            left_fit = np.polyfit(lefty, leftx, deg)
        if len(rightx) != 0:
            right_fit = np.polyfit(righty, rightx, deg)
        visualization_data = (rectangle_data, histogram)
        return left_fit, right_fit, self.left_lane_inds, self.right_lane_inds, visualization_data

    def sliding_1st_deg_inspection(self, img, left_fit, right_fit):
        out_img = np.uint8(np.dstack((img, img, img))*255)
        rectangles = self.visualization_data[0]
        for rect in rectangles:
            cv2.rectangle(out_img,(rect[2],rect[0]),(rect[3],rect[1]),(0,255,0), 2) 
            cv2.rectangle(out_img,(rect[4],rect[0]),(rect[5],rect[1]),(0,255,0), 2) 
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        left_fitx, right_fitx = 0, 0
        try:
            left_fit_x_int = left_fit[0]*self.h + left_fit[1]
            left_fitx = left_fit[0]*ploty + left_fit[1]
            out_img[nonzeroy[self.left_lane_inds], nonzerox[self.left_lane_inds]] = [255, 0, 0]
        except:
            pass
        try:
            right_fit_x_int = right_fit[0]*self.h + right_fit[1]
            right_fitx = right_fit[0]*ploty + right_fit[1]
            out_img[nonzeroy[self.right_lane_inds], nonzerox[self.right_lane_inds]] = [100, 200, 255]
        except:
            pass
        return out_img, left_fitx, right_fitx, ploty

    def inspect_no_angle(self):
        plt.imshow(self.window_image)
        try:
            plt.plot(self.left_fitx, self.ploty, color='yellow')
        except:
            pass
        try:
            plt.plot(self.right_fitx, self.ploty, color='yellow')
        except:
            pass
        plt.xlim(0, self.w)
        plt.ylim(self.h, 0)

    def inspect(self):
        point_inplot_1 = [self.start_point[0], self.move_point[0]]
        point_inplot_2 = [self.start_point[1], self.move_point[1]]
        plt.imshow(self.window_image)
        try:
            plt.plot(self.left_fitx, self.ploty, color='yellow')
        except:
            pass
        try:
            plt.plot(self.right_fitx, self.ploty, color='yellow')
        except:
            pass
        try:
            plt.plot(self.point_avg, self.ploty, color='orange')
        except:
            pass
        plt.plot(point_inplot_1, point_inplot_2, linewidth=5, color='m')
        plt.xlim(0, self.w)
        plt.ylim(self.h, 0)

    def implement_old(self, image):
        IMAGE_SIZE = (640, 480)
        self.w, self.h = IMAGE_SIZE
        car_pos = self.w//2
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 80])
        kernel = np.ones((3, 3), np.uint8)

        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        self.hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        self.mask_image = cv2.inRange(self.hsv_image, lower, upper)
        self.opening_image = cv2.morphologyEx(self.mask_image, cv2.MORPH_OPEN, kernel)
        self.cropped_image = self.region_of_interest(self.opening_image)
        self.histogram, self.leftx_base, self.rightx_base  = self.lane_base(self.cropped_image)
        self.left_fit, self.right_fit, self.left_lane_inds, self.right_lane_inds, self.visualization_data = self.sliding_window_1st_deg(self.cropped_image, self.histogram, self.leftx_base, self.rightx_base)
        
        if self.left_fit is None:
            self.left_fit = np.array([0, 0])
        if self.right_fit is None:
            self.right_fit = np.array([0, self.w])
        self.window_image, self.left_fitx, self.right_fitx, self.ploty = self.sliding_1st_deg_inspection(self.cropped_image, self.left_fit, self.right_fit)
        
        self.lane_avg = (self.left_fitx + self.right_fitx) // 2
        if (abs(self.left_fit[1] - self.right_fit[1]) < 10):
            self.lane_mid_pos = self.right_fit[0]*(self.h-1) + self.right_fit[1] - self.lane_width_base//2
            self.error_pos = car_pos - self.lane_mid_pos
            self.move_point = [(self.right_fit[0]*(self.h//2) + self.right_fit[1] - self.lane_width_base//2*self.perspective_ratio), self.h//2]
        else:
            self.lane_mid_pos = self.lane_avg[self.h-1]
            self.error_pos = car_pos - self.lane_mid_pos
            self.move_point = [self.lane_avg[self.h//2], self.h//2]

        self.error_pos = round(self.error_pos*self.lane_width_base_real/self.lane_width_base)
        self.desired_angle = round(math.degrees(math.atan2(self.move_point[1], self.move_point[0]-self.w//2)))

        return self.desired_angle, self.error_pos

    def implement(self, image):
        IMAGE_SIZE = (640, 480)
        self.w, self.h = IMAGE_SIZE
        self.start_point = [self.w//2, self.h]
        car_pos = self.w//2
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 80])
        kernel = np.ones((3, 3), np.uint8)

        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        self.hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        self.mask_image = cv2.inRange(self.hsv_image, lower, upper)
        self.opening_image = cv2.morphologyEx(self.mask_image, cv2.MORPH_OPEN, kernel)
        self.cropped_image = self.region_of_interest(self.opening_image)
        self.histogram, self.leftx_base, self.rightx_base  = self.lane_base(self.cropped_image)
        self.left_fit, self.right_fit, self.left_lane_inds, self.right_lane_inds, self.visualization_data = self.sliding_window_1st_deg(self.cropped_image, self.histogram, self.leftx_base, self.rightx_base)

        try:
            self.window_image, self.left_fitx, self.right_fitx, self.ploty = self.sliding_1st_deg_inspection(self.cropped_image, self.left_fit, self.right_fit)
        except:
            if self.left_fit is None:
                self.left_fit = np.array([0, 0])
                self.window_image, self.left_fitx, self.right_fitx, self.ploty = self.sliding_1st_deg_inspection(self.cropped_image, self.left_fit, self.right_fit)
                self.left_fit = None
            if self.right_fit is None:
                self.right_fit = np.array([0, self.w])
                self.window_image, self.left_fitx, self.right_fitx, self.ploty = self.sliding_1st_deg_inspection(self.cropped_image, self.left_fit, self.right_fit)
                self.right_fit = None

        if self.left_fit is not None and self.right_fit is not None:
            self.lane_avg = (self.left_fitx + self.right_fitx) // 2
            if (abs(self.left_fit[1] - self.right_fit[1]) < 10):
                self.lane_mid_pos = self.right_fit[0]*(self.h-1) + self.right_fit[1] - self.lane_width_base//2
                self.error_pos = car_pos - self.lane_mid_pos
                self.move_point = [(self.right_fit[0]*(self.h//2) + self.right_fit[1] - self.lane_width_base//2*self.perspective_ratio), self.h//2]
            else:
                self.lane_mid_pos = self.lane_avg[self.h-1]
                self.error_pos = car_pos - self.lane_mid_pos
                self.move_point = [self.lane_avg[self.h//2], self.h//2]
        elif self.left_fit is None and self.right_fit is not None:
            self.lane_mid_pos = self.right_fit[0]*(self.h-1) + self.right_fit[1] - self.lane_width_base//2
            self.error_pos = car_pos - self.lane_mid_pos
            self.move_point = [(self.right_fit[0]*(self.h//2) + self.right_fit[1] - self.lane_width_base//2*self.perspective_ratio), self.h//2]
        elif self.left_fit is not None and self.right_fit is None:
            self.lane_mid_pos = self.left_fit[0]*(self.h-1) + self.left_fit[1] + self.lane_width_base//2
            self.error_pos = car_pos - self.lane_mid_pos
            self.move_point = [(self.left_fit[0]*(self.h//2) + self.left_fit[1] + self.lane_width_base//2*self.perspective_ratio), self.h//2]
        
        self.error_pos = round(self.error_pos*self.lane_width_base_real/self.lane_width_base, 2)
        self.desired_angle = round(math.degrees(math.atan2(self.move_point[1], self.move_point[0]-self.w//2)))

        return self.desired_angle, self.error_pos

if __name__ == "__main__":
    meas = error_meas()
    image_path = r"test_images\old_images\20210330_13194097_86.png"
    image = meas.imread(image_path)
    angle, error_pos = meas.implement(image)
    print(angle)
    print(error_pos)