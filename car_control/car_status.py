import time
import numpy as np
import cv2

class CarStatus:
    def __init__(self):
        self.start_time = 0
        self.stop_time = 0
        self.status = []

    def imread(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return image

    def pre_stop_check(self, image, y_ratio=0.6):
        h, _, _ = image.shape
        y_lower = int((1 - y_ratio) * h)
        image = image[y_lower:,:,:]
        return image

    def stop_non_black(self, image, thres_percent=0.5):
        height, width, _ = image.shape
        lower = np.array([0, 120, 60])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(image, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        image_denoised = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, hierarchy = cv2.findContours(image_denoised, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_area = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area>max_area:
                cnt = contours[i]
                max_area = area
                max_area_index = i
        try:
            x,y,w,h = cv2.boundingRect(contours[max_area_index])
            percent = (w*h)*100/(height*width)
            if percent >= thres_percent:
                return 0
            else:
                return 1
        except:
            return 1

    def inspect(self, image):
        self.status = []
        self.start_time = time.time()
        image = self.pre_stop_check(image)
        self.status.append(self.stop_non_black(image))
        arr = np.array(self.status)
        is_all_one = np.all((arr==1))
        self.stop_time = time.time()
        if is_all_one:
            return 1
        else:
            return 0

    def print_status(self):
        print("car status :", self.status)

    def print_time(self):
        time = (self.stop_time - self.start_time)*1000
        print("time using :", round(time), "ms")

if __name__ == "__main__":
    car_status = CarStatus()
    image = car_status.imread("/home/pi/mce06/capture/images/20210403_15394712.png")
    car_status.inspect(image)
    car_status.print_status()
    car_status.print_time()