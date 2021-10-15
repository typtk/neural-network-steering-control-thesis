import numpy as np
import time
import tensorflow as tf
import cv2

class tflite_model:
    def __init__ (self, model_path):
        self.model_path = model_path
        self.interpreter = 0
        self.input_index = 0
        self.output_index = 0
        self.input_height = 0
        self.input_width = 0

        self.start_time = 0
        self.stop_time = 0

    def imread(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return image

    def load(self):
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        self.input_index = self.interpreter.get_input_details()
        self.output_index = self.interpreter.get_output_details()

        self.input_height = self.input_index[0]['shape'][1]
        self.input_width = self.input_index[0]['shape'][2]
        # self.floating_model = (self.input_index[0]['dtype'] == np.float32)

    def prep_image(self, image):
        height, _, _ = image.shape
        image = image[int(height/2):,:,:]
        ######## using in noon ########
        # image = self.adjust_brightness(image)
        ###############################
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        image = cv2.GaussianBlur(image, (5,5), 0)
        image = cv2.resize(image, (self.input_width, self.input_height))
        norm_image = [(np.float32(image)) / 255]
        return norm_image

    def adjust_brightness(image, ratio=0.85):
        image = int(image*ratio)
        return image

    def predict(self, image):
        self.start_time = time.time()
        norm_image = self.prep_image(image)
        self.interpreter.set_tensor(self.input_index[0]['index'], norm_image)
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output_index[0]['index'])[0][0]
        self.stop_time = time.time()
        return int(result)

    def print_time(self):
        time = (self.stop_time - self.start_time)*1000
        print("time using :", round(time), "ms")

if __name__ == "__main__":
    model_path = "/home/pi/model/tf_lite/20210409_CNN/model.tflite"
    model = tflite_model(model_path)
    model.load()
    image = model.imread("/home/pi/mce06/capture/images/20210416_16370975.png")
    angle = model.predict(image)
    print(angle)
    model.print_time()