from picamera import PiCamera
from picamera.array import PiRGBArray
import serial
import time
import threading
from datetime import datetime

import numpy as np
import cv2

from car_status import CarStatus
from tf_lite import tflite_model

def frameCapture():
    camera.capture(rawCapture, format="bgr", use_video_port=True)
    image = rawCapture.array
    rawCapture.truncate(0)
    return image

def saveImage(image, angle):
    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-4]
    file_name = save_dir + "/" + time_stamp + "_" + angle + ".png"
    file_name.replace(" ", "")
    cv2.imwrite(file_name, image)

def thread_save(image, speed, process_time):
    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-4]
    file_name = save_dir + "/" + time_stamp + ".png"
    file_name.replace(" ", "")
    cv2.imwrite(file_name, image)
    write_data = time_stamp + ".png" + "," + str(angle) + "," + str(speed) + "," + str(process_time)
    fd = open(csv_file, 'a')
    fd.write("\n" + write_data)
    fd.close()

def thread_serRead():
    global cycle_time
    global speed
    while True:
        line = ser_speed.readline().decode('utf-8').rstrip()
        # print("speed is :", line)
        if len(line) > 0:
            speed = float(line)
            cycle_time = calCycleTime(speed)

def calCycleTime(speed):
    # deadzone front of car in meters
    deadzone = 0.1
    if speed < 0.2:
        cycle_time = 0.1
    else:
        cycle_time = deadzone / speed
    return cycle_time

def steeringControl(angle, speed, car_status):
    send = str(angle) + "," + str(speed) + "," + str(car_status) + "\n"
    # send = str(angle) + "\n"
    ser_steer.write((send).encode('utf-8'))
    # print(send)

# camera configuration
resolution = [640, 480]
camera = PiCamera()
camera.resolution = (resolution[0], resolution[1])
camera.framerate = 62
camera.sensor_mode = 7
rawCapture = PiRGBArray(camera)
time.sleep(0.1)

# serial setting
try:
    ser_steer = serial.Serial("/dev/ttyACM0", 115200)
    ser_speed = serial.Serial("/dev/ttyACM1", 9600, timeout=0.5)
except:
    print("Serial error!")
# ls /dev/tty*

# car status setup
car_status = CarStatus()

# model setup
model_path = "/home/pi/model/tf_lite/20210409_CNN/model.tflite"
model = tflite_model(model_path)
model.load()

# directory for save image
save_dir = "/home/pi/histories/new_route"
csv_file = save_dir + "/data.csv"

thread_speed = threading.Thread(target=thread_serRead)
thread_speed.start()

cycle_time = 0.1
speed = 0
print("READY!")
while True:
    try:
        start_time = time.time()
        image = frameCapture()
        status = car_status.inspect(image)
        # car_status.print_status()
        angle = model.predict(image)
        # print("angle : ", angle)
        print("speed : ", speed)
        # model.print_time()
        process_time = time.time() - start_time
        thread_savefile = threading.Thread(target=thread_save, args=(image, speed, int(process_time*1000)))
        thread_savefile.start()
        if process_time < cycle_time:
            time.sleep(cycle_time - process_time)
        steeringControl(angle, speed, 1)
    except:
        rawCapture.truncate(0)
        break