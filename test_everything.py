# Code and comments for:
# test_everything.py
# by tassioborges

# import all necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import math
import time
import busio
import board
from scipy.interpolate import griddata
import adafruit_amg88xx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm
from datetime import datetime

#datetime is not used yet, but will be in the future to create to export video/data

#This portion of the file was created by Adrian Rosebrock (pyimagesearch.com)
#if you want to know more about his work, please visit his website, it has some really interesting stuff

#-----------------------------------
#by: pyimagesearch.com:

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

        #-------- 
        #Note: this TFLITE portion was not made by pyimagesearch, so don't blame them if it doesnt work :D

        #FOLLOWING LINES NEED TO BE UNCOMMENTED IF USING TFLITE MODEL:
        #input_data = np.array(image_a, dtype=np.float32)
        #interpreter.set_tensor(input_details[0]['index'], input_data)
        #interpreter.invoke()
        #preds = interpreter.get_tensor(output_details[0]['index'])
        #-------- 
    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
    default="face_detector",
    help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
    default="model1.model",
    help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")


maskNet = load_model(args["model"])

#-----------------------------------

#We can use a tflite model if we want to, the difference is not that great, but if you prefer using tflite, uncomment the following lines


#interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
#interpreter.allocate_tensors()
#input_details = interpreter.get_input_details()
#output_details = interpreter.get_output_details()
#input_shape = input_details[0]['shape']
#maskNet = interpreter

#If it doesnt work, don't blame pyimagesearch :D
#-----------------------------------

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)


# Variable to define the transparency of the overlay: 0 - 1
alpha = 0.25

#this part of the code is based on the AMG8833, but quite some changes to allow 
#sensor conf
i2c_bus = busio.I2C(board.SCL, board.SDA)

sensor = adafruit_amg88xx.AMG88XX(i2c_bus)
max_temp = 34
min_temp = 27
points = [(math.floor(ix / 8), (ix % 8)) for ix in range(0, 64)]
grid_x, grid_y = np.mgrid[0:7:32j, 0:7:32j]
#the number 128 can be modified to suit your needs, take a look at matplotlib colormap to have a better understanding
viridis = cm.get_cmap('viridis',128)
#####

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 427 pixels
    frame = vs.read()

    frame = imutils.resize(frame, width=427)
    #the frame, in my case, must have 427 width to have a 320 height, so I can easily overlay the thermal image

    pixels = []
    for row in sensor.pixels:
        pixels = pixels + row

    bicubic = griddata(points, pixels, (grid_x, grid_y), method='cubic')
    datagrid = np.resize(pixels,(8,8))
    datagrid = np.rot90(datagrid,3)
    norm_grid_bicubic = (bicubic-min_temp)/(max_temp-min_temp)
    scaled_bicubic = norm_grid_bicubic*255
    pilimg = Image.fromarray(viridis(scaled_bicubic.astype('uint8'),bytes=True))
    pilimg = pilimg.convert('RGB')
    colorimg = cv2.cvtColor(np.array(pilimg),cv2.COLOR_RGB2BGR)

    #I did scale to 1000% in order to get a 320x320, you can use your own scale here
    scale_percent = 1000 
    widththermal = int(colorimg.shape[1] * scale_percent / 100)
    heightthermal = int(colorimg.shape[0] * scale_percent / 100)
    dim = (widththermal, heightthermal)
    
    #the thermal img is called foreground from now on:
    foreground = cv2.resize(colorimg, dim, interpolation = cv2.INTER_AREA)
    
    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    #place the thermal image on top of the webcam img (location is static here, but I'll make it dynamic eventually)
    added_image = cv2.addWeighted(frame[0:320,53:373,:],alpha,foreground[0:320,0:320,:],1-alpha,0)
    frame[0:320,53:373,:]= added_image
    
    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("over",frame)
    
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, breathermalimgk from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

