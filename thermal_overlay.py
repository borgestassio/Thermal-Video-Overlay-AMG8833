# Code and comments for:
# thermal_overlay.py
# by tassioborges

#created: 17-Aug-2020

#feel free to use this code however you want :D


# import all necessary packages
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


ap = argparse.ArgumentParser()
ap.add_argument("-A", "--alpha", type=float,
    default=0.25,
    help="alpha value for overlay transparency, default is 0.25")
args = vars(ap.parse_args())

#initialize video capture
cap = cv2.VideoCapture(0)



#initialize I2C bus and sensor
i2c_bus = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_amg88xx.AMG88XX(i2c_bus)

#define max and min temp for the colormap
max_temp = 34
min_temp = 27

#creat the grid 64x64
points = [(math.floor(ix / 8), (ix % 8)) for ix in range(0, 64)]
grid_x, grid_y = np.mgrid[0:7:32j, 0:7:32j]

#define the colormap
viridis = cm.get_cmap('viridis',128)
#take a look at the matplotlib doc if you wish to have a better understanding of this

while True:

	# Capture frame-by-frame
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=427)
    #the frame, in this case, must be 427px width and 320px height, so I can easily overlay the thermal image

    #initialize the pixels and add to the variable from the sensor
    pixels = []
    for row in sensor.pixels:
        pixels = pixels + row

    #we manipulate the data a bit here. we need to create the grid here, where it'll become a 64x64 instead of a 8x8
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

    added_image = cv2.addWeighted(frame[0:320,53:373,:],args["alpha"],foreground[0:320,0:320,:],1-args["alpha"],0)
    frame[0:320,53:373,:]= added_image

    cv2.imshow("over",frame)
    
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, breathermalimgk from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()

