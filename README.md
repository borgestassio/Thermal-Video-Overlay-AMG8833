# Thermal-Video-Overlay-AMG8833

<img src="https://raw.githubusercontent.com/borgestassio/Thermal-Video-Overlay-AMG8833/master/Results/Thermal_Mask.png" title="ThermalOverlay" alt="ThermalOverlay"></a>


This repo has a code to use the Adafruit AMG8833 and a USB camera and turn them into a thermal camera with image overlay. With the temperature grid plus the video feed, we can safely measure the temperature of someone in front of the camera+sensor. Also, I did include mask usage identification using tensorflow, therefore, you can do all the same time:

This code was written to run on a Raspberry Pi, but if you plan to use the Face Mask detection functionality, I advise you to use on a Raspberry Pi 4 with 4Gb of RAM, otherwise the video feed will update **painfully** slow

* Video feed

* Thermal Image overlay

* Mask usage identification
