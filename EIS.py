import pickle
import cv2
import matplotlib.pyplot as plt
import gyrodata
import videodata
import numpy as np
import os, math
import calibration

filepath = "C:/Users/qinxing/Desktop/tmp/Recorder/"
csvfilename = "VID_20200406_190021gyro.csv"
videoname = "VID_20200406_190021"
videotype = ".mp4"
csv = filepath + csvfilename
video = filepath + videoname + videotype
video_timestamps_fullpath = "./video_timestamps.data"
video_data_fullpath = "./video_data.data"
shutter_duration = 33  # 40ms

# cameraParams from matlab
radialDistortion = [0.0894, -0.2675]
tangentialDistortion = [0, 0]
intrinsicMatix = [
        [2963.92162113863, 0, 0],
        [0, 2949.13665742050, 0],
        [1507.83245419482, 2036.84950656832, 1]
    ]
FocalLength = [2.963921621138634e+03, 2.949136657420503e+03]
PrincipalPoint = [1.507832454194816e+03, 2.036849506568323e+03]


if __name__ == "__main__":

    print("EIS start")
    # gyroscopedata = gyrodata.GyroscopeDataFile(csv)
    # gyroscopedata.read_data()
    # videodata = videodata.VideoDataFile(video)
    # videodata = videodata.read_data()
    # cal = calibration.CalibrateGyroStabilize(video, csv)
    # cal.calibrate()
    cameraParams = calibration.LenParametersFromMatlab(radialDistortion,
                                                        tangentialDistortion,
                                                        intrinsicMatix,
                                                        FocalLength,
                                                        PrincipalPoint)
    print(cameraParams.getCameraMatrix())
    print(cameraParams.getdistCoeffs())
    print("EIS end")