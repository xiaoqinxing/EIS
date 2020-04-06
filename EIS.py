import pickle
import cv2
import matplotlib.pyplot as plt
import gyrodata
import videodata
import numpy as np
import os,math

filepath = "C:/Users/qinxing/Desktop/tmp/Recorder/"
csvfilename = "VID_20200406_190021gyro.csv"
videoname = "VID_20200406_190021"
videotype = ".mp4"
csv = filepath + csvfilename
video = filepath + videoname + videotype
video_timestamps_fullpath = "./video_timestamps.data"
video_data_fullpath = "./video_data.data"
shutter_duration = 40 # 40ms

if __name__ == "__main__":
    print("EIS start")
    gyroscopedata = gyrodata.GyroscopeDataFile(csv)
    gyroscopedata.read_data()
    videodata = videodata.VideoDataFile(video)
    videodata = videodata.read_data()
    print("EIS end")