import cv2
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

class VideoDataFile(object):
    # the file path to read
    # a dictionary of angular velocities
    # This dictionary will store mappings between the timestamp
    # and the angular velocity at that instant.
    def __init__(self, mp4):
        self.mp4 = mp4
        self.frameInfo = []
        self.numFrames = 0
        self.duration = 0
        self.frameWidth = 0
        self.frameHeight = 0
        self.videofiledata = "./.videodata.data"

    # search keypoints with timestamp in every frames
    def searchkeypoints(self, skip_keypoints=False):
        vidcap = cv2.VideoCapture(self.mp4)
        success, frame = vidcap.read()
        prev_frame = [[[0]]]
        previous_timestamp = 0
        frameCount = 0
        keypoints = list()

        self.frameWidth = frame.shape[1]
        self.frameHeight = frame.shape[0]
        # if video is ok ,start this loop
        while success:
            current_timestamp = vidcap.get(0) * 1000 * 1000
            print("Processing frame#%d (%f ns)" % (frameCount, current_timestamp))

            # first frame break this loop
            if prev_frame[0][0][0] == 0:
                self.frameInfo.append(
                    {"keypoints": None, "timestamp": current_timestamp}
                )
                prev_frame = frame
                previous_timestamp = current_timestamp
                continue

            # if skip_keypoints == true
            # it'll just store the timestamps of each frame.
            # use this parameter to read a video after you've already
            # calibrated your device and already have
            # the values of the various unknowns.
            if skip_keypoints:
                self.frameInfo.append(
                    {"keypoints": None, "timestamp": current_timestamp}
                )
                continue

            old_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            #            plt.imshow(old_gray, cmap='gray')
            #            plt.figure()
            #            plt.show()
            new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #            plt.figure()
            #            plt.imshow(new_gray, cmap='gray')
            #            plt.show()

            # extract some good features to track
            old_corners = cv2.goodFeaturesToTrack(old_gray, 1000, 0.3, 30)

            if old_corners is not None and len(old_corners) > 0:
                for x, y in np.float32(old_corners).reshape(-1, 2):
                    keypoints.append((x, y))

            if keypoints is not None and len(keypoints) > 0:
                for x, y in keypoints:
                    cv2.circle(prev_frame, (int(x + 200), y), 3, (255, 255, 0))

            # plt.imshow(prev_frame, cmap='gray')
            # plt.show()

            if old_corners.any() == None:
                self.frameInfo.append(
                    {"keypoints": None, "timestamp": current_timestamp}
                )
                frameCount += 1
                previous_timestamp = current_timestamp
                prev_frame = frame
                success, frame = vidcap.read()
                continue

            # If we did find keypoints to track, we use optical flow
            # to identify where they are in the new frame:
            # there may be the big defect!!
            new_corners, status, err = cv2.calcOpticalFlowPyrLK(
                old_gray,
                new_gray,
                old_corners,
                None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )

            if new_corners is not None and len(new_corners) > 0:
                for x, y in np.float32(new_corners).reshape(-1, 2):
                    keypoints.append((x, y))

            if keypoints is not None and len(keypoints) > 0:
                for x, y in keypoints:
                    cv2.circle(frame, (int(x + 200), y), 3, (255, 255, 0))

            #            plt.imshow(frame, cmap='gray')
            #            plt.show()

            if len(old_corners) > 4:
                homography, mask = cv2.findHomography(
                    old_corners, new_corners, cv2.RANSAC, 5.0
                )
                # convert to one dimension
                mask = mask.ravel()
                new_corners_homography = np.asarray(
                    [new_corners[i] for i in range(len(mask)) if mask[i] == 1]
                )
                old_corners_homography = np.asarray(
                    [old_corners[i] for i in range(len(mask)) if mask[i] == 1]
                )
            else:
                new_corners_homography = new_corners
                old_corners_homography = old_corners

            self.frameInfo.append(
                {
                    "keypoints": (old_corners_homography, new_corners_homography),
                    "timestamp": current_timestamp,
                }
            )

            frameCount += 1
            previous_timestamp = current_timestamp
            prev_frame = frame
            success, frame = vidcap.read()
        self.numFrames = frameCount
        self.duration = current_timestamp
        return

    def read_data(self):
        if not os.path.exists(self.videofiledata):
            print("start creating "+self.videofiledata)
            self.searchkeypoints()
            with open(self.videofiledata, "wb") as fp:
                pickle.dump(self, fp)
                return self
        else:
            with open(self.videofiledata, "rb") as fp:
                return pickle.load(fp)