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



def render_trio(signal_x, signal_y, signal_z, timestamps):
    """
    plot gyro data
    """
    plt.plot(
        timestamps,
        signal_x,
        "b-",
        timestamps,
        signal_y,
        "g-",
        timestamps,
        signal_z,
        "r-",
    )
    plt.ylabel("Y")
    plt.show()


class CalibrateGyroStabilize(object):
    def get_gaussian_kernel(self, sigma2, v1, v2, normalize=True):
        gauss = [math.exp(-(float(x * x) / sigma2)) for x in range(v1, v2 + 1)]
        total = sum(gauss)

        if normalize:
            gauss = [x / total for x in gauss]

        return gauss

    def gaussian_filter(self, input_array, sigma=10000, r=256):
        """
        """
        # Step 1: Define the convolution kernel
        kernel = self.get_gaussian_kernel(sigma, -r, r)

        # Step 2: Convolve
        return np.convolve(input_array, kernel, "same")

    def calcErrorScore(self, set1, set2):
        if len(set1) != len(set2):
            raise Exception("The given two sets don't have the same length")

        score = 0
        set1 = [(x[0][0], x[0][1]) for x in set1.tolist()]
        if not DO_ROLLING_SHUTTER:
            set2 = [(x[0][0], x[0][1]) for x in set2.tolist()]

        for first, second in zip(set1, set2):
            diff_x = math.pow(first[0] - second[0], 2)
            diff_y = math.pow(first[1] - second[1], 2)

            score += math.sqrt(diff_x + diff_y)

        return score

    def calcErrorAcrossVideo(
        self,
        videoObj,
        theta,
        timestamps,
        focal_length,
        gyro_delay=None,
        gyro_drift=None,
        rolling_shutter=None,
    ):
        total_error = 0
        frame_height = videoObj.frameHeight
        for frameCount in range(videoObj.numFrames):
            frameInfo = videoObj.frameInfo[frameCount]
            current_timestamp = frameInfo["timestamp"]

            if frameCount == 0:
                # INCRMENT
                # frameCount += 1
                previous_timestamp = current_timestamp
                continue

            keypoints = frameInfo["keypoints"]
            if keypoints:
                old_corners = frameInfo["keypoints"][0]
                new_corners = frameInfo["keypoints"][1]
            else:
                # Don't use this for calculating errors
                continue

            # Ideally, after our transformation, we should get points from the
            # thetas to match new_corners

            #########################
            # Step 0: Work with current parameters and calculate the error score

            transformed_corners = []
            if DO_ROLLING_SHUTTER:
                for pt in old_corners:
                    x = pt[0][0]
                    y = pt[0][1]

                    # The time when this pixel was captured - the timestamp is centered around the
                    # the center
                    pt_timestamp = (
                        int(current_timestamp)
                        + rolling_shutter * (y - frame_height / 2) / frame_height
                    )

                    transform = getAccumulatedRotation(
                        videoObj.frameWidth,
                        videoObj.frameHeight,
                        theta[0],
                        theta[1],
                        theta[2],
                        timestamps,
                        int(previous_timestamp),
                        int(pt_timestamp),
                        focal_length,
                        gyro_delay,
                        gyro_drift,
                        doSub=True,
                    )
                    output = transform * np.matrix("%f;%f;1.0" % (x, y)).tolist()
                    tx = (output[0][0] / output[2][0]).tolist()[0][0]
                    ty = (output[1][0] / output[2][0]).tolist()[0][0]
                    transformed_corners.append(np.array([tx, ty]))
            else:
                transform = getAccumulatedRotation(
                    videoObj.frameWidth,
                    videoObj.frameHeight,
                    theta[0],
                    theta[1],
                    theta[2],
                    timestamps,
                    int(previous_timestamp),
                    int(current_timestamp),
                    focal_length,
                    gyro_delay,
                    gyro_drift,
                    doSub=True,
                )
                transformed_corners = cv2.perspectiveTransform(old_corners, transform)

            error = self.calcErrorScore(new_corners, transformed_corners)

            # print "Error(%d) = %f" % (frameCount, error)

            total_error += error

            # For a random frame - write out the outputs
            if frameCount == MAX_FRAMES / 2:
                img = np.zeros((videoObj.frameHeight, videoObj.frameWidth, 3), np.uint8)
                for old, new, transformed in zip(
                    old_corners, new_corners, transformed_corners
                ):
                    pt_old = (int(old[0][0]), int(old[0][1]))
                    pt_new = (int(new[0][0]), int(new[0][1]))
                    pt_transformed = (int(transformed[0]), int(transformed[1]))
                    cv2.line(img, pt_old, pt_old, (0, 0, 255), 2)
                    cv2.line(img, pt_new, pt_new, (0, 255, 0), 1)
                    cv2.line(img, pt_transformed, pt_transformed, (0, 255, 255), 1)
                cv2.imwrite("./image_error/ddd%04d-a.png" % frameCount, img)

            # INCRMENT
            # frameCount += 1
            previous_timestamp = current_timestamp

        return total_error

    def calcErrorAcrossVideoObjective(self, parameters, videoObj, theta, timestamps):
        """
        Wrapper function for scipy
        """
        focal_length = float(parameters[0])
        gyro_delay = float(parameters[1])
        gyro_drift = (float(parameters[2]), float(parameters[3]), float(parameters[4]))
        rolling_shutter = float(parameters[5])

        # print "Focal length = %f" % focal_length
        # print "gyro_delay = %f" % gyro_delay
        # print "gyro_drift = (%f, %f, %f)" % gyro_drift

        error = self.calcErrorAcrossVideo(
            videoObj,
            theta,
            timestamps,
            focal_length,
            gyro_delay,
            gyro_drift,
            rolling_shutter,
        )
        print("Error = %f" % (error / videoObj.numFrames))
        return error

    def diff(self, timestamps):
        """
        Returns differences between consecutive elements
        """
        return np.ediff1d(timestamps)

    def __init__(self, mp4, csv):
        self.mp4 = mp4
        self.csv = csv

    def import_gyro_data(self, filename):
        gdf = calibration.GyroscopeDataFile(filename)
        gdf.parse()
        return gdf

    def read_video(self, filename):
        videoObj = GyroVideo(self.mp4)
        videoObj.read_video()
        return videoObj
        
    def calibrate(self):
        gdf = stored_data(gyrofilename,self.import_gyro_data,csv)
        

        signal_x = gdf.get_signal_x()
        signal_y = gdf.get_signal_y()
        signal_z = gdf.get_signal_z()
        timestamps = gdf.get_timestamps()

        # Smooth out the noise
        smooth_signal_x = self.gaussian_filter(signal_x)
        smooth_signal_y = self.gaussian_filter(signal_y)
        smooth_signal_z = self.gaussian_filter(signal_z)

        # plot
        # render_trio(signal_x, signal_y, signal_z, timestamps)
        # render_trio(smooth_signal_x, smooth_signal_y, smooth_signal_z, timestamps)

        # g is the difference between the smoothed version and the actual version
        g = [[], [], []]
        delta_g = [[], [], []]
        delta_g[0] = np.subtract(signal_x, smooth_signal_x).tolist()
        delta_g[1] = np.subtract(signal_y, smooth_signal_y).tolist()
        delta_g[2] = np.subtract(signal_z, smooth_signal_z).tolist()
        g[0] = signal_x  # np.subtract(signal_x, smooth_signal_x).tolist()
        g[1] = signal_y  # np.subtract(signal_y, smooth_signal_y).tolist()
        g[2] = signal_z  # np.subtract(signal_z, smooth_signal_z).tolist()
        dgt = utilities.diff(timestamps) # dgt 是每个时间戳的间隔

        theta = [[], [], []]
        delta_theta = [[], [], []]
        for component in [0, 1, 2]:
            sum_of_consecutives = np.add(g[component][:-1], g[component][1:]) #首尾相加，作为零漂进行校准
            # The 2 is for the integration - and 10e9 for the nanosecond
            dx_0 = np.divide(sum_of_consecutives, 2 * 1000000000) # 计算出每段事件的速度漂移
            num_0 = np.multiply(dx_0, dgt)
            theta[component] = [0]
            theta[component].extend(np.cumsum(num_0)) 

            sum_of_delta_consecutives = np.add(
                delta_g[component][:-1], delta_g[component][1:]
            )
            dx_0 = np.divide(sum_of_delta_consecutives, 2 * 1000000000)
            num_0 = np.multiply(dx_0, dgt)
            delta_theta[component] = [0]
            delta_theta[component].extend(np.cumsum(num_0)) # 同理计算出delta_theta的漂移

        # UNKNOWNS
        pixel_size = 2.9  # 2.9um
        focus_efficient = 3.2  # 21 mm
        focus_in_pixel = focus_efficient / (pixel_size / 1000)
        focal_length = focus_in_pixel
        gyro_delay = 0
        gyro_drift = (0, 0, 0)
        shutter_duration = 0

        # parts = self.mp4.split("/")
        # pickle_file_name = parts[-1].split(".")[0]
        # pickle_full_path = "%s/%s.pickle" % ("/".join(parts[:-1]), pickle_file_name)
        # print("Pickle file = %s" % pickle_full_path)
        pickle_full_path = "./videodata.data"
        videoObj = stored_data(pickle_full_path, self.read_video,self.mp4)

        print("Calibrating parameters")
        print("=====================+")

        parameters = np.asarray([focal_length, gyro_delay, gyro_drift[0], 
                                gyro_drift[1], gyro_drift[2], shutter_duration])

        import scipy.optimize

        result = scipy.optimize.minimize(
            self.calcErrorAcrossVideoObjective,
            parameters,
            (videoObj, theta, timestamps),
            "Nelder-Mead",
            tol=0.001,
        )
        print(result)

        focal_length = result["x"][0]
        gyro_delay = result["x"][1]
        gyro_drift = (result["x"][2], result["x"][3], result["x"][4])
        shutter_duration = result["x"][5]

        print("Focal length = %f" % focal_length)
        print("Gyro delay   = %f" % gyro_delay)
        print("Gyro drift   = (%f, %f, %f)" % gyro_drift)
        print("Shutter duration= %f" % shutter_duration)

        # Smooth out the delta_theta values - they must be fluctuating like crazy

        smooth_delta_x = self.gaussian_filter(delta_theta[0], 128, 16)
        smooth_delta_y = self.gaussian_filter(delta_theta[1], 128, 16)
        smooth_delta_z = self.gaussian_filter(delta_theta[2], 128, 16)
        return (
            delta_theta,
            timestamps,
            focal_length,
            gyro_delay,
            gyro_drift,
            shutter_duration,
        )
        # return ( (smooth_delta_x, smooth_delta_y, smooth_delta_z), timestamps, 
        # focal_length, gyro_delay, gyro_drift, shutter_duration)