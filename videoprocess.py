import numpy
import cv2
import matplotlib.pyplot as plt

class VideoProcessFunc(object):
    def __init__(self, mp4, csv):
        self.mp4 = mp4
        self.csv = csv

    def rotateImage(self, image):
        pixel_size = 2.9  # 2.9um
        focus_efficient = 21  # 21 mm
        focus_in_pixel = focus_efficient / (pixel_size / 1000)
        plt.figure(2)
        plt.subplot(211)
        plt.imshow(frame)
        # frame1 = stable.rotateImage(
        #     frame, 0, 0, 0, 0, 0, focus_in_pixel, focus_in_pixel, True
        # )
        frame1 = stable.rotateImage(
            frame, 0.000427, 0.000046, -0.001556, 0, 0, 1103, 1103, False
        )
        plt.subplot(212)
        plt.imshow(frame1)
        plt.show()
        return
    
    # def stabilize_video(self):
    #     calib_obj = CalibrateGyroStabilize(mp4, csv)
    #     delta_theta, timestamps, focal_length, gyro_delay, gyro_drift, shutter_duration = (
    #         calib_obj.calibrate()
    #     )
        
    #     # Now start reading the frames
    #     vidcap = cv2.VideoCapture(mp4)

    #     frameCount = 0
    #     success, frame = vidcap.read()
    #     previous_timestamp = 0
    #     while success:
    #         print("Processing frame %d" % frameCount)
    #         # Timestamp in nanoseconds
    #         current_timestamp = vidcap.get(0) * 1000 * 1000
    #         print("    timestamp = %s ns" % current_timestamp)
    #         rot, prev, current = fetch_closest_trio(
    #             delta_theta[0],
    #             delta_theta[1],
    #             delta_theta[2],
    #             timestamps,
    #             current_timestamp,
    #         )

    #         rot = accumulateRotation(
    #             frame,
    #             delta_theta[0],
    #             delta_theta[1],
    #             delta_theta[2],
    #             timestamps,
    #             previous_timestamp,
    #             prev,
    #             focal_length,
    #             gyro_delay,
    #             gyro_drift,
    #         )

    #         # print "    rotation: %f, %f, %f" % (rot[0] * 180 / math.pi,
    #         #                                    rot[1] * 180 / math.pi,
    #         #                                    rot[2] * 180 / math.pi)
    #         cv2.imwrite("./tmp/frame%04d.png" % frameCount, frame)
    #         cv2.imwrite("./tmp/rotated%04d.png" % frameCount, rot)
    #         frameCount += 1
    #         previous_timestamp = prev
    #         success, frame = vidcap.read()

    #         if frameCount == MAX_FRAMES:
    #             break