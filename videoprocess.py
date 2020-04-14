import numpy as np
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

    # rotate image function
    # rx,ry,rz: Angle of rotation around the X/Y/Z axis
    # dx,dy,dz: translation around X/Y/Z, an optional translation
    # f:  the focal length in pixels
    # convertToRadians: whether the angles are in radians or not
    def rotateImage(self, src, rx, ry, rz, dx, dy, dz, f, convertToRadians=False):
        '''
        x: 向上为正数
        y: 向右为正数
        z: 顺时针为正数
        '''
        if convertToRadians:
            rx = (rx) * math.pi / 180
            ry = (ry) * math.pi / 180
            rz = (rz) * math.pi / 180

        rx = float(rx)
        ry = float(ry)
        rz = float(rz)

        # calculate the width and the height of the source image.
        w = src.shape[1]
        h = src.shape[0]

        x = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 0], [0, 0, 1]])
        # data interpreted as a matrix and does not make a copy
        A1 = np.asmatrix(x)

        # Calculate the rotation matrix from the angle
        smallR = cv2.Rodrigues(np.array([rx, ry, rz]))[0]

        #  convert that into a 4x4 homogeneous matrix
        # so that we can apply transformations to it.
        R = np.array(
            [
                [smallR[0][0], smallR[0][1], smallR[0][2], 0],
                [smallR[1][0], smallR[1][1], smallR[1][2], 0],
                [smallR[2][0], smallR[2][1], smallR[2][2], 0],
                [0, 0, 0, 1],
            ]
        )

        R_x = np.array([
            [1, 0 ,0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)]
        ])
        R_y = np.array([
            [math.cos(ry), 0 , math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)]
        ])
        R_z = np.array([
            [math.cos(rz), -math.sin(rz) ,0],
            [math.sin(rz), math.cos(rz), 0],
            [0, 0, 1]
        ])
        smallR = np.dot(np.dot(R_x, R_y),R_z)
        R = np.array(
            [
                [smallR[0][0], smallR[0][1], smallR[0][2], 0],
                [smallR[1][0], smallR[1][1], smallR[1][2], 0],
                [smallR[2][0], smallR[2][1], smallR[2][2], 0],
                [0, 0, 0, 1],
            ]
        )

        # It's usually a good idea to keep dz equal to the focal length.
        # This implies that the image was captured at just the right focal length
        # and needs to be rotated about that point.
        x = np.array([[1.0, 0, 0, dx], [0, 1.0, 0, dy], [0, 0, 1.0, dz], [0, 0, 0, 1.0]])

        T = np.asmatrix(x)

        x = np.array([[f, 0, w / 2, 0], [0, f, h / 2, 0], [0, 0, 1, 0]])

        # A1 = T = x = A2
        A2 = np.asmatrix(x)

        tmp = R*A1
        tmp1 = T*tmp
        transform = A2* tmp1
        transform = A2 * (T * (R * A1))
        # transform = np.array([
        #     [1.0,1.0,0.0],
        #     [1.0,1.0,0.0],
        #     [0.0,0.0,1.0]
        # ])
        o = cv2.warpPerspective(src, transform, (w, h))
        return o
    


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

def fetch_closest_trio(theta_x, theta_y, theta_z, timestamps, req_timestamp):
    """
    Returns the closest match for a given timestamp
    theta_x,theta_y,theta_z,timestamps: csv data
    req_timestamp: required timestamp
    first param : real theta with interpolation
    second and third params : preview and current timestamp
    """
    try:
        if req_timestamp in timestamps:
            indexOfTimestamp = timestamps.index(req_timestamp)
            return (
                (
                    theta_x[indexOfTimestamp],
                    theta_y[indexOfTimestamp],
                    theta_z[indexOfTimestamp],
                ),
                req_timestamp,
                None,
            )
    except IndexError:
        pdb.set_trace()

    i = 0
    sorted_keys = sorted(timestamps)
    for ts in sorted_keys:
        if ts > req_timestamp:
            break

        i += 1

    # We're looking for the ith and the i+1th req_timestamp
    t_previous = sorted_keys[i - 1]
    t_current = sorted_keys[i]
    dt = float(t_current - t_previous)

    slope = (req_timestamp - t_previous) / dt

    t_previous_index = timestamps.index(t_previous)
    t_current_index = timestamps.index(t_current)

    new_x = theta_x[t_previous_index] * (1 - slope) + theta_x[t_current_index] * slope
    new_y = theta_y[t_previous_index] * (1 - slope) + theta_y[t_current_index] * slope
    new_z = theta_z[t_previous_index] * (1 - slope) + theta_z[t_current_index] * slope

    return ((new_x, new_y, new_z), t_previous, t_current)

def getRodrigues(rx, ry, rz):
    smallR = cv2.Rodrigues(np.array([float(rx), float(ry), float(rz)]))[0]
    R = np.array(
        [
            [smallR[0][0], smallR[0][1], smallR[0][2], 0],
            [smallR[1][0], smallR[1][1], smallR[1][2], 0],
            [smallR[2][0], smallR[2][1], smallR[2][2], 0],
            [0, 0, 0, 1],
        ]
    )
    return R

def getAccumulatedRotation(
    w,
    h,
    theta_x,
    theta_y,
    theta_z,
    timestamps,
    prev,
    current,
    f,
    gyro_delay=None,
    gyro_drift=None,
    shutter_duration=None,
    doSub=False,
):
    """
    w, h: We need to know the size of the image to convert it from world space to image space.
    theta_*: Currently, we have access to angular velocity. From there, we can evaluate actual
    angles and that is what this function accepts as parameters.
    Timestamps: The time each sample was taken.
    prev, current: Accumulate rotations between these timestamps. This will usually provide
    the timestamp of the previous frame and the current frame.
    f, gyro_delay, gyro_drift, and shutter_duration are used to improve the estimate of the
    rotation matrix. The last three of these are optional
    (and they get set to zero if you don't pass them).
    """
    if not gyro_delay:
        gyro_delay = 0

    if not gyro_drift:
        gyro_drift = (0, 0, 0)

    if not shutter_duration:
        shutter_duration = 0

    x = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 0], [0, 0, 1]])
    A1 = np.asmatrix(x)
    transform = A1.copy()

    prev = prev + gyro_delay
    current = current + gyro_delay
    if prev in timestamps and current in timestamps:
        start_timestamp = prev
        end_timestamp = current
    else:
        (rot_start, start_timestamp, start_timestamp_next) = fetch_closest_trio(
            theta_x, theta_y, theta_z, timestamps, prev
        )
        (rot_end, end_timestamp, end_timestamp_next) = fetch_closest_trio(
            theta_x, theta_y, theta_z, timestamps, current
        )

    # add gyro_delay and gyro drift
    if start_timestamp == end_timestamp:
        time_shifted = start_timestamp + gyro_delay
        trio, t_previous, t_current = fetch_closest_trio(
            theta_x, theta_y, theta_z, timestamps, time_shifted
        )
    else:
        for time in range(
            timestamps.index(start_timestamp), timestamps.index(end_timestamp)
        ):
            time_shifted = timestamps[time] + gyro_delay
            trio, t_previous, t_current = fetch_closest_trio(
                theta_x, theta_y, theta_z, timestamps, time_shifted
            )

    gyro_drifted = (
        float(rot_end[0] + gyro_drift[0]),
        float(rot_end[1] + gyro_drift[1]),
        float(rot_end[2] + gyro_drift[2]),
    )
    # 这个地方肯定是有问题的啊，角速度不是最终的角度，那能直接相减呢！！！
    if doSub:
        gyro_drifted = (
            gyro_drifted[0] - rot_start[0],
            gyro_drifted[1] - rot_start[1],
            gyro_drifted[2] - rot_start[2],
        )

    # 需要根据实际的陀螺仪轴对运动进行补偿！！！
    R = getRodrigues(gyro_drifted[1], -gyro_drifted[0], -gyro_drifted[2])

    x = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, f], [0, 0, 0, 1.0]])
    T = np.asmatrix(x)
    x = np.array([[f, 0, w / 2, 0], [0, f, h / 2, 0], [0, 0, 1, 0]])
    transform = R * (T * transform)

    A2 = np.asmatrix(x)

    transform = A2 * transform

    return transform