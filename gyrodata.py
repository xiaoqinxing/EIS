import numpy
import pickle
import os

class GyroscopeDataFile(object):
    # the file path to read
    # a dictionary of angular velocities
    # This dictionary will store mappings between the timestamp
    # and the angular velocity at that instant.
    def __init__(self, filepath):
        self.filepath = filepath
        self.omega = {}
        self.gyrofilename = "./.gyroscopedata.data"

    def _getfile_object(self):
        return open(self.filepath)

    # read the file and populate the Omega dictionary.
    def parse(self):
        with self._getfile_object() as fp:
            # We validate that the first line of the csv file matches our
            # expectation. If not, the csv file was probably not compatible
            # and will error out over the next few lines.
            firstline = fp.readline().strip()
            if not firstline == "gyro":
                raise Exception("The first line isn't valid")

            # The strip function removed any additional whitespace
            # (tabs, spaces, newline characters, among others)
            # that might be stored in the file.
            for line in fp.readlines():
                line = line.strip()
                parts = line.split(",")
                # convert strings into numetric type
                timestamp = int(parts[3])
                ox = float(parts[0])
                oy = float(parts[1])
                oz = float(parts[2])
                '''
                if timestamp < 100000000:
                    timestamp = timestamp * 10
                if timestamp < 100000000:
                    timestamp = timestamp * 10
                '''
                print("%s: %s, %s, %s" % (timestamp, ox, oy, oz))
                self.omega[timestamp] = (ox, oy, oz)
            return

    # return a sorted list of timestamps from small to large num
    def get_timestamps(self):
        return sorted(self.omega.keys())

    # extract a specific component of the signal
    # get_signal(0) returns the X component of angular velocity
    def get_signal(self, index):
        return [self.omega[k][index] for k in self.get_timestamps()]

    def get_signal_x(self):
        return self.get_signal(0)

    def get_signal_y(self):
        return self.get_signal(1)

    def get_signal_z(self):
        return self.get_signal(2)

    # simple linear interpolation to estimate the angular velocity
    def fetch_approximate_omega(self, timestamp):
        if timestamp in self.omega:
            return self.omega[timestamp]

        # walking over the timestamps and finding the timestamp
        # that is closest to the one requested.
        i = 0
        sorted_timestamps = self.get_timestamps()
        for ts in sorted_timestamps:
            if ts > timestamp:
                break
            i += 1
        t_previous = sorted_timestamps[i - 1]  # this is a error in the book
        t_current = sorted_timestamps[i]
        dt = float(t_current - t_previous)
        slope = (timestamp - t_previous) / dt

        est_x = (
            self.omega[t_previous][0] * (1 - slope) + self.omega[t_current][0] * slope
        )
        est_y = (
            self.omega[t_previous][1] * (1 - slope) + self.omega[t_current][1] * slope
        )
        est_z = (
            self.omega[t_previous][2] * (1 - slope) + self.omega[t_current][2] * slope
        )
        return (est_x, est_y, est_z)
    
    def read_gyro_data(self):
        if not os.path.exists(self.gyrofilename):
            print("start creating "+self.gyrofilename)
            self.parse()
            with open(self.gyrofilename, "wb") as fp:
                pickle.dump(self.omega, fp)
                return self.omega
        else:
            with open(self.gyrofilename, "rb") as fp:
                return pickle.load(fp)