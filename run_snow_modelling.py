import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Test the conversion accuracy
TEST = True
threshold = 0.00005

DEBUG = False
VERBOSE = True

### Config ###
# date = '2018_03_06'
# seq = '0001'
# frame_id = 33
date = '2019_02_27'
seq = '0004'
frame_id = 50
cam_id = '0'
DISTORTED = False

# Change this to your dataset path
BASE = '../../../dataset/cadcd/'
OUTPUT = 'output/'

if DISTORTED:
  path_type = 'raw'
else:
  path_type = 'labeled'

lidar_path = BASE + date + '/' + seq + "/" + path_type + "/lidar_points/data/" + format(frame_id, '010') + ".bin"
calib_path = BASE + date + '/' + seq + "/calib/"
img_path =  BASE + date + '/' + seq + "/" + path_type + "/image_0" + cam_id + "/data/" + format(frame_id, '010') + ".png"
annotations_path =  BASE + '/' + date + '/' + seq + "/3d_ann.json"
######

def azimuth_resolution(rpm):
    # 55.296 us / firing cycle
    return 0.02
    # return rpm * 360 / 60.0 * 55.296 * 10**(-6)


class SensorData:

    sensor_elevation_angles = np.array([15, 10.33, 7, 4.667, 3.333, 2.333, 1.667, 1.333, 1.0, 0.667, 0.333, 0, -0.333, \
        -0.667, -1, -1.333, -1.667, -2, -2.333, -2.667, -3, -3.333, -3.667, -4, -4.667, -5.333, -6.148, -7.254, \
        -8.843, -11.31, -15.639, -25])

    def __init__(self, input_file, beam_count, rpm) -> None:
        scan_data = np.fromfile(input_file, dtype=np.float32) # read binary file

        # scan_data is a single row of all the lidar values
        # we covert scan_data to a 2D array where each row contains a point [x, y, z, intensity]
        self.cartesian_data = scan_data.reshape((-1, 4))

        # each row contains a point[azimuth, elevation, range, intensity]
        self.spherical_data = self.cartesian_to_spherical(self.cartesian_data)

        # Sensor Model
        self.beam_count = beam_count
        self.azimuth_resolution = azimuth_resolution(rpm)
        self.azimuth_angles_count = int(round(360 / self.azimuth_resolution))
        if VERBOSE:
            print("beam_count: ", self.beam_count)
            print("azimuth_angles_count: ", self.azimuth_angles_count)
            print("azimuth_resolution: ", self.azimuth_resolution)


    def get_elevation_index(self, input_angle):
        return np.abs(self.sensor_elevation_angles - input_angle).argmin()


    def get_azimuth_index(self, input_angle):
        azimuth = (input_angle + 360 ) % 360 # Convert to positive angles
        return int(round((self.azimuth_angles_count - 1) * azimuth / 360))

    def test(self, input_angle):
        azimuth = (input_angle + 2 * np.pi ) % (2 * np.pi)

        return int(round(((self.azimuth_angles_count  - 1) * azimuth) / (2 * np.pi)))


    # If channel_count=1: range channel
    # If channel_count=2: range and intensity channels
    def init_range_img(self, channel_count=2):
        if VERBOSE:
            print("H: ", self.beam_count)
            print("W: ", self.azimuth_angles_count)
            print("C: ", channel_count)

        self.range_img = np.zeros((self.beam_count, self.azimuth_angles_count, channel_count))

    def generate_range_img(self):
        overlap_count = 0
        self.init_range_img()

        for i, datapoint in enumerate(self.spherical_data):
            azimuth = datapoint[0]
            elevation = datapoint[1]
            range = datapoint[2]
            intensity = datapoint[3]

            elevation_index = self.get_elevation_index(np.degrees(elevation))
            azimuth_index = self.test(azimuth)

            if self.range_img[elevation_index, azimuth_index, 0] != 0 or \
                self.range_img[elevation_index, azimuth_index, 1] != 0:
                if DEBUG:
                    print("Previous data:")
                    print("range: ", self.range_img[elevation_index, azimuth_index, 0])
                    print("intensity: ", self.range_img[elevation_index, azimuth_index, 1])

                    print("Incoming data:")
                    print("i: ", i)
                    print("azimuth: ", np.degrees(azimuth))
                    print("elevation: ", np.degrees(elevation))
                    print("range: ", range)
                    print("intensity: ", intensity)
                    print("elevation_index: ", elevation_index)            
                    print("azimuth_index: ", azimuth_index)
                    # raise Exception("NON ZERO")
                
                overlap_count = overlap_count + 1
                    

            self.range_img[elevation_index, azimuth_index, :] = [range, intensity]


        if VERBOSE:
            print("Number of MAPPED non-zero datapoints (Range image): ", int(len(np.nonzero(self.range_img)[0]) / 2))
            print("Number of ORIGINAL non-zero datapoints (Spherical): ", int(len(np.nonzero(self.spherical_data[:, 0])[0])))
            print("Number of ORIGINAL non-zero datapoints (Cartesian): ", int(len(np.nonzero(self.cartesian_data[:, 0])[0])))
            print("Number of overlapped datapoints: ", overlap_count)
        


    # Inverse Sensor Model
    def spherical_to_cartesian(self, data):
        azimuth = data[:, 0]
        elevation = data[:, 1]
        range = data[:, 2]
        intensity = data[:, 3]

        cos_e = np.cos(elevation)

        x = np.multiply(range, np.multiply(np.cos(azimuth), cos_e))
        y = np.multiply(range, np.multiply(np.sin(azimuth), cos_e))
        z = np.multiply(range, np.sin(elevation))

        return np.vstack((x, y, z, intensity)).T

    # Forward Sensor Model
    def cartesian_to_spherical(self, data):
        origin = (0, 0, 0)
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        intensity = data[:, 3]

        range = np.linalg.norm([(x, y, z), origin])

        azimuth = np.arctan2(y, x)
        
        elevation = np.arcsin(np.divide(z, range))

        return np.vstack((azimuth, elevation, range, intensity)).T 

    def plot_range_img(self, selection='both', save=True, color=True):
        fig = plt.figure(figsize=(40,5))
        if selection == 'both':
            plt.subplot(2, 1, 1)
            if color:
                plt.imshow(self.range_img[:, :, 0])
            else:
                plt.imshow(self.range_img[:, :, 0], cmap='gray')
            plt.title("Distance")

            plt.subplot(2, 1, 2)
            if color:
                plt.imshow(self.range_img[:, :, 1])
            else:
                plt.imshow(self.range_img[:, :, 1], cmap='gray')
            plt.title("Intensity")

            plt.suptitle("Range Image")
        elif selection == 'distance':
            if color:
                plt.imshow(self.range_img[:, :, 0])
            else:
                plt.imshow(self.range_img[:, :, 0], cmap='gray')
            plt.title("Range Image: Distance")
        elif selection == 'intensity':
            if color:
                plt.imshow(self.range_img[:, :, 1])
            else:
                plt.imshow(self.range_img[:, :, 1], cmap='gray')
            plt.title("Range Image: Intensity")
        else:
            raise Exception("Possible 'selection' parameters are: 'both', 'distance', 'intensity'. ")

        if save == True:
            output_dir = Path(OUTPUT)
            output_dir.mkdir(exist_ok=True)
            fig.savefig(str(output_dir) + '/range_image_' + str(frame_id) + '.png', dpi=200)
        else:
            plt.show()

    def test_conversion_accuracy(self, threshold=0.00005):
        new_cartesian_data = self.spherical_to_cartesian(self.spherical_data)
        
        for i, _ in enumerate(new_cartesian_data):
            if (new_cartesian_data[i, 0] - self.cartesian_data[i, 0]) > threshold or \
                (new_cartesian_data[i, 1] - self.cartesian_data[i, 1]) > threshold or \
                (new_cartesian_data[i, 2] - self.cartesian_data[i, 2]) > threshold:

                print("Initial Data point in Cartesian Coordinates")
                print("lidar_x: ", self.cartesian_data[i, 0])
                print("lidar_y: ", self.cartesian_data[i, 1])
                print("lidar_z: ", self.cartesian_data[i, 2])

                print("Data point converted to Spherical Coordinates")
                print("lidar_range: ", self.spherical_data[i, 0])
                print("lidar_azimuth: ", self.spherical_data[i, 1])
                print("lidar_elevation: ", self.spherical_data[i, 2])

                print("Data point converted back to Cartesian Coordinates")
                print("new_lidar_x: ", new_cartesian_data[i, 0])
                print("new_lidar_y: ", new_cartesian_data[i, 1])
                print("new_lidar_z: ", new_cartesian_data[i, 2])

                raise Exception("The conversion loss is greater than the set threshold of " + str(threshold))

    def plot_spherical_points_distribution(self):
        plt.figure(figsize=(15,15))
        for col_idx, col_name in enumerate(["azimuth", "elevation", "range", "intensity"]):
            plt.subplot(2,2, col_idx + 1)
            plt.hist(np.degrees(self.spherical_data[:, np.r_[col_idx:col_idx+1]]), alpha=0.5)
            plt.title("Distribution of " + col_name)
        plt.show()

def main():

    # Convert Point Cloud to Range Map
    lidar_data = SensorData(input_file=lidar_path, beam_count=32, rpm=600)
    lidar_data.generate_range_img()
    # lidar_data.plot_range_img()
    lidar_data.plot_spherical_points_distribution()
    

    if TEST:
        lidar_data.test_conversion_accuracy(threshold)
    

if __name__ == "__main__":
    main()

