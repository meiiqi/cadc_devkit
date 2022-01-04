import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Test the conversion accuracy
TEST = True
threshold = 0.00005

DEBUG = False
VERBOSE = True
DISTORTED = False
OVERLAPS = False
SAVE = False

point_start = 0
point_end = None

### Config ###
date = '2018_03_06'
seq = '0001'
frame_id = 1
# date = '2019_02_27'
# seq = '0004'
# frame_id = 50
cam_id = '0'

# Change this to your dataset path
BASE = '../../../dataset/cadcd/'
OUTPUT = 'output/'

if DISTORTED:
  path_type = 'raw'
else:
  path_type = 'labeled'

lidar_path = BASE + date + '/' + seq + "/" + path_type + "/lidar_points/data/" + format(frame_id, '010') + ".bin"
######

def azimuth_resolution(rpm):
    # 55.296 us / firing cycle
    # return 0.2
    # return (rpm * 360 / 60.0 * 55.296 * 10**(-6)) / 2
    return rpm * 360 / 60.0 * 55.296 * 10**(-6)


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
        self.azimuth_angles_count = int(np.round(360 / self.azimuth_resolution))
        self.new_azimuth_resolution = 360 / self.azimuth_angles_count
        self.elevation_resolution = 0.2
        self.elevation_count = int((15 + 25) / self.elevation_resolution)

        if VERBOSE:
            print("beam_count: ", self.beam_count)
            print("azimuth_angles_count: ", self.azimuth_angles_count)
            print("azimuth_resolution: ", self.azimuth_resolution)
            print("new_azimuth_resolution: ", self.new_azimuth_resolution)
            print("elevation_count: ", self.elevation_count)


    def get_elevation_index(self, input_angle):
        tmp = np.abs(self.sensor_elevation_angles - input_angle).argmin()
        # print("elevation index: ", tmp)
        return tmp

    def elevation_test(self, input_angle):
        elevation = (input_angle * (-1) + 15 ) % 40
        idx =  int(np.floor(elevation / self.elevation_resolution))
        return idx



    def get_azimuth_index(self, input_angle):
        azimuth = (input_angle + 360 ) % 360 # Convert to positive angles
        return int(round((self.azimuth_angles_count - 1) * azimuth / 360))
        # return int(round((self.azimuth_angles_count ) * azimuth / 360)) - 1


    def azimuth_test2(self, input_angle):
        azimuth = (input_angle + 360 ) % 360
        idx = int(np.floor(azimuth / self.new_azimuth_resolution))
        return idx

    def azimuth_test(self, input_angle, toprint=False):
        # print("input_angle: ", input_angle)
        # print("input_angle: ", np.degrees(input_angle))
        azimuth = (input_angle + 2 * np.pi ) % (2 * np.pi)
        if toprint:
            print("azimuth:", azimuth)
            print("azimuth:", np.degrees(azimuth))
        intermediate = (((self.azimuth_angles_count  - 1) * azimuth) / (2 * np.pi))
        # print("intermediate: ", intermediate)
        # intermediate = (((self.azimuth_angles_count) * azimuth) / (2 * np.pi))
        # print("intermediate: ", intermediate2)
        if toprint:
            print("(intermediate): ", intermediate)
            print("round(intermediate): ", round(intermediate))
        # result = int(np.around(intermediate)) - 1
        result = int(round(intermediate))
        return result


    # If channel_count=1: range channel
    # If channel_count=2: range and intensity channels
    def init_range_img(self, channel_count=2):
        if VERBOSE:
            print("H: ", self.beam_count)
            print("W: ", self.azimuth_angles_count)
            print("C: ", channel_count)

        self.range_img = np.zeros((self.beam_count, self.azimuth_angles_count, channel_count))
        self.range_img_overlaps = np.zeros((self.beam_count, self.azimuth_angles_count, channel_count))

    def generate_range_img(self):
        overlap_count = 0
        self.init_range_img()

        for i, datapoint in enumerate(self.spherical_data[point_start:point_end, :]):
            if True:
            # if (point_start + i) not in [267, 279, 324, 438, 462, 663, 666, 764]:
                azimuth = datapoint[0]
                elevation = datapoint[1]
                range = datapoint[2]
                intensity = datapoint[3]

                # print("==============")
                # print("POINT: ", point_start + i)
                # print("(original) azimuth: ", np.degrees(azimuth))
                # print("(original) elevation: ", np.degrees(elevation))
                # elevation_index = self.get_elevation_index(np.degrees(elevation))
                elevation_index = self.get_elevation_index(np.degrees(elevation))
                toprint = False
                # if i==521 or i==546:
                #     toprint =True
                azimuth_index = self.azimuth_test2(np.degrees(azimuth))

                # if azimuth_index == 876 and elevation_index == 30:
                #     print("POINT: ", point_start + i)
                #     print("(original) azimuth: ", np.degrees(azimuth))
                #     print("(original) elevation: ", np.degrees(elevation))
                #     print("azimuth_index: ", azimuth_index)
                #     print("elevation_index: ", elevation_index) 

                if self.range_img[elevation_index, azimuth_index, 0] != 0 or \
                    self.range_img[elevation_index, azimuth_index, 1] != 0:
                    if DEBUG:
                        print("*****")
                        print("Previous data:")
                        print("range: ", self.range_img[elevation_index, azimuth_index, 0])
                        print("intensity: ", self.range_img[elevation_index, azimuth_index, 1])

                        print("Incoming data:")
                        print("i: ", i)
                        print("azimuth: ", np.degrees(azimuth))
                        print("elevation: ", np.degrees(elevation))
                        print("range: ", range)
                        print("intensity: ", intensity)
                        print("azimuth_index: ", azimuth_index)
                        print("elevation_index: ", elevation_index)            
                        # raise Exception("NON ZERO")
                    
                    overlap_count = overlap_count + 1
                    if range < 10 :
                        self.range_img_overlaps[elevation_index, azimuth_index, :] = [1, 1]
                    else:
                        self.range_img_overlaps[elevation_index, azimuth_index, :] = [2, 2]
                        

                self.range_img[elevation_index, azimuth_index, :] = [range, intensity]


        if VERBOSE:
            print("Number of MAPPED non-zero Range datapoints (Range image): ", len(np.nonzero(self.range_img[:, :, 0])[0]))
            print("Number of MAPPED non-zero Intensity datapoints (Range image): ", len(np.nonzero(self.range_img[:, :, 1])[0]))
            print("Number of ORIGINAL non-zero Range datapoints (Spherical): ", int(len(np.nonzero(self.spherical_data[point_start:point_end, 2])[0])))
            print("Number of ORIGINAL non-zero Intensity datapoints (Spherical): ", int(len(np.nonzero(self.spherical_data[point_start:point_end, 3])[0])))
            # print("Number of ORIGINAL non-zero datapoints (Spherical): ", int(len(np.nonzero(self.spherical_data[:, 0])[0])))
            print("Number of ORIGINAL non-zero Intensity datapoints (Cartesian): ", int(len(np.nonzero(self.cartesian_data[point_start:point_end, 3])[0])))
            print("Number of ORIGINAL non-zero datapoints (Cartesian): ", int(len(np.nonzero(self.cartesian_data[:, 0])[0])))
            print("Number of overlapped datapoints: ", overlap_count)
        


    # Inverse Sensor Model
    def spherical_to_cartesian(self, data):
        azimuth = data[:, 0]
        elevation = data[:, 1]
        range = data[:, 2]
        intensity = data[:, 3]

        cos_e = np.cos(elevation)

        x = range * (np.cos(azimuth) * cos_e)
        y = range * (np.sin(azimuth) * cos_e)
        z = range * np.sin(elevation)
        # x = np.multiply(range, np.multiply(np.cos(azimuth), cos_e))
        # y = np.multiply(range, np.multiply(np.sin(azimuth), cos_e))
        # z = np.multiply(range, np.sin(elevation))

        return np.vstack((x, y, z, intensity)).T

    # Forward Sensor Model
    def cartesian_to_spherical(self, data):
        origin = (0, 0, 0)
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        intensity = data[:, 3]

        range = np.sqrt(x**2 + y**2 + z**2)
        # range = np.linalg.norm([(x, y, z), origin])

        azimuth = np.arctan2(y, x)
        
        elevation = np.arcsin(z / range)
        # elevation = np.arcsin(np.divide(z, range))

        return np.vstack((azimuth, elevation, range, intensity)).T 

    def plot_range_img(self, selection='both', save=True, color=True):
        fig = plt.figure(figsize=(40,5))
        if selection == 'both':
            plt.subplot(2, 1, 1)
            if color:
                if OVERLAPS:
                    plt.imshow(self.range_img_overlaps[:, :, 0])
                else:
                    plt.imshow(self.range_img[:, :, 0])
            else:
                plt.imshow(self.range_img[:, :, 0], cmap='gray')
            plt.title("Distance")

            plt.subplot(2, 1, 2)
            if color:
                if OVERLAPS:
                    plt.imshow(self.range_img_overlaps[:, :, 1])
                else:
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
            if DISTORTED:
                if OVERLAPS:
                    fig.savefig(str(output_dir) + '/range_image_distorted_overlaps_' + str(frame_id) + '.png', dpi=200)
                else:
                    fig.savefig(str(output_dir) + '/range_image_distorted_' + str(frame_id) + '.png', dpi=200)
            else:
                if OVERLAPS:
                    fig.savefig(str(output_dir) + '/range_image_overlaps_' + str(frame_id) + '.png', dpi=200)
                else:
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

    def plot_spherical_points_distribution(self, save=True):
        fig = plt.figure(figsize=(20,20))
        for col_idx, col_name in enumerate(["azimuth", "elevation", "range", "intensity"]):
            plt.subplot(2,2, col_idx + 1)
            plt.hist(np.degrees(self.spherical_data[:, np.r_[col_idx:col_idx+1]]), alpha=0.5)
            plt.xlabel(col_name)
            plt.ylabel('Frequency')
            plt.title("Distribution of " + col_name)

        if save == True:
            output_dir = Path(OUTPUT)
            output_dir.mkdir(exist_ok=True)
            fig.savefig(str(output_dir) + '/spherical_pointclouds_distribution_' + str(frame_id) + '.png', dpi=200)
        else:
            plt.show()

def main():

    # Convert Point Cloud to Range Map
    lidar_data = SensorData(input_file=lidar_path, beam_count=32, rpm=600)
    lidar_data.generate_range_img()
    lidar_data.plot_range_img(save=SAVE)
    # lidar_data.plot_spherical_points_distribution(save=False)
    

    if TEST:
        lidar_data.test_conversion_accuracy(threshold)
    

if __name__ == "__main__":
    main()

