import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import yaml

class SensorData:

    def __init__(self, args) -> None:
        self.input_dir = args.input_dir
        self.frame = args.frame
        self.frame_start = args.frame_start
        self.frame_end = args.frame_end 
        self.point_start = args.point_start
        self.point_end = args.point_end 
        self.channel_count = args.channel_count 
        self.beam_count = args.beam_count
        self.rpm = args.rpm
        self.firing_cycle_us = args.firing_cycle_us

        self.elevation_angles = np.array(args.elevation_angles.strip("[]").split(", "), dtype=np.float32)
        if len(self.elevation_angles) != self.beam_count:
            raise Exception("Number of elevation angles ", len(self.elevation_angles) ," does not equal to Beam count ", self.beam_count, ".")

        self.verbose = args.verbose
        self.test = args.test
        self.save = args.save
        self.plot = args.plot
        self.overlaps = args.overlaps
        self.frame_count_in_dir = len([datafile for datafile in os.listdir(self.input_dir) if os.path.isfile(os.path.join(self.input_dir, datafile))])
        self.frames_to_run = []
        self.raw_scans = {}
        self.cartesian_data = {}
        self.spherical_data = {}
        self.range_images = {}
        
        if self.overlaps:
            self.overlap_count = {}
            self.range_images_overlaps = {}

        # Determine which frames to run
        if self.frame is None and self.frame_start is None and self.frame_end is None:
            self.frames_to_run = range(self.frame_count_in_dir)   
        elif self.frame is None and self.frame_start is None and self.frame_end is not None:
            self.frames_to_run = range(self.frame_end)
        elif self.frame is None and self.frame_start is not None and self.frame_end is not None:
            self.frames_to_run = range(self.frame_start, self.frame_end)
        elif self.frame is not None:
            self.frames_to_run = [self.frame]

        self.azimuth_resolution = self.get_azimuth_resolution(self.rpm, self.firing_cycle_us)
        self.azimuth_count = self.get_azimuth_count(self.azimuth_resolution)
        self.updated_azimuth_resolution = 360.0 / self.azimuth_count


        if self.verbose:
            print("input_dir: ", self.input_dir)
            print("frame: ", self.frame)
            print("frame_start: ", self.frame_start)
            print("frame_end: ", self.frame_end)
            print("point_start: ", self.point_start)
            print("point_end: ", self.point_end)
            print("frames_to_run: ", self.frames_to_run)
            print("frame_count_in_dir: ", self.frame_count_in_dir)
            print("beam_count: ", self.beam_count)
            print("rpm: ", self.rpm)
            print("elevation_angles: ", self.elevation_angles)
            print("firing_cycle_us: ", self.firing_cycle_us)
            print("azimuth_resolution: ", self.azimuth_resolution)
            print("azimuth_count: ", self.azimuth_count)


    def get_azimuth_resolution(self, rpm, firing_cycle_us) -> float:
        return rpm * 360 / 60.0 * firing_cycle_us * 10**(-6)

    def get_azimuth_count(self, az_resolution) -> int:
        return int(np.round(360.0 / az_resolution))

    def get_raw_scans_cartesian(self) -> dict:
        '''Load raw scans point cloud data from binary files'''
        for i in self.frames_to_run:
            filepath = os.path.join(self.input_dir, format(i, '010') + ".bin")
            print(filepath)
            data = np.fromfile(filepath, dtype=np.float32)
            self.raw_scans[i] = data
            self.cartesian_data[i] = data.reshape((-1, 4))

        return self.cartesian_data

    def cartesian_to_spherical(self, cartesian_data: dict) -> dict:
        '''Convert points from the Cartesian coordinate system to the Spherical coordinate system'''
        for i, data in cartesian_data.items():
            x = data[:, 0]
            y = data[:, 1]
            z = data[:, 2]
            intensity = data[:, 3]

            range = np.sqrt(x**2 + y**2 + z**2)

            azimuth = np.arctan2(y, x)
            
            elevation = np.arcsin(z / range)

            self.spherical_data[i] = np.vstack((azimuth, elevation, range, intensity)).T 

        if self.test:
            self.test_conversion_accuracy(self.test)

        return self.spherical_data

    
    def spherical_to_cartesian(self, spherical_data: dict) -> dict:
        '''Convert points from the Cartesian coordinate system to the Spherical coordinate system'''
        cartesian_data = {}
        for i, data in spherical_data.items():
            azimuth = data[:, 0]
            elevation = data[:, 1]
            range = data[:, 2]
            intensity = data[:, 3]

            cos_e = np.cos(elevation)

            x = range * (np.cos(azimuth) * cos_e)
            y = range * (np.sin(azimuth) * cos_e)
            z = range * np.sin(elevation)

            cartesian_data[i] = np.vstack((x, y, z, intensity)).T

        return cartesian_data

    def get_elevation_index(self, input_angle):
        return np.abs(self.elevation_angles - input_angle).argmin()

    def get_azimuth_index(self, input_angle):
        azimuth = (input_angle + 360 ) % 360 # Convert to positive angles
        return int(np.floor(azimuth / self.updated_azimuth_resolution))

    def pixel_is_filled(self, pixel) -> bool:
        return pixel[0] != 0 or pixel[1] != 0
     
    def generate_range_img(self, spherical_data: dict) -> dict:
        '''Generate a range image for each lidar frame.'''
        for frame_idx, frame_data in spherical_data.items():
            # Init
            range_img = np.zeros((self.beam_count, self.azimuth_count, self.channel_count))
            
            if self.overlaps:
                range_img_overlaps = range_img.copy()
                overlap = 0
            
            for i, datapoint in enumerate(frame_data[self.point_start:self.point_end, :]):
                azimuth = datapoint[0]
                elevation = datapoint[1]
                range = datapoint[2]
                intensity = datapoint[3]

                elevation_index = self.get_elevation_index(np.degrees(elevation))
                azimuth_index = self.get_azimuth_index(np.degrees(azimuth))

                if self.pixel_is_filled(range_img[elevation_index, azimuth_index]):
                    # print("[ERROR]: point # ", i , " overlaps at elevation ", elevation_index, " and azimuth ", azimuth_index)
                    if self.overlaps:
                        overlap = overlap + 1
                        range_img_overlaps[elevation_index, azimuth_index, :] = [1, 1]
        
                range_img[elevation_index, azimuth_index, :] = [range, intensity]

            self.range_images[frame_idx] = range_img
            
            if self.overlaps:
                self.range_images_overlaps[frame_idx] = range_img_overlaps
                self.overlap_count[frame_idx] = overlap

            if self.verbose:
                print("Number of MAPPED non-zero Range datapoints (Range image): ", len(np.nonzero(self.range_images[frame_idx][:, :, 0])[0]))
                print("Number of MAPPED non-zero Intensity datapoints (Range image): ", len(np.nonzero(self.range_images[frame_idx][:, :, 1])[0]))
                print("Number of ORIGINAL non-zero Range datapoints (Spherical): ", int(len(np.nonzero(self.spherical_data[frame_idx][self.point_start:self.point_end, 2])[0])))
                print("Number of ORIGINAL non-zero Intensity datapoints (Spherical): ", int(len(np.nonzero(self.spherical_data[frame_idx][self.point_start:self.point_end, 3])[0])))
                print("Number of ORIGINAL non-zero Intensity datapoints (Cartesian): ", int(len(np.nonzero(self.cartesian_data[frame_idx][self.point_start:self.point_end, 3])[0])))
                print("Number of ORIGINAL non-zero datapoints (Cartesian): ", int(len(np.nonzero(self.cartesian_data[frame_idx][:, 0])[0])))
                if self.overlaps:
                    print("Number of overlapped datapoints: ", overlap)

        if self.save:
            if self.overlaps:
                with open(self.save, 'w') as f:
                    yaml.dump(self.overlap_count, f, default_flow_style=False) 

    def is_raw(self):
        if 'raw' in self.input_dir and 'raw\lidar_points_corrected' not in self.input_dir and 'labeled' not in self.input_dir:
            return True
        else:
            return False

    def plot_range_img(self, frame_idx):
        fig = plt.figure(figsize=(40,5))
        if self.overlaps:
            plt.subplot(3, 1, 1)
        else:
            plt.subplot(2, 1, 1)
        plt.imshow(self.range_images[frame_idx][:, :, 0])
        plt.title("Distance")

        if self.overlaps:
            plt.subplot(3, 1, 2)
        else:
            plt.subplot(2, 1, 2)
        plt.imshow(self.range_images[frame_idx][:, :, 1])
        plt.title("Intensity")

        if self.overlaps:
            plt.subplot(3, 1, 3)
            plt.imshow(self.range_images_overlaps[frame_idx][:, :, 0])
            plt.title("Overlapping Points")
        
        plt.suptitle("Range Image")
        
        if self.plot:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            if self.is_raw():
                if self.overlaps:
                    fig.savefig(str(output_dir) + '/range_image_distorted_overlaps_' + str(frame_idx) + '.png', dpi=200)
                else:
                    fig.savefig(str(output_dir) + '/range_image_distorted_' + str(frame_idx) + '.png', dpi=200)
            else:
                if self.overlaps:
                    fig.savefig(str(output_dir) + '/range_image_overlaps_' + str(frame_idx) + '.png', dpi=200)
                else:
                    fig.savefig(str(output_dir) + '/range_image_' + str(frame_idx) + '.png', dpi=200)
        else:
            plt.show()

    def test_conversion_accuracy(self, threshold):
        new_cartesian_data = self.spherical_to_cartesian(self.spherical_data)
        
        for frame_idx, frame_data in new_cartesian_data.items():
            for i, _ in enumerate(frame_data):
                if (frame_data[i, 0] - self.cartesian_data[frame_idx][i, 0]) > threshold or \
                    (frame_data[i, 1] - self.cartesian_data[frame_idx][i, 1]) > threshold or \
                    (frame_data[i, 2] - self.cartesian_data[frame_idx][i, 2]) > threshold:

                    print("Initial Data point in Cartesian Coordinates")
                    print("lidar_x: ", self.cartesian_data[frame_idx][i, 0])
                    print("lidar_y: ", self.cartesian_data[frame_idx][i, 1])
                    print("lidar_z: ", self.cartesian_data[frame_idx][i, 2])

                    print("Data point converted to Spherical Coordinates")
                    print("lidar_range: ", self.spherical_data[frame_idx][i, 0])
                    print("lidar_azimuth: ", self.spherical_data[frame_idx][i, 1])
                    print("lidar_elevation: ", self.spherical_data[frame_idx][i, 2])

                    print("Data point converted back to Cartesian Coordinates")
                    print("new_lidar_x: ", frame_data[i, 0])
                    print("new_lidar_y: ", frame_data[i, 1])
                    print("new_lidar_z: ", frame_data[i, 2])

                    raise Exception("The conversion loss is greater than the set threshold of " + str(threshold))


def parse_cmdline():
    parser = argparse.ArgumentParser(description='Get Range Image from Point Cloud')
    parser.add_argument(
        '-i', '--input_dir', dest="input_dir", type=str, required=True, help='Input directory containing the lidar binary files.')
    parser.add_argument(
        '-f', '--frame', dest="frame", type=int, help='Specify which frame to run. If not specified, will run on all the frames_to_run.')
    parser.add_argument(
        '-fs', '--frame_start', dest="frame_start", type=int, default=0, help='Specify which frame to start running. If not specified, will start on first frame.')
    parser.add_argument(
        '-fe', '--frame_end', dest="frame_end", type=int, help='Specify which frame to stop running (non inclusive). If not specified, will end until last frame.')
    parser.add_argument(
        '-ps', '--point_start', dest="point_start", type=int, default=0, help='Specify which point to start processing. If not specified, will start on first point.')
    parser.add_argument(
        '-pe', '--point_end', dest="point_end", type=int, help='Specify which point to stop processing (non inclusive). If not specified, will end until last point.')
    parser.add_argument(
        '-c', '--channel_count', dest="channel_count", type=int, default=2, help='Specify the number of channels to generate for the range image. Distance and Intensity are two channels.')
    parser.add_argument(
        '-b', '--beam_count', dest="beam_count", type=int, default=32, help='Specify the number of beams in your lidar sensor.')
    parser.add_argument(
        '-r', '--rpm', dest="rpm", type=int, default=600, help='Specify the RPM rotating speed of your lidar sensor.')
    parser.add_argument(
        '-e', '--elevation_angles', dest="elevation_angles", type=str, default="[15, 10.33, 7, 4.667, 3.333, 2.333, 1.667, 1.333, 1.0, 0.667, 0.333, 0, -0.333, -0.667, -1, -1.333, -1.667, -2, -2.333, -2.667, -3, -3.333, -3.667, -4, -4.667, -5.333, -6.148, -7.254, -8.843, -11.31, -15.639, -25]", help='Specify the list of elevation angles of the lidar beams.')
    parser.add_argument(
        '--firing_cycle_us', dest="firing_cycle_us", type=float, default=55.296, help='Specify the firing cycle in micro seconds of the lidar sensor.')
    parser.add_argument(
        '-v', '--verbose', dest="verbose", action="store_true", help='Verbosity to enable logging')
    parser.add_argument(
        '-o', '--overlaps', dest="overlaps", action="store_true", help='Show range image of pixels that overlapped.')
    parser.add_argument(
        '-t', '--test', dest="test",  type=float, default=0.00005, help='Enable unit test. Define a precision threshold for which the test passes.')
    parser.add_argument(
        '-s', '--save', dest="save",  type=str, help='Dump desired output to file.')
    parser.add_argument(
        '-p', '--plot', dest="plot",  action="store_true", help='Dump desired output to file.')
    
    
    
    
    return parser.parse_args()

def main():
    lidar_data = SensorData(args=parse_cmdline())
    lidar_data.cartesian_to_spherical(lidar_data.get_raw_scans_cartesian())

    # Test
    lidar_data.spherical_to_cartesian(lidar_data.spherical_data)

    lidar_data.generate_range_img(lidar_data.spherical_data)

    lidar_data.plot_range_img(lidar_data.frame)


if __name__ == "__main__":
    main()

