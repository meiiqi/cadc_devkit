import pickle
import os 
import numpy as np
import matplotlib.pyplot as plt


for frame_id in range(2, 10):
    print("frame_id:",frame_id)

    input_dir = 'input'
    filename = "packets_"+ str(frame_id) + ".pkl"

    # Data packet structure
    packet_size = 1206
    num_datablocks = 12
    num_channels = 32
    datablock_size = 100
    azimuth_size = 2
    channel_size = 3 
    flag_size = 2

    # Load laser info
    laser_table = np.genfromtxt('laser_table.csv', dtype=float, delimiter=',', names=True) 

    # Load packets from one frame
    is_binary = True

    frame_raw_data = None
    file = open(os.path.join(input_dir, filename), 'rb')
    if is_binary:
        # data = pickle.load(file, encoding="bytes")
        data = pickle.load(file, encoding="bytes")[151 * frame_id:]
        frame_raw_data = np.ndarray((len(data), packet_size))  # (151, 1206)

        for i, packet in enumerate(data):
            byte_array_list = list(bytearray(packet))
            
            if len(byte_array_list) != packet_size:
                raise Exception("Packet Size not as expected")

            frame_raw_data[i] = byte_array_list

    else:
        frame_raw_data = np.array(pickle.load(file)) # (151, 1206)

        if frame_raw_data.shape[1] != packet_size:
            raise Exception("Packet Size not as expected")

    print(frame_raw_data.shape)


    # Init 
    datablocks = np.ndarray((num_datablocks * frame_raw_data.shape[0], datablock_size), dtype='uint16') # Number of packets in frame x 12
    azimuth_values = []
    frame_parsed_data = []

    def get_spacing(values):
        spacing = []
        for i in range(len(values)):
            diff = values[(i+1) % len(values)] - values[i]
            if abs(diff) > 35900:
                diff = diff + 35900

            # Don't include outlier data points
            if abs(diff) < 40:
                spacing.append(diff/100.0)    

        return spacing

    # Little Endian bytes
    def pack_2_bytes(bytes : list):
        return bytes[1] << 8 | bytes[0]

    def get_elevation_index(input_angle):
        # print("elevation angle [deg]:", input_angle)
        # print("row idx", int(laser_table['ROW_IDX'][np.abs(np.array(laser_table['ELEVATION_ANGLE']) - input_angle).argmin()]))
        return int(laser_table['ROW_IDX'][np.abs(np.array(laser_table['ELEVATION_ANGLE']) - input_angle).argmin()])

            
    def get_azimuth_resolution(rpm, firing_cycle_us) -> float:
        return rpm * 360 / 60.0 * firing_cycle_us * 10**(-6)

    def get_azimuth_count(az_resolution) -> int:
        return int(np.round(360.0 / az_resolution))

    rpm = 600
    firing_cycle_us = 55.296
    azimuth_resolution = get_azimuth_resolution(rpm, firing_cycle_us)
    azimuth_count = get_azimuth_count(azimuth_resolution)
    updated_azimuth_resolution = 360.0 / azimuth_count

    def get_azimuth_index(input_angle):
        azimuth = (input_angle + 360 ) % 360 # Convert to positive angles
        return int(np.floor(azimuth / updated_azimuth_resolution))


    # Every file contains a list of packets. Each packet is of size 1206.
    # Each packet contains 12 datablocks.
    # Each datablock contains 32 channels.

    # Get list of datablocks
    for ipacket, packet in enumerate(frame_raw_data):
        for idatablock in range(num_datablocks):
            datablocks[ipacket * num_datablocks + idatablock] = packet[idatablock * datablock_size : idatablock * datablock_size + datablock_size]

    # Get Azimuth and Channel data for each datablock
    for idatablock, datablock in enumerate(datablocks):
        next_datablock = datablocks[(idatablock + 1) % len(datablocks)]
        idx = 0
        # print("Datablock:", datablock[idx:idx + 100])

        # Flag
        # print("Flag", datablock[idx:idx + flag_size])
        idx += flag_size

        # Azimuth
        # print("idx", idx)
        azimuth = pack_2_bytes(datablock[idx:idx + azimuth_size]) / 100.0
        next_azimuth = pack_2_bytes(next_datablock[idx:idx + azimuth_size]) / 100.0
        azimuth_gap = next_azimuth - azimuth

        # print("Azimuth", azimuth)
        # print("Next azimuth", next_azimuth)
        # print("Azimuth gap", azimuth_gap)

        if azimuth > 35999:
            print("Azimuth out of range", azimuth)
            print(datablock)
            raise Exception("Azimuth out of range")

        azimuth_values.append(azimuth)
        idx += azimuth_size

        # Channels
        channels = np.ndarray((32, 4))
        for ichannel in range(num_channels):
            channel_data = datablock[idx:idx + channel_size]
            # print("channel_data:", channel_data )
            
            # elevation row index (index of the elevation row in range image) [0-31]
            irow = laser_table['ROW_IDX'].astype(int)[ichannel]
            # print("ichannel", ichannel)
            # print("irow", irow)

            elevation = laser_table['ELEVATION_ANGLE'][ichannel] # [deg]
            # print("elevation [deg]", elevation)

            # Corrected azimuth wrt current azimuth gap
            azimuth_corrected = azimuth + laser_table['AZIMUTH_OFFSET'][ichannel] \
                + azimuth_gap * (laser_table['TIMING_OFFSET_US'][ichannel] / 55.296) # [deg]
            # print("azimuth_corrected", azimuth_corrected)

            # Distance
            distance = float(str(channel_data[1] * 4) + '.' + str(channel_data[0] * 4))
            # print("distance: ", distance)

            # Reflectivity
            reflectivity = channel_data[2]
            # print("reflectivity: ", reflectivity)
            if (reflectivity > 255 or reflectivity < 0):
                raise Exception
            
            # print("inserting: ", irow, [elevation, azimuth_corrected, distance, reflectivity])
            channels[irow] = [elevation, azimuth_corrected, distance, reflectivity]

            # print("channels: ", channels)

            # print("Channel", ichannel, datablock[idx:idx + channel_size])
            # print("i: ", ichannel, "data:", [distance, reflectivity])
            idx += channel_size

        # print("channels: ", channels)
        # break

        # print("channels len: ", len(channels)) # 32
        frame_parsed_data.append(channels)


    # print(azimuth_values)
    # print("Number of azimuth values:", len(azimuth_values)) # 1812
    # print("Number of datablocks:",len(datablocks)) # 1812
    frame_parsed_data = np.asarray(frame_parsed_data)
    frame_parsed_data = np.transpose(frame_parsed_data, (1, 0, 2)) # transpose height and width, without changing depth
    print("frame_parsed_data.shape:", frame_parsed_data.shape) # (32, 1812, 3)

    h, w, d = frame_parsed_data.shape
    print("h, w, d:", h, w, d)

    range_image_elevations = frame_parsed_data[:, :, 0]
    range_image_azimuths = frame_parsed_data[:, :, 1]
    range_image_distance = frame_parsed_data[:, :, 2]
    range_image_reflectivity = frame_parsed_data[:, :, 3]

    # print("range_image_elevations.shape:", range_image_elevations.shape) # Get the distance range image (32, 1812)
    # print("range_image_azimuths.shape:", range_image_azimuths.shape) # (32, 1812)
    # print("range_image_distance.shape:", range_image_distance.shape) # Get the distance range image (32, 1812)
    # print("range_image_reflectivity.shape:", range_image_reflectivity.shape) # Get the distance range image (32, 1812)
    # print("range_image_elevations[:, 0]:", range_image_elevations[:, 0]) # Get a column
    # print("range_image_azimuths[:, 0]:", range_image_azimuths[:, 0]) # Get a column
    # print("range_image_distance[:, 0]:", range_image_distance[:, 0]) # Get a column
    # print("range_image_reflectivity[:, 0]:", range_image_reflectivity[:, 0]) # Get a column

    # parse table to create point cloud 
    point_cloud = []
    testlist = []
    for i in range(h):
        for j in range(w):
            # print(i, j, frame_parsed_data[i, j])
            datapoint = frame_parsed_data[i, j]

            elevation, azimuth_corrected, distance, reflectivity = datapoint
            # print("azimuth_corrected:",azimuth_corrected)
            # print("elevation:", elevation)
            # print("distance:", distance)

            azimuth_corrected_rad = np.radians(azimuth_corrected)
            # print("azimuth_corrected_rad:", azimuth_corrected_rad)

            elevation_rad = np.radians(elevation)
            # print("elevation_rad:", elevation_rad)

            point_x = distance * np.cos(elevation_rad) * np.sin(azimuth_corrected_rad)
            point_y = distance * np.cos(elevation_rad) * np.cos(azimuth_corrected_rad)
            point_z = distance * np.sin(elevation_rad)

            

            # if point_x > 0.0 or point_y > 0.0 or point_z > 0.0:
            if distance > 0.0:
                # print("azimuth_corrected (deg)", azimuth_corrected) # [deg]
                # print("elevation (deg)", i) # [deg]
                # print("azimuth_corrected (rad)", azimuth_corrected_rad) # [rad]
                # print("elevation (rad)", elevation_rad) # [rad]
                # print("distance", distance) # [m]
                # print("reflectivity", reflectivity) # 15.0

                # print("point_x", point_x) # [m]
                # print("point_y", point_y) # [m]
                # print("point_z", point_z) # [m]
                # print("====")
                point_cloud.append([point_y, -point_x, point_z, reflectivity]) # x_coord = y, y_coord=-x, z = z (convert to ROS coord system)

    print("len(point_cloud):",len(point_cloud))

    # convert back point cloud to spherical coordinates
    overlap_count = 0
    all_azimuths = {}
    all_elevations = {}
    range_image_points = []
    range_image = np.zeros((h, w, 2))
    overlaps_image = np.zeros((h, w, 1))
    for point in point_cloud:

        # print(point)
            
        point_x, point_y, point_z, reflectivity = point

        point_azimuth = np.arctan2(point_y, point_x)
        point_azimuth_deg = np.degrees(np.arctan2(point_y, point_x))
        point_range = np.sqrt(point_x**2 + point_y**2 + point_z**2)    
        point_elevation = np.arcsin(point_z / point_range)
        point_elevation_deg = np.degrees(np.arcsin(point_z / point_range))

        # print("point_azimuth", point_azimuth) # [rad]
        # print("point_azimuth_deg", point_azimuth_deg) # [deg]
        # print("point_range", point_range) # [m]
        # print("point_elevation", point_elevation) # [rad]
        # print("point_elevation_deg", point_elevation_deg) # [deg]

        range_image_points.append([point_azimuth, point_elevation, point_range, reflectivity])

        elevation_index = get_elevation_index(point_elevation_deg) # [0-31]
        azimuth_index = get_azimuth_index(point_azimuth_deg) # [0 - 1812]

        # print("elevation_index", elevation_index)
        # print("azimuth_index", azimuth_index)

        testlist.append(azimuth_index)
        all_elevations[elevation_index] = all_elevations.get(elevation_index,0) + 1
        all_azimuths[azimuth_index] = all_azimuths.get(azimuth_index,0) + 1

        if range_image[elevation_index, azimuth_index, :][0] != 0 or range_image[elevation_index, azimuth_index, :][1] != 0:
            overlaps_image[elevation_index, azimuth_index] += 1
            overlap_count += 1

        range_image[elevation_index, azimuth_index, :] = [point_range, reflectivity]


    # PLOT HISTORGRAM
    # fig = plt.figure()
    # plt.hist(testlist)
    # plt.bar(list(all_elevations.keys()), all_elevations.values(), color='g')
    # plt.bar(list(all_azimuths.keys()), all_azimuths.values(), color='g')
    # plt.ylabel("Freq")
    # plt.xlabel("Values")
    # plt.title("Value frequency plot")
    # plt.show()


    # PLOT RANGE IMAGE
    # fig = plt.figure()
    # plt.imshow(overlaps_image[: ,:, 0])
    # plt.ylabel("Elevation Values")
    # plt.xlabel("Azimuth Values")
    # plt.title("Range image - Distance")
    # plt.show()

    print("Overlap count:", overlap_count)

    