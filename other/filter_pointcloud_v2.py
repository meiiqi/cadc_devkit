
# Implementation of DROR in python
# https://github.com/nickcharron/lidar_snow_removal/blob/master/src/DROR.cpp
# Slightly modified due to radiusSearch not being implemented in python-pcl

import numpy as np
import open3d as o3d
import os, math
import glob
from typing import OrderedDict
import json

def dror_filter(input_cloud):
    radius_multiplier_ = 3
    azimuth_angle_ = 0.16 # 0.04
    min_neighbors_ = 3
    k_neighbors_ = min_neighbors_ + 1
    min_search_radius_ = 0.04

    filtered_cloud_list = []
    snow_cloud = []

    # init. kd search tree
    kd_tree = o3d.geometry.KDTreeFlann(input_cloud)

    # input_cloud_size = np.asarray(input_cloud.points).shape[0]

    for point in input_cloud.points:
        x = point[0]
        y = point[1]
        range = math.sqrt(pow(x, 2) + pow(y, 2))
        search_radius_dynamic = radius_multiplier_ * azimuth_angle_ * 3.14159265359 / 180 * range;

        if (search_radius_dynamic < min_search_radius_):
            search_radius_dynamic = min_search_radius_

        [k, idx, _] = kd_tree.search_radius_vector_3d(point, search_radius_dynamic)

        # This point is not snow, add it to the filtered_cloud
        if (k >= min_neighbors_):
            filtered_cloud_list.append(point)
        else:
            snow_cloud.append(point)

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_cloud_list)

    snow_pcd = o3d.geometry.PointCloud()
    snow_pcd.points = o3d.utility.Vector3dVector(snow_cloud)

    return filtered_pcd, snow_pcd

def compute_snow_points(drive, seq, frame):
    if isinstance(seq, int):
        seq = format(seq, '04')

    if isinstance(frame, int):
        frame = format(frame, '010') # format into 10 digits e.g. 1 -> 0000000001

    # Load lidar msg for this frame
    lidar_path = BASE + drive + "/" + seq + "/labeled/lidar_points/data/" + frame + ".bin"
    scan_data = np.fromfile(lidar_path, dtype=np.float32)
    # 2D array where each row contains a point [x, y, z, intensity]
    lidar = scan_data.reshape((-1, 4))[:, 0:3]
    # Convert to Open3D point cloud
    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lidar))

    # Crop the pointcloud to around autnomoose
    cropped_point_cloud = point_cloud.crop(cadc_crop_box)

    # Run DROR
    filtered_point_cloud, snow_point_cloud = dror_filter(cropped_point_cloud)

    if VISUALIZE:
        print("Visualizing original point cloud")
        o3d.visualization.draw_geometries([point_cloud])
        print("Visualizing cropped point cloud")
        o3d.visualization.draw_geometries([cropped_point_cloud])
        print("Visualizing filtered (de-snowed) point cloud")
        o3d.visualization.draw_geometries([filtered_point_cloud])
        print("Visualizing snow points")
        o3d.visualization.draw_geometries([snow_point_cloud])

    return len(snow_point_cloud.points)


def print_snow_points(drive, seq, frame):
    snow_point_count = compute_snow_points(drive, seq, frame)

    # Print number of snow points
    print(drive, seq, frame, ":", snow_point_count)

BASE = '/media/mqtang/DATA/datasets/cadcd/'

VISUALIZE = False

# cadc_crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-4,-4,-2), max_bound=(4,4,10))
cadc_crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-4,-4,-3), max_bound=(4,4,10))

dataset_info = {
    '2018_03_06': [
        '0001','0002',
        '0005','0006','0008','0009','0010',
        '0012','0013','0015','0016','0018'
    ],
    '2018_03_07': [
        '0001','0002','0004','0005','0006','0007'
    ],
    '2019_02_27': [
        '0002','0003','0004','0005','0006','0008','0009','0010',
        '0011','0013','0015','0016','0018','0019','0020',
        '0022','0024','0025','0027','0028','0030',
        '0031','0033','0034','0035','0037','0039','0040',
        '0041','0043','0044','0045','0046','0047','0049','0050',
        '0051','0054','0055','0056','0058','0059',
        '0060','0061','0063','0065','0066','0068','0070',
        '0072','0073','0075','0076','0078','0079',
        '0080','0082'
    ]
}

# total_frame_count = 0
# for drive, sequences in dataset_info.items():
#     for seq in sequences:
#         frames = glob.glob(BASE + drive + '/' + seq + "/labeled/lidar_points/data/*.bin")
#         total_frame_count += len(frames)

# print(total_frame_count) # 7000


# Compute snow points for dataset
drive_data = OrderedDict()
# Structure of `drive_data`
# {
#     drive0: {
#         seq0: [
#             frame0,
#             frame1,
#             frame2
#         ]
#     }
# }
# E.g.,
# {
#     "2018_03_06": {
#         "0001": [
#             12,
#             20,
#             24
#         ]
#     }
# }
for drive, sequences in dataset_info.items():
    sequence_data = OrderedDict()
    for seq in sequences:
        frames = glob.glob(BASE + drive + '/' + seq + "/labeled/lidar_points/data/*.bin")
        frame_data = [0] * len(frames)
        for frame in sorted(frames):
            frame = os.path.basename(frame).strip(".bin")
            frame_data[int(frame)] = compute_snow_points(drive, seq, frame)

        sequence_data[seq] = frame_data

    drive_data[drive] = sequence_data

json_object = json.dumps(drive_data, indent=4)

# Writing to sample.json
with open("dror_results.json", "w") as f:
    f.write(json_object)

# print_snow_points(drive='2019_02_27', seq='0070', frame=0)

# Low 74
# Medium 81
# High 80
# High example 68