#!/usr/bin/env python

import numpy as np
import cv2
import load_calibration
from lidar_utils import lidar_utils

date = '2018_03_06'
frame_id = 33
cam_id = '0'
seq = '0001'
DISTORTED = False
MOVE_FORWARD = True

BASE = '../../../dataset/cadcd/'
OUTPUT = 'output/'

if DISTORTED:
  path_type = 'raw'
else:
  path_type = 'labeled'

lidar_path = BASE + date + '/' + seq + "/" + path_type + "/lidar_points/data/" + format(frame_id, '010') + ".bin"
calib_path = BASE + date + "/calib"
img_path =  BASE + date + '/' + seq + "/" + path_type + "/image_0" + cam_id + "/data/" + format(frame_id, '010') + ".png"
annotations_path =  BASE + '/' + date + '/' + seq + "/3d_ann.json"

# load calibration dictionary
calib = load_calibration.load_calibration(calib_path)

# Projection matrix from camera to image frame
T_IMG_CAM = np.eye(4)
T_IMG_CAM[0:3,0:3] = np.array(calib['CAM0' + cam_id]['camera_matrix']['data']).reshape(-1, 3)
T_IMG_CAM = T_IMG_CAM[0:3,0:4] # remove last row

T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM0' + cam_id]))

dist_coeffs = np.array(calib['CAM0' + cam_id]['distortion_coefficients']['data'])

lidar_utils_obj = lidar_utils(T_CAM_LIDAR)

while True:
  print(frame_id)
  # read image
  img = cv2.imread(img_path)

  # Project points onto image
  img = lidar_utils_obj.project_points(img, lidar_path, T_IMG_CAM, T_CAM_LIDAR, dist_coeffs, DISTORTED)
  # cv2.imwrite("test.png", img)

  cv2.imshow('image',img)
  cv2.waitKey(1000)

  if MOVE_FORWARD:
    frame_id += 1
    lidar_path = BASE + date + '/' + seq + "/" + path_type + "/lidar_points/data/" + format(frame_id, '010') + ".bin"
    img_path =  BASE + date + '/' + seq + "/" + path_type + "/image_0" + cam_id + "/data/" + format(frame_id, '010') + ".png"
    img = cv2.imread(img_path)
