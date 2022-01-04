import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import matplotlib.patches as patches
from mpl_toolkits import mplot3d
from matplotlib import colors

### Config ###
date = '2019_02_27'
seq = '0009'
frame_id = 0
cam_id = '0'
DISTORTED = True
MOVE_FORWARD = True
DISPLAY_LIDAR = False
DISPLAY_CUBOID_CENTER = False
MIN_CUBOID_DIST = 40.0

BASE = '../../../dataset/cadcd/'
OUTPUT = 'output/'

if DISTORTED:
  path_type = 'raw'
else:
  path_type = 'labeled'

lidar_path = BASE + '/' + date + '/' + seq + "/" + path_type + "/lidar_points/data/" + format(frame_id, '010') + ".bin"
calib_path = BASE + '/' + date + "/calib"
img_path =  BASE + '/' + date + '/' + seq + "/" + path_type + "/image_0" + cam_id + "/data/" + format(frame_id, '010') + ".png"
annotations_path =  BASE + '/' + date + '/' + seq + "/3d_ann.json"
######

def bev(s1,s2,f1,f2,frame_id,lidar_path,annotations_path):
    '''

    :param s1: example 15 (15 meter to the left of the car)
    :param s2: s2 meters from the right of the car
    :param f1: f1 meters from the front of the car
    :param f2: f2 meters from the back of the car
    :param frame_id: the frame number
    :return:
    '''

    #limit the viewing range
    side_range = [s1,s2] #15 meters from either side of the car
    fwd_range = [f1,f2] # 15 m infront of the car

    scan_data = np.fromfile(lidar_path, dtype= np.float32) #numpy from file reads binary file
    #scan_data is a single row of all the lidar values
    # 2D array where each row contains a point [x, y, z, intensity]
    #we covert scan_data to format said above
    lidar = scan_data.reshape((-1, 4))

    lidar_x = lidar[:,0]
    lidar_y = lidar[:,1]
    lidar_z = lidar [:,2]
    intensity = lidar[:, 3]

    snow_points_x = []
    snow_points_y = []
    snow_points_z = []
    snow_points_intensity = []

    removed_points_x = []
    removed_points_y = []
    removed_points_z = []
    removed_points_intensity = []

    for i in range(len(lidar_x)):
        snow_condition = lidar_z[i] > -0.5 and intensity[i] < 0.05
        # intensity[i] < 0.004 and intensity[i] > 0.003

        snow_condition_2 = lidar_z[i] < 0.1 and lidar_z[i] > -0.1 and intensity[i] > 0.0005

        if lidar_x[i] > fwd_range[0] and lidar_x[i] < fwd_range[1] and \
        lidar_y[i] > side_range[0] and lidar_y[i] < side_range[1] \
        and snow_condition: 

            # if snow_condition_2:
            #     removed_points_x.append(lidar_x[i])
            #     removed_points_y.append(lidar_y[i])
            #     removed_points_z.append(lidar_z[i])
            #     removed_points_intensity.append(intensity[i])
            #     continue

            snow_points_x.append(lidar_x[i])
            snow_points_y.append(lidar_y[i])
            snow_points_z.append(lidar_z[i])
            snow_points_intensity.append(intensity[i])
        else:
            removed_points_x.append(lidar_x[i])
            removed_points_y.append(lidar_y[i])
            removed_points_z.append(lidar_z[i])
            removed_points_intensity.append(intensity[i])


    # fig = plt.figure()
    # plt.hist(intensity_values)
    # # plt.bar(lidar_z, color='g')
    # # plt.bar(list(all_azimuths.keys()), all_azimuths.values(), color='g')
    # plt.ylabel("Freq")
    # plt.xlabel("Values")
    # plt.title("Value frequency plot")
    # plt.show()

    for plot_i in range(0, 0):
        # PLOT THE IMAGE
        dpi = 100       # Image resolution

        fig, ax = plt.subplots(figsize=(30,30), dpi=dpi)
        # fig, ax = plt.subplots(figsize=(4000/dpi, 4000/dpi), dpi=dpi)

        ax = plt.axes(projection ='3d')
        ax.scatter(snow_points_x, snow_points_y, snow_points_z, s=0.2, c=snow_points_intensity)
        # ax.scatter(removed_points_x, removed_points_y, removed_points_z, s=0.2, c=removed_points_intensity)

        ax.view_init(0, -180)
        # ax.view_init(0, -90)
        # ax.view_init(0, -90 + 10 * plot_i)

        ax.set_xlabel('$X$', fontsize=20)
        ax.set_ylabel('$Y$', fontsize=20)
        ax.set_zlabel('$Z$', fontsize=20)

        # ax.set_facecolor((0, 0, 0))  # backgrounD is black

        ax.axis('scaled')  # {equal, scaled}

        # ax.xaxis.pane.fill = False
        # ax.yaxis.pane.fill = False
        # ax.zaxis.pane.fill = False

        ax.xaxis.set_visible(True)  # Do not draw axis tick marks
        ax.yaxis.set_visible(True)  # Do not draw axis tick marks
        ax.zaxis.set_visible(True)  # Do not draw axis tick marks
        ax.grid(False)

        plt.xlim([fwd_range[0], fwd_range[1]])  
        plt.ylim([side_range[0], side_range[1]])

        fig.savefig(OUTPUT + "bev_3D_" + str(frame_id) + "_" + str(plot_i) + ".png", dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        # fig.savefig(OUTPUT + "bev_3D_removed_" + str(frame_id) + "_" + str(plot_i) + ".png", dpi=dpi, bbox_inches='tight', pad_inches=0.0)


    distance_distribution = []
    intensity_distribution = []
    n_bins = len(snow_points_x)
    mu = 8.5
    sigma = 3

    fig, ax = plt.subplots(figsize=(8, 4))

    for i in range(len(snow_points_x)):
        distance = np.sqrt(snow_points_x[i]**2 + snow_points_y[i]**2 + snow_points_z[i]**2)
        distance_distribution.append(distance) 
        intensity_distribution.append(snow_points_intensity[i])

    # print(sorted(intensity_distribution))
    

    # plot the cumulative histogram
    # h, xedges, yedges, image = ax.hist2d(distance_distribution, intensity_distribution, bins=n_bins, cmap = plt.cm.jet)
    # n, bins, patches = ax.hist(distance_distribution, bins=n_bins, density=True, histtype='step', cumulative=True, label='Empirical')

    # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
    #     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    # y = y.cumsum()
    # y /= y[-1]

    ax.hist2d(distance_distribution, intensity_distribution, bins=(n_bins/2, 30)) # 2D histogram
    # ax.hist(intensity_distribution, bins=n_bins, histtype='step', density=True, cumulative=True) # 2D histogram
    # ax.plot(bins[1:], np.cumsum(n), 'k--', linewidth=1.5, label='Theoretical')

    ax.grid(True)
    # ax.legend(loc='right')
    ax.set_title('Cummulative Density of Intensities')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('CDF')

    plt.show()

    # print(h)
    # print(len(n))
    # print(sorted(distance_distribution))


    # distance_range = 10
    # for range_i in range(0, int(50/distance_range)):
    #     start = distance_range * range_i
    #     end = start + distance_range

        # for i in range(len(snow_points_x)):
        #     distance = np.sqrt(snow_points_x[i]**2 + snow_points_y[i]**2 + snow_points_z[i]**2)
            # distance_distribution.append(distance) 
        #     # if distance > 15 and distance < 20:
        #     # if snow_points_intensity[i] > 0.0 and distance >=40 and distance < 50:
        #     if distance >= start and distance < end:
        #         intensity_distribution.append(snow_points_intensity[i])

            
        # # print(intensity_per_distance)
        # # for distance, intensities in intensity_per_distance.items():
        # #     fig2 = plt.figure()
        # #     plt.hist(intensities)
        # #     # plt.bar(list(intensity_per_distance.keys()), intensity_per_distance.values(), color='g')
        # #     # plt.bar(list(all_azimuths.keys()), all_azimuths.values(), color='g')
        # #     plt.ylabel("Freq")
        # #     plt.xlabel("Values")
        # #     plt.title("Value frequency plot")
        # #     plt.show()
        # #     break
        # # print("start:", start, "end:", end)
        # # print(intensity_distribution)
        # # print("---")

        # if intensity_distribution:
        #     fig2 = plt.figure()
        #     plt.hist(intensity_distribution) # PDF
        #     plt.hist(intensity_distribution, bins=len(snow_points_x), cumulative=True) # CDF
        #     # plt.hist(intensity_distribution, bins=500)
        #     # plt.bar(list(intensity_per_distance.keys()), intensity_per_distance.values(), color='g')
        #     # plt.bar(list(all_azimuths.keys()), all_azimuths.values(), color='g')
        #     plt.ylabel("Freq")
        #     plt.xlabel("Values")
        #     plt.title("Value frequency plot")
        #     # plt.show()
        #     fig2.savefig(OUTPUT + "hist_intensity_range_" + str(start) + "_" + str(end) + ".png", bbox_inches='tight', pad_inches=0.0)


# bev(-50,50,-50,50,frame_id,lidar_path,annotations_path)
bev(-10,16,-50,50,frame_id,lidar_path,annotations_path)
