#!/usr/bin/python
# Author: Srikanth Malla
# Date: 28 Aug, 2020
# project lidar points in top view image

import matplotlib.pyplot as plt
import numpy as np
import glob
from joblib import Parallel, delayed
import multiprocessing

def lidar_top_view(lidar_file):
  lidar_points = np.load(lidar_file)

  fig = plt.figure(frameon=False)
  DPI = fig.get_dpi()
  fig.set_size_inches(1080.0/float(DPI),1080.0/float(DPI))
  ax = fig.add_subplot(111, xticks=[], yticks=[])
  height = lidar_points[:,2]
  intensity = lidar_points[:,3]


  ######## style 1: combined height and intensity map ########
  # height = np.interp(height, (height.min(), height.max()), (0, 1))
  # # height = np.clip(height, 0, 1)
  # height = np.expand_dims(height, axis=1)
  # intensity = np.expand_dims(intensity, axis=1)
  # zeros = np.zeros_like(height)
  # colors = np.hstack((zeros, height, intensity))
  # ax.scatter(x = lidar_points[:,0], y=lidar_points[:,1], s = 0.01, c=colors)

  ######## style 2: using height to visuzalize ground and obstacles (precog paper style) ########
  gray = [153/255, 153/255, 153/255]
  red = [228/255, 27/255, 28/255]
  ground_points = lidar_points[height<0.7,:] #meters threshold
  non_ground_points = lidar_points[height>0.7,:] #meters threshold
  ax.scatter(x = ground_points[:,0], y=ground_points[:,1], s = 0.01, c=np.tile(gray,(ground_points.shape[0],1)))
  ax.scatter(x = non_ground_points[:,0], y=non_ground_points[:,1], s = 0.01, c=np.tile(red,(non_ground_points.shape[0],1)))

  ### plot adjustments
  ax.set_xlim(-60,60)
  ax.set_ylim(-60,60)
  ax.axis('off')
  fig.subplots_adjust(bottom = 0)
  fig.subplots_adjust(top = 1)
  fig.subplots_adjust(right = 1)
  fig.subplots_adjust(left = 0)
  ax.axis('on')
  # plt.show()
  out_file = lidar_file.replace("pointclouds", "lidar_top")
  out_file = out_file.replace("npy", "png")
  print(out_file)
  fig.savefig(out_file)
  plt.close('all')

if __name__ == '__main__':
  waymo_root = "/home/smalla/datasets/waymo_data/outputs/"
  waymo_files = glob.glob(waymo_root+"*/pointclouds/*.npy")
  num_cores = multiprocessing.cpu_count()
  results = Parallel(n_jobs=num_cores)(delayed(lidar_top_view)(file) for file in waymo_files)
