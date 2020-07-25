#!/usr/bin/python
# Author: Srikanth Malla

import cv2
import glob
import os
from os import path
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
from progressbar import ProgressBar
import multiprocessing


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

input_dir = "/home/smalla/waymo_data/outputs/"
scenes = glob.glob(input_dir+"*")
scenes.sort()
# global_counter = 0

height = 1080
width = 1080

def create_dir(folder):
	if not path.exists(folder):
		os.mkdir(folder)

def top_view(bb, color, ax):
	# for i in range(np.shape(boxes)[0]):
	# 	bb=boxes[i,:]
	m = [bb[0],bb[1],bb[0]+bb[2],bb[1],bb[0]+bb[2],bb[1]+bb[3],bb[0],bb[1]+bb[3]]
	m = np.asarray(m)
	m = m.reshape(4,2)
	m[0,0]-=bb[2]/2
	m[1,0]-=bb[2]/2
	m[2,0]-=bb[2]/2
	m[3,0]-=bb[2]/2
	m[0,1]-=bb[3]/2
	m[1,1]-=bb[3]/2
	m[2,1]-=bb[3]/2
	m[3,1]-=bb[3]/2
	t = mpl.transforms.Affine2D().rotate_deg_around((m[0,0]+m[1,0])/2,(m[0,1]+m[3,1])/2,np.rad2deg(-bb[4]))+ax.transData
	rect = patches.Polygon([m[0,0:2], m[1,0:2], m[2,0:2],m[3,0:2]],fill=False,edgecolor=color)
	rect.set_transform(t)
	ax.add_patch(rect)

def draw_box(bb, img, color):
	#takes bbox, image and color
	img = cv2.rectangle(img,(int(bb[0]-bb[2]/2),int(bb[1]-bb[3]/2)),(int(bb[0]+bb[2]/2),int(bb[1]+bb[3]/2)), color, 5)
	return img

def draw_labels_on_img(top_img, labels_tag, global_track_ids, cam_image, colors):
	labels_file = top_img.replace("lidar_top",labels_tag)
	labels_file = labels_file.replace("png","txt.npy")
	if path.isfile(labels_file):
		# label.type, label.id, label.box.center_x, label.box.center_y, label.box.length, label.box.width
		cam_labels = np.load(labels_file)
		cam_labels = np.array(cam_labels)
		if cam_labels.size ==0:
			return cam_image
		track_ids = cam_labels[:,1]
		# print(labels_tag," ",track_ids)
		cx = np.array(cam_labels[:,2]).astype(float)
		cy = np.array(cam_labels[:,3]).astype(float)
		l = np.array(cam_labels[:,4]).astype(float)
		w = np.array(cam_labels[:,5]).astype(float)

	for ind2, track_id in enumerate(track_ids):
		if track_id not in global_track_ids:
			global_track_ids.append(track_id)
		ind = global_track_ids.index(track_id)%100
		bb = [cx[ind2], cy[ind2], l[ind2], w[ind2]]
		color = tuple([int(colors[ind,2]*255), int(colors[ind,1]*255), int(colors[ind,0]*255)])
		cam_image = draw_box(bb, cam_image, color)
	return cam_image

def main(scene):
	print(scene)
	# out = cv2.VideoWriter('merged_video.avi',cv2.VideoWriter_fourcc(*'MPEG'), 10, (width, height))
	# global global_counter
	# scene_count
	colors = np.random.rand(100, 3)

	lidar_top_images = glob.glob(scene+"/lidar_top/*.png")
	create_dir(scene+"/merged_image")
	lidar_top_images.sort()
	local_counter = 0
	camera_labels = False
	if "_with_camera_labels" in scene:
		camera_labels = True

	prev_ego_motion = None

	global_track_ids = []
	for top_img in lidar_top_images:
		# fig init
		fig = plt.figure(frameon=False)
		DPI = fig.get_dpi()
		fig.set_size_inches(1080.0/float(DPI),1080.0/float(DPI))
		ax = fig.add_subplot(111, xticks=[], yticks=[])

		merged = cv2.imread(top_img)

		# ego-motion
		ego_motion_file = top_img.replace("lidar_top","ego_motion")
		ego_motion_file = ego_motion_file.replace("png","npy")
		if path.isfile(ego_motion_file):
			ego_motion = np.load(ego_motion_file)
			if prev_ego_motion is None:
				lin_speed = 0
				prev_ego_motion = ego_motion
			else:
				lin_speed = ((ego_motion[:,3]-prev_ego_motion[:,3])/0.1)
				lin_speed*=2.23694 # convert m/s to miles/hr
				lin_speed = np.linalg.norm(lin_speed) # scalar
				prev_ego_motion = ego_motion

		# lidar labels
		lidar_labels_file = top_img.replace("lidar_top","labels_pc")
		lidar_labels_file = lidar_labels_file.replace("png","npy")
		# print(lidar_labels_file)
		if path.isfile(lidar_labels_file):
			# [label.type, label.id, label.box.center_x, label.box.center_y, label.box.center_z, label.box.length, label.box.width, label.box.height, label.box.heading, label.metadata.speed_x, label.metadata.speed_y, label.metadata.accel_x, label.metadata.accel_y]
			lidar_labels = np.load(lidar_labels_file)
			lidar_labels = np.array(lidar_labels)
			scale = 1080/120 # scale with 1080/120, pixels/m and offset to center
			label = lidar_labels[:,0]
			track_ids = lidar_labels[:,1]
			# print("lidar: ",track_ids)
			cx = np.array(lidar_labels[:,2]).astype(float)*scale+(1080/2)
			cy = -np.array(lidar_labels[:,3]).astype(float)*scale+(1080/2)
			# cz = np.array(lidar_labels[:,4]).astype(float)*scale+(1080/2)
			l = np.array(lidar_labels[:,5]).astype(float)*scale
			w = np.array(lidar_labels[:,6]).astype(float)*scale
			# h = lidar_labels[:,7]*scale
			yaw = np.array(lidar_labels[:,8]).astype(float)
			# print(cx)

		img_resize_w = int(width/3)
		img_resize_h = int(height/4)

		# front left image
		if path.isfile(top_img.replace("lidar_top","FRONT_LEFT")):
			cam_front_left_image = cv2.imread(top_img.replace("lidar_top","FRONT_LEFT"))
		x_offset=0; y_offset=0;
		cam_front_left_image = draw_labels_on_img(top_img, "labels_FRONT_LEFT", global_track_ids, cam_front_left_image, colors)
		cam_front_left_image = cv2.resize(cam_front_left_image, (img_resize_w, img_resize_h))
		merged[y_offset:y_offset+cam_front_left_image.shape[0], x_offset:x_offset+cam_front_left_image.shape[1]] = cam_front_left_image

		# front image
		if path.isfile(top_img.replace("lidar_top","FRONT")):
			cam_front_image = cv2.imread(top_img.replace("lidar_top","FRONT"))
		x_offset=int(width/3); y_offset=0;
		cam_front_image = draw_labels_on_img(top_img, "labels_FRONT", global_track_ids, cam_front_image, colors)
		cam_front_image = cv2.resize(cam_front_image, (img_resize_w, img_resize_h))
		merged[y_offset:y_offset+cam_front_image.shape[0], x_offset:x_offset+cam_front_image.shape[1]] = cam_front_image

		#front right
		if path.isfile(top_img.replace("lidar_top","FRONT_RIGHT")):
			cam_front_right_image = cv2.imread(top_img.replace("lidar_top","FRONT_RIGHT"))
		x_offset=int(2*width/3); y_offset=0;
		cam_front_right_image = draw_labels_on_img(top_img, "labels_FRONT_RIGHT", global_track_ids, cam_front_right_image, colors)
		cam_front_right_image = cv2.resize(cam_front_right_image, (img_resize_w, img_resize_h))
		merged[y_offset:y_offset+cam_front_right_image.shape[0], x_offset:x_offset+cam_front_right_image.shape[1]] = cam_front_right_image

		img_resize_w = int(width/3)
		img_resize_h = int(height/4)
		# side left
		if path.isfile(top_img.replace("lidar_top","SIDE_LEFT")):
			cam_side_left_image = cv2.imread(top_img.replace("lidar_top","SIDE_LEFT"))
		x_offset=0; y_offset=int(3*height/4);
		cam_side_left_image = draw_labels_on_img(top_img, "labels_SIDE_LEFT", global_track_ids, cam_side_left_image, colors)
		cam_side_left_image = cv2.resize(cam_side_left_image, (img_resize_w, img_resize_h))
		merged[y_offset:y_offset+cam_side_left_image.shape[0], x_offset:x_offset+cam_side_left_image.shape[1]] = cam_side_left_image

		# side right
		if path.isfile(top_img.replace("lidar_top","SIDE_RIGHT")):
			cam_side_right_image = cv2.imread(top_img.replace("lidar_top","SIDE_RIGHT"))
		x_offset=int(2*width/3); y_offset=int(3*height/4);
		cam_side_right_image = draw_labels_on_img(top_img, "labels_SIDE_RIGHT", global_track_ids, cam_side_right_image, colors)
		cam_side_right_image = cv2.resize(cam_side_right_image, (img_resize_w, img_resize_h))
		merged[y_offset:y_offset+cam_side_right_image.shape[0], x_offset:x_offset+cam_side_right_image.shape[1]] = cam_side_right_image

		file = scene.split("/")[-1]
		file = file.replace("segment-","")
		file = file.replace("_with_camera_labels","")
		offset = 20
		y_start = 860
		merged = cv2.putText(merged, "segment:", (int(width/3), y_start), cv2.FONT_HERSHEY_SIMPLEX,
								 0.5, (0, 0, 0), 1, cv2.LINE_AA)

		y_start+=offset
		merged = cv2.putText(merged, file, (int(width/3), y_start), cv2.FONT_HERSHEY_SIMPLEX,
								 0.5, (0, 0, 0), 1, cv2.LINE_AA)

		y_start+=2*offset
		text = "ego speed: "+str(int(lin_speed)).zfill(3)+" mph"
		merged = cv2.putText(merged, text, (int(width/3), y_start), cv2.FONT_HERSHEY_SIMPLEX,
								 0.5, (0, 0, 255), 1, cv2.LINE_AA)

		# y_start+=offset
		# text = "scene count: "+str(scene_count).zfill(6)
		# merged = cv2.putText(merged, text, (int(width/3), y_start), cv2.FONT_HERSHEY_SIMPLEX,
		# 						 0.5, (0, 0, 0), 1, cv2.LINE_AA)


		y_start+=offset
		text = "local frame count: "+str(local_counter).zfill(6)
		merged = cv2.putText(merged, text, (int(width/3), y_start), cv2.FONT_HERSHEY_SIMPLEX,
								 0.5, (0, 0, 0), 1, cv2.LINE_AA)

		# y_start+=offset
		# text = "global frame count: "+str(global_counter).zfill(6)
		# merged = cv2.putText(merged, text, (int(width/3), y_start), cv2.FONT_HERSHEY_SIMPLEX,
		# 						 0.5, (0, 0, 0), 1, cv2.LINE_AA)

		y_start+=offset
		text = "camera labels: "+str(camera_labels)
		merged = cv2.putText(merged, text, (int(width/3), y_start), cv2.FONT_HERSHEY_SIMPLEX,
								 0.5, (0, 0, 0), 1, cv2.LINE_AA)

		# global_counter+=1
		local_counter+=1
		# write to video
		# out.write(merged)
		# save output
		# cv2.imwrite(top_img.replace("lidar_top","merged_image"), merged)
		# visualize
		# cv2.imshow("merged", merged)
		# cv2.waitKey(100)
		merged = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)

		ax.imshow(merged)
		for ind2, track_id in enumerate(track_ids):
			if track_id not in global_track_ids:
				global_track_ids.append(track_id)
			ind = global_track_ids.index(track_id)%100
			bb = [cx[ind2], cy[ind2], l[ind2], w[ind2], yaw[ind2]]
			top_view(bb,tuple(colors[ind,:]), ax)

		ax.quiver(cx, cy, 20*np.cos(yaw), 20*np.sin(yaw), units='xy' ,scale=1)
		ax.plot(cx, cy, 'ro', markersize=3)

		ax.set_xlim(0,1080)
		ax.set_ylim(1080,0)
		ax.axis('off')

		fig.subplots_adjust(bottom = 0)
		fig.subplots_adjust(top = 1)
		fig.subplots_adjust(right = 1)
		fig.subplots_adjust(left = 0)
		fig.savefig(top_img.replace("lidar_top","merged_image"))
		# plt.show()


		fig.clear()
		plt.close()
	# out.release()
	# cv2.destroyAllWindows()

if __name__ == '__main__':
	parallel = True

	## sequential
	if not parallel:
		for scene_count, scene in enumerate(tqdm(scenes)):
			main(scene)

	## parallel
	if parallel:
		num_cores = multiprocessing.cpu_count()
		results = Parallel(n_jobs=num_cores)(delayed(main)(scene) for scene in scenes)
