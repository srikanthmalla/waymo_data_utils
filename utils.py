# original:
# Modified: Srikanth Malla
# Date: 11 July 2020
# ----
# details:
# 7 functions
# ego_motion (ego car motion)
# labels_pc (pointcloud 3d box labels)
# labels_camera (camera labels for each camera)
# visualize_cameras (all 5 camera data)
# lidar_front_view (diff channels like intensity, depth, ..)
# lidar_projection_on_camera (lidar points projected on camera)
# lidar_top_view (top view of lidar)

import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_dir(folder):
	if not os.path.exists(folder):
		os.mkdir(folder)

##### ego-motion ####
def ego_motion(frame, out_file, save=True):
	frame_pose = np.reshape(np.array(frame.pose.transform), [4, 4])
	np.save(out_file, frame_pose)


##### laser labels ###
def labels_pc(frame, range_images, camera_projections, range_image_top_pose, out_file, indx, save=False):
	def extract_labels(laser_labels, out_file):
		labels = []
		for label in laser_labels:
			# print(label, label.type)
			labels.append([label.type, label.id, label.box.center_x, label.box.center_y, label.box.center_z, label.box.length, label.box.width, label.box.height, label.box.heading, label.metadata.speed_x, label.metadata.speed_y, label.metadata.accel_x, label.metadata.accel_y])
		np.save(out_file, labels)

	extract_labels(frame.laser_labels, out_file)

##### camera labels ###
def labels_camera(frame, range_images, camera_projections, range_image_top_pose, out_folder, indx, save=False):
	def extract_labels(camera_image, camera_labels, out_file):
		labels = []
		for c_labels in camera_labels:
			if c_labels.name != camera_image.name:
				continue
			for label in c_labels.labels:
				labels.append([label.type, label.id, label.box.center_x, label.box.center_y, label.box.length, label.box.width])
		np.save(out_file, labels)
		# f = open(out_file, 'w+')
		# print("-"*30)
		# print(np.array(labels))
	for index, image in enumerate(frame.images):
		# print(index)
		out_file1 = out_folder+"/labels_"+str(open_dataset.CameraName.Name.Name(image.name))
		create_dir(out_file1)
		out_file = out_file1+"/"+str(indx).zfill(6)+".txt"
		if not os.path.isfile(out_file):
		  extract_labels(image, frame.camera_labels, out_file)

####### visualize camera images ######
def visualize_cameras(frame, range_images, camera_projections, range_image_top_pose, out_folder, indx, show=False, save=False):
	def show_camera_image(camera_image, camera_labels, layout, out_file, cmap=None):
		"""Show a camera image and the given camera labels."""
		# if show:
		# 	ax = plt.subplot(*layout)
			# Draw the camera labels.
			# for camera_labels in frame.camera_labels:
			# 	# Ignore camera labels that do not correspond to this camera.
			# 	if camera_labels.name != camera_image.name:
			# 		continue

			# 	# Iterate over the individual labels.
			# 	for label in camera_labels.labels:
			# 		# Draw the object bounding box.
			# 		ax.add_patch(patches.Rectangle(
			# 			xy=(label.box.center_x - 0.5 * label.box.length,
			# 					label.box.center_y - 0.5 * label.box.width),
			# 			width=label.box.length,
			# 			height=label.box.width,
			# 			linewidth=1,
			# 			edgecolor='red',
			# 			facecolor='none'))

		# Show the camera image.
		image = tf.image.decode_jpeg(camera_image.image)
		if show:
			plt.imshow(image, cmap=cmap)
			plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
			plt.grid(False)
			plt.axis('off')
		if save:
			# png.from_array(image, 'L').save(out_file)
			matplotlib.image.imsave(out_file, image.numpy())
	if show:
		plt.figure(figsize=(25, 20))

	for index, image in enumerate(frame.images):
		# print(index)
		out_file1 = out_folder+"/"+str(open_dataset.CameraName.Name.Name(image.name))
		create_dir(out_file1)
		out_file = out_file1+"/"+str(indx).zfill(6)+".png"
		if not os.path.isfile(out_file):
			show_camera_image(image, frame.camera_labels, [2, 3, index+1], out_file)
	if show:
		plt.show()

###### visualize range images #####
def lidar_front_view(frame, range_images, camera_projections, range_image_top_pose):
	plt.figure(figsize=(64, 20))
	def plot_range_image_helper(data, name, layout, vmin = 0, vmax=1, cmap='gray'):
		"""Plots range image.

		Args:
			data: range image data
			name: the image title
			layout: plt layout
			vmin: minimum value of the passed data
			vmax: maximum value of the passed data
			cmap: color map
		"""
		plt.subplot(*layout)
		plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
		plt.title(name)
		plt.grid(False)
		plt.axis('off')

	def get_range_image(laser_name, return_index):
		"""Returns range image given a laser name and its return index."""
		return range_images[laser_name][return_index]

	def show_range_image(range_image, layout_index_start = 1):
		"""Shows range image.

		Args:
			range_image: the range image data from a given lidar of type MatrixFloat.
			layout_index_start: layout offset
		"""
		range_image_tensor = tf.convert_to_tensor(range_image.data)
		range_image_tensor = tf.reshape(range_image_tensor, range_image.shape.dims)
		lidar_image_mask = tf.greater_equal(range_image_tensor, 0)
		range_image_tensor = tf.where(lidar_image_mask, range_image_tensor,
																	tf.ones_like(range_image_tensor) * 1e10)
		# print(range_image_tensor.shape)
		range_image_range = range_image_tensor[...,0]
		range_image_intensity = range_image_tensor[...,1]
		range_image_elongation = range_image_tensor[...,2]
		plot_range_image_helper(range_image_range.numpy(), 'range',
										 [8, 1, layout_index_start], vmax=75, cmap='gray')
		plot_range_image_helper(range_image_intensity.numpy(), 'intensity',
										 [8, 1, layout_index_start + 1], vmax=1.5, cmap='gray')
		plot_range_image_helper(range_image_elongation.numpy(), 'elongation',
										 [8, 1, layout_index_start + 2], vmax=1.5, cmap='gray')
	frame.lasers.sort(key=lambda laser: laser.name)
	show_range_image(get_range_image(open_dataset.LaserName.TOP, 0), 1)
	show_range_image(get_range_image(open_dataset.LaserName.TOP, 1), 4)
	plt.show()

def lidar_projection_on_camera(frame, range_images, camera_projections, range_image_top_pose):
	points, cp_points = frame_utils.convert_range_image_to_point_cloud(
		frame,
		range_images,
		camera_projections,
		range_image_top_pose)
	points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
			frame,
			range_images,
			camera_projections,
			range_image_top_pose,
			ri_index=1)

	# 3d points in vehicle frame.
	points_all = np.concatenate(points, axis=0)
	points_all_ri2 = np.concatenate(points_ri2, axis=0)
	# camera projection corresponding to each point.
	cp_points_all = np.concatenate(cp_points, axis=0)
	cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

	images = sorted(frame.images, key=lambda i:i.name)
	cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
	cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

	# The distance between lidar points and vehicle frame origin.
	points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
	cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

	mask = tf.equal(cp_points_all_tensor[..., 0], images[0].name)

	cp_points_all_tensor = tf.cast(tf.gather_nd(
			cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
	points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))

	projected_points_all_from_raw_data = tf.concat(
			[cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()
	def rgba(r):
		"""Generates a color based on range.

		Args:
			r: the range value of a given point.
		Returns:
			The color for a given range
		"""
		c = plt.get_cmap('jet')((r % 20.0) / 20.0)
		c = list(c)
		c[-1] = 0.5  # alpha
		return c

	def plot_image(camera_image):
		"""Plot a cmaera image."""
		plt.figure(figsize=(20, 12))
		plt.imshow(tf.image.decode_jpeg(camera_image.image))
		plt.grid("off")

	def plot_points_on_image(projected_points, camera_image, rgba_func,
													 point_size=5.0):
		"""Plots points on a camera image.

		Args:
			projected_points: [N, 3] numpy array. The inner dims are
				[camera_x, camera_y, range].
			camera_image: jpeg encoded camera image.
			rgba_func: a function that generates a color from a range value.
			point_size: the point size.

		"""
		plot_image(camera_image)

		xs = []
		ys = []
		colors = []

		for point in projected_points:
			xs.append(point[0])  # width, col
			ys.append(point[1])  # height, row
			colors.append(rgba_func(point[2]))

		plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")

	plot_points_on_image(projected_points_all_from_raw_data,
										 images[0], rgba, point_size=5.0)
	plt.show()

def lidar_top_view(frame, range_images, camera_projections, range_image_top_pose,out_file, show=False, save=False):
	points, cp_points = frame_utils.convert_range_image_to_point_cloud(
		frame,
		range_images,
		camera_projections,
		range_image_top_pose)
	# top lidar, 1st one
	# lidar_points = points[0]

	# merged lidars
	lidar_points = np.concatenate(points, axis=0)

	fig = plt.figure(frameon=False)
	DPI = fig.get_dpi()
	fig.set_size_inches(1080.0/float(DPI),1080.0/float(DPI))
	ax = fig.add_subplot(111, xticks=[], yticks=[])
	# height map
	# ax.scatter(x = lidar_points[:,0], y=lidar_points[:,1], s = 0.01, c=lidar_points[:,2], cmap='gray')

	# # intensity map
	# ax.scatter(x = lidar_points[:,0], y=lidar_points[:,1], s = 0.01, c=lidar_points[:,3], cmap='YlOrRd')

	# combined height and intensity map
	height = lidar_points[:,2]
	height = np.interp(height, (height.min(), height.max()), (0, 1))
	# height = np.clip(height, 0, 1)

	height = np.expand_dims(height, axis=1)
	intensity = lidar_points[:,3]
	intensity = np.expand_dims(intensity, axis=1)
	zeros = np.zeros_like(height)
	# ones = np.ones_like(height)
	colors = np.hstack((zeros, height, intensity))
	# print(np.min(height), np.max(intensity))
	# print(np.shape(colors), lidar_points[:,0].shape, lidar_points[:,1].shape)
	ax.scatter(x = lidar_points[:,0], y=lidar_points[:,1], s = 0.01, c=colors)

	ax.set_xlim(-60,60)
	ax.set_ylim(-60,60)
	ax.axis('off')
	fig.subplots_adjust(bottom = 0)
	fig.subplots_adjust(top = 1)
	fig.subplots_adjust(right = 1)
	fig.subplots_adjust(left = 0)
	ax.axis('on')
	if show:
		plt.show()
	if save:
		fig.savefig(out_file)
		plt.close('all')

def lidar_data(frame, range_images, camera_projections, range_image_top_pose, out_file, save=True):
	pts, cp_points = frame_utils.convert_range_image_to_point_cloud(
		frame,
		range_images,
		camera_projections,
		range_image_top_pose)
	# top_lidar_points = points[0] # 1st lidar, top one
	merged_pointcloud = np.concatenate(pts, axis=0)
	np.save(out_file, merged_pointcloud)

