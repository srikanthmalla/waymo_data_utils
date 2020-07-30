# Author: Srikanth Malla
# Date: 11 July 2020
# extract the sensor data from tf records and save in folders

import os
from utils import *
import glob
import tensorflow.compat.v1 as tf
from joblib import Parallel, delayed
from progressbar import ProgressBar
import multiprocessing
tf.enable_eager_execution()
## hyperparams
output_folder = "/home/smalla/waymo_data/outputs/"
# input_folder = "/home/smalla/waymo_data/v_1_2/Training/tfrecords/"
# input_folder = "/home/smalla/waymo_data/v_1_2/Validation/tfrecords/"
input_folder = "/home/smalla/waymo_data/v_1_2/Domain_Adaptation/tfrecords/"
files = glob.glob(input_folder+"*.tfrecord")

camera_images = False
pointclouds = False
top_view_lidar_image = False
camera_labels = False
lidar_labels = False
ego_vehicle_motion = False
projected_lidar_labels = True

def create_dir(folder):
	if not os.path.exists(folder):
		os.mkdir(folder)

def process(file):
	outfolder = output_folder+file.split("/")[-1].split(".")[0]
	create_dir(outfolder)
	dataset = tf.data.TFRecordDataset(file, compression_type='')
	## read all frames
	for indx, data in enumerate(dataset):
		frame = open_dataset.Frame()
		frame.ParseFromString(bytearray(data.numpy()))
		(range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

		# # 1.camera images
		if camera_images:
			visualize_cameras(frame, range_images, camera_projections, range_image_top_pose, outfolder, indx, save=True)

		# # # 2.top view lidar image
		if top_view_lidar_image:
			out_file1 = outfolder+"/lidar_top"
			create_dir(out_file1)
			out_file = out_file1+"/"+str(indx).zfill(6)+".png"
			if not os.path.isfile(out_file):
				lidar_top_view(frame, range_images, camera_projections, range_image_top_pose, out_file, save=True)

		# # 3.pointclouds
		if pointclouds:
			out_file1 = outfolder+"/pointclouds"
			create_dir(out_file1)
			out_file = out_file1+"/"+str(indx).zfill(6)+".npy"
			if not os.path.isfile(out_file):
				lidar_data(frame, range_images, camera_projections, range_image_top_pose, out_file, save=True)

		#4. camera labels
		if camera_labels:
			labels_camera(frame, range_images, camera_projections, range_image_top_pose, outfolder, indx, save=True)

		#5. pointcloud labels
		if lidar_labels:
			out_file1 = outfolder+"/labels_pc"
			create_dir(out_file1)
			out_file = out_file1+"/"+str(indx).zfill(6)+".npy"
			if not os.path.isfile(out_file):
				labels_pc(frame, range_images, camera_projections, range_image_top_pose, out_file, indx, save=True)

		#6. ego motion
		if ego_vehicle_motion:
			out_file1 = outfolder+"/ego_motion"
			create_dir(out_file1)
			out_file = out_file1+"/"+str(indx).zfill(6)+".npy"
			if not os.path.isfile(out_file):
				ego_motion(frame, out_file, save=True)

		#7. projection of lidar labels on to image (for the purpose of tracking ID association across sensors)
		if projected_lidar_labels:
			lidar_labels_on_cam(frame, outfolder, indx, save=True)

	print(file)


if __name__ == '__main__':
	## read all files
	parallel= False

	files.sort()

	# batch = files[0:int(len(files)/3)]
	# batch = files[int(len(files)/3):int(2*len(files)/3)]
	# batch = files[int(2*len(files)/3):len(files)]

	batch = files

	## sequential
	if not parallel:
		pbar = ProgressBar()
		for file in pbar(batch):
			process(file)
			print(file)
	## parallel
	if parallel:
		num_cores = multiprocessing.cpu_count()
		results = Parallel(n_jobs=num_cores)(delayed(process)(file) for file in files)



