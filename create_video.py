#!/usr/bin/python
# Author: Srikanth Malla

import cv2
import glob
import os
from os import path
from tqdm import tqdm


input_dir = "/home/smalla/waymo_data/outputs/"
scenes = glob.glob(input_dir+"*")
scenes.sort()
global_counter = 0

height = 1080
width = 1080

def create_dir(folder):
	if not path.exists(folder):
		os.mkdir(folder)

def main():
	out = cv2.VideoWriter('merged_video.avi',cv2.VideoWriter_fourcc(*'MPEG'), 10, (width, height))
	global global_counter
	for scene_count, scene in enumerate(tqdm(scenes)):
		lidar_top_images = glob.glob(scene+"/lidar_top/*.png")
		create_dir(scene+"/merged_image")
		lidar_top_images.sort()
		local_counter = 0
		camera_labels = False
		if "_with_camera_labels" in scene:
			camera_labels = True
		for top_img in lidar_top_images:
			merged = cv2.imread(top_img)

			img_resize_w = int(width/3)
			img_resize_h = int(height/4)
			# front left
			if path.isfile(top_img.replace("lidar_top","FRONT_LEFT")):
				cam_front_left_image = cv2.imread(top_img.replace("lidar_top","FRONT_LEFT"))
			x_offset=0; y_offset=0;
			cam_front_left_image = cv2.resize(cam_front_left_image, (img_resize_w, img_resize_h))
			merged[y_offset:y_offset+cam_front_left_image.shape[0], x_offset:x_offset+cam_front_left_image.shape[1]] = cam_front_left_image

			# front left
			if path.isfile(top_img.replace("lidar_top","FRONT")):
				cam_front_image = cv2.imread(top_img.replace("lidar_top","FRONT"))
			x_offset=int(width/3); y_offset=0;
			cam_front_image = cv2.resize(cam_front_image, (img_resize_w, img_resize_h))
			merged[y_offset:y_offset+cam_front_image.shape[0], x_offset:x_offset+cam_front_image.shape[1]] = cam_front_image

			#front right
			if path.isfile(top_img.replace("lidar_top","FRONT_RIGHT")):
				cam_front_right_image = cv2.imread(top_img.replace("lidar_top","FRONT_RIGHT"))
			x_offset=int(2*width/3); y_offset=0;
			cam_front_right_image = cv2.resize(cam_front_right_image, (img_resize_w, img_resize_h))
			merged[y_offset:y_offset+cam_front_right_image.shape[0], x_offset:x_offset+cam_front_right_image.shape[1]] = cam_front_right_image

			img_resize_w = int(width/3)
			img_resize_h = int(height/4)
			# side left
			if path.isfile(top_img.replace("lidar_top","SIDE_LEFT")):
				cam_side_left_image = cv2.imread(top_img.replace("lidar_top","SIDE_LEFT"))
			x_offset=0; y_offset=int(3*height/4);
			cam_side_left_image = cv2.resize(cam_side_left_image, (img_resize_w, img_resize_h))
			merged[y_offset:y_offset+cam_side_left_image.shape[0], x_offset:x_offset+cam_side_left_image.shape[1]] = cam_side_left_image

			# side right
			if path.isfile(top_img.replace("lidar_top","SIDE_RIGHT")):
				cam_side_right_image = cv2.imread(top_img.replace("lidar_top","SIDE_RIGHT"))
			x_offset=int(2*width/3); y_offset=int(3*height/4);
			cam_side_right_image = cv2.resize(cam_side_right_image, (img_resize_w, img_resize_h))
			merged[y_offset:y_offset+cam_side_right_image.shape[0], x_offset:x_offset+cam_side_right_image.shape[1]] = cam_side_right_image

			file = scene.split("/")[-1]
			file = file.replace("segment-","")
			file = file.replace("_with_camera_labels","")
			offset = 20
			y_start = 880
			merged = cv2.putText(merged, "segment:", (int(width/3), y_start), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 0, 0), 1, cv2.LINE_AA)

			y_start+=offset
			merged = cv2.putText(merged, file, (int(width/3), y_start), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 0, 0), 1, cv2.LINE_AA)

			text = "scene count: "+str(scene_count).zfill(6)
			y_start+=2*offset
			merged = cv2.putText(merged, text, (int(width/3), y_start), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 0, 0), 1, cv2.LINE_AA)

			text = "local frame count: "+str(local_counter).zfill(6)
			y_start+=offset
			merged = cv2.putText(merged, text, (int(width/3), y_start), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 0, 0), 1, cv2.LINE_AA)

			text = "global frame count: "+str(global_counter).zfill(6)
			y_start+=offset
			merged = cv2.putText(merged, text, (int(width/3), y_start), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 0, 0), 1, cv2.LINE_AA)

			text = "camera labels: "+str(camera_labels)
			y_start+=offset
			merged = cv2.putText(merged, text, (int(width/3), y_start), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 0, 0), 1, cv2.LINE_AA)

			global_counter+=1
			local_counter+=1
			# write to video
			out.write(merged)
			# save output
			cv2.imwrite(top_img.replace("lidar_top","merged_image"), merged)
			# visualize
			# cv2.imshow("merged", merged)
			# cv2.waitKey(100)
	out.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
