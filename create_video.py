#!/usr/bin/python
#Author: Srikanth Malla
#Date: July 26, 2020

import os
import numpy as np
import cv2
import glob

if __name__ == '__main__':
  global base_input,base_output
  base_input="/home/smalla/waymo_data/outputs/"
  output_path="/home/smalla/waymo_ws/merged_video.avi"
  scenes=glob.glob(base_input+"*")
  all_imgs_len = len(glob.glob(base_input+"*/merged_image/*.png"))
  scenes.sort()
  fourcc = cv2.VideoWriter_fourcc(*'MPEG')  # 'x264' doesn't work
  # img_c=cv2.imread(images_list[0])
  height=width=1080
  video=cv2.VideoWriter(output_path,fourcc, 10, (width,height))
  counter=0

  for scene_count, scene in enumerate(scenes):
    images_list = glob.glob(scene+"/merged_image/*.png")
    images_list.sort()
    print("completed: ", counter,"/",all_imgs_len," = ",round(float(counter/all_imgs_len)*100,2),"%")
    for img in images_list:
      counter+=1
      img_c=cv2.imread(img)
      # Write some Text
      font                   = cv2.FONT_HERSHEY_SIMPLEX
      fontScale              = 0.5
      fontColor              = (0,0,0)
      lineType               = 1

      text = "scene count: "+str(scene_count).zfill(6)
      bottomLeftCornerOfText = (int(width/3),900)
      cv2.putText(img_c, text,
          bottomLeftCornerOfText,
          font,
          fontScale,
          fontColor,
          lineType, cv2.LINE_AA)

      text = "global frame count: "+str(counter).zfill(6)
      bottomLeftCornerOfText = (int(width/3),980)
      cv2.putText(img_c, text,
          bottomLeftCornerOfText,
          font,
          fontScale,
          fontColor,
          lineType, cv2.LINE_AA)
      # cv2.imshow("img", img_c)
      # cv2.waitKey(300)
      video.write(img_c)
