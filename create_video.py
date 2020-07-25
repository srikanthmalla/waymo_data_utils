#!/usr/bin/python
#Author: Srikanth Malla
#Date: May 3, 2018
import os
import numpy as np
import cv2
import glob

if __name__ == '__main__':
  global base_input,base_output
  base_input="/home/smalla/scaleAPI_results/vasili/"
  output_path="/home/smalla/scaleAPI_results/vasili/output.mp4"
  images_list=glob.glob(base_input+"*/debug_*")
  images_list.sort()
  fourcc = cv2.VideoWriter_fourcc(*'X264')  # 'x264' doesn't work
  img_c=cv2.imread(images_list[0])
  height,width,layers=img_c.shape
  video=cv2.VideoWriter(output_path,fourcc,10,(width,height))
  counter=0
  for img in images_list:
      print(counter)
      counter+=1
      img_c=cv2.imread(img)
      # Write some Text
      font                   = cv2.FONT_HERSHEY_SIMPLEX
      bottomLeftCornerOfText = (10,500)
      fontScale              = 1
      fontColor              = (255,255,255)
      lineType               = 2
      cv2.putText(img_c,img.split('/')[-2],
          bottomLeftCornerOfText,
          font,
          fontScale,
          fontColor,
          lineType)
      video.write(img_c)
