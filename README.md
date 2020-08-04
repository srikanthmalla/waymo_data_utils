# scripts for waymo dataset

output video [with annotated labels]: https://www.youtube.com/watch?v=YjOUamkRBRo
output video [with lidar projected labels]: https://www.youtube.com/watch?v=ekizQEwPqew

## prerequisites:

1. download and extract v_1_2 of waymo (with labels)

2. ``` pip install waymo-open-dataset-tf-2-1-0==1.2.0 ```

## data extraction and visualization utils:

``
save_data.py: save images, ego-motion, lidar and labels (for both images and lidar)
``

``
visualization.py: creates merged sensor and label data for visualization
``

``
create_video.py: creates video using images created by visualization
``


## Note:

Tracking IDS are not same across sensors in waymo data for same object.
