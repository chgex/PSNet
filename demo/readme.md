

## Introduction

The demo of using PSNet to segment and locate iris.

## How to use

### detect and segment iris from eye images:
"./images-001" has some sample of eye images, you can run script directly:
```
    python detect_eye_images.py
```

The sample image is from I-Social-DB dataset, you can get the full dataset by visiting the download link in their paper: "I-SOCIAL-DB: A labeled database of images collected from websites and social media for Iris recognition".

### detect from other image:
detect face from image, then, detect and segment iris from eye region, use:
```
    python detect_images.py
```
"./images-002" has one image ([baidu link](https://img9.51tietu.net/pic/20190917/n0hajedc5w4n0hajedc5w4.jpg)), you can run above script directly.

when the above script is run, it will first use [yolov5_face](https://github.com/deepcam-cn/yolov5-face) to detect the face area, and then use the psnet algorithm to locate and segment the iris.

### detect from camera
open the camera first, ant then, run script:
```
   python detect_cam.py 
```

## Demo

The first demo in "__md__/t1.gif".

<img src="__md__/t1.gif" style="zoom:35%;" />


The second demo is a video, in "__md__/t2.avi".
![](__md__/t2.avi)



