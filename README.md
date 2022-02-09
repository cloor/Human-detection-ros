# Human Detection with Ros

- This repo illustrates how to use Human Detection modul with usb_cam by Ros
- We need Ros, pytorch, Cvbridge, etc..
- requirments must be installed (see Reference).
- usb_cam launch file is in face-recognition-ros repo see that.
## Ros
- We will skip step of basic Ros settigs.
- Architecture
    - Publish usb_cam images with Ros
        ```
        roslaunch usb_cam usb_cam-test.launch
        ```
    - Subscribe image topics in code, and convert it to Opencv_images with CVbridge
    - with Opencv images do Human Detection
    - Publish result bbox, score, num of human.

# Object Detection
- Extract bbox(bounding box) and class
- input : image data
- output : [xmin, ymin, xmax, ymax, box confidence, class, class confidence(score)]
## Human Detection
- In Object Detection only detect human.
- Using transfer learning
- Trained by AVA dataset
# Dataset
 ## AVA dataset 
- In Youtube video, doing bbox annotation 
- you can download dataset here [https://github.com/DoranLyong/AVA-Dataset-Processing-for-Person-Detection] 
    - How to download
        - clone github repository
        ```bash
        git clone https://github.com/DoranLyong/AVA-Dataset-Processing-for-Person-Detection.git
        cd AVA_Dataset_Processing-for-Person_Detection.git
        mkdir dataset
        ```
        - Get in[https://research.google.com/ava/download.html#ava_kinetics_download]download ava_v2.2.zip and Unzip the file into dataset
        - setup.sh
        ```bash
        bash setup.sh
        ```

        - Requiremnet
        ```bash
        pip install -r requiremnets.txt
        ```

        - Download YouTube Video and Image Frames
        ```bash
        python ava_youtube_download.py
        python cut_frames_from_video.py
        ```

        - Detection labels 
            - AVA dataset has 430 video files(235 : training / 64 : validation / 131 : test) 
            - Label have data about human localization & action recognition
            - Get YOLO format
            ```bash
            python cvt_annotation_format_csv_to_txt.py
            python label_test.py
            ```
## COCO dataset
- You can download in here https://cocodataset.org/#home
- 80 object classes but we use only human class.
- coco data format
## YOLO -> COCO format
-  To train YOLOX, we need to convert dataset YOLO format to COCO format. 
- https://github.com/RapidAI/YOLO2COCO 
    ```
    YOLOV5
    ├── classes.txt
    ├── xxxx
    │   ├── images
    │   └── labels
    ├── train.txt
    └── val.txt
    ```
- In dataset make dir like above.
- classes.txt file has only human class
- train.txt and val.txt have to made by self.
- Run 
    ```bash
    python yolov5_2_coco.py --dir_path dataset/YOLOV5
    ```
# YOLOX-nano for human detection
## 1. Clone YOLOX repository
  ```bash
  git clone https://github.com/Megvii-BaseDetection/YOLOX.git
  ```

## 2. Prepare Dataset
  ### AVA dataset 
  - move yolo2coco dataset to YOLOX/dataset 
  
## 3. Training YOLOX nano
- In YoloX github download weight of YOLOX-nano
- Set weight file to YOLOX
- Traning
```bash
cd YOLOX
python tools/train.py -f exp/default/nano.py -d 1 -b 8 --fp16 -c yolox_nano.pth
```
  - -d : num of devices (# of gpus) -b : num of batch (recommended : 8 times of device)
  - -c : pth file for transfer learning  
## How to use (you must launch 'usb_cam')
### - Run model
    rosrun yolox inference.py
### - Publish msg to activate
    rostopic pub --once /human_detection_msg std_msgs/String "data: 'On'"

# Reference
- YOLOX-nano link: https://github.com/Megvii-BaseDetection/YOLOX.git
    ```
    git clone https://github.com/Megvii-BaseDetection/YOLOX.git
    ```
- Usb_cam link: https://github.com/ros-drivers/usb_cam.git
    ```
    git clone https://github.com/ros-drivers/usb_cam.git
    ```
- yolo2coco link: https://github.com/RapidAI/YOLO2COCO 
    ```
    git clone https://github.com/RapidAI/YOLO2COCO 
    ```