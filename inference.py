#!/usr/bin/env python

import os
import time
from loguru import logger

import cv2

import torch

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msg.msg import human_detection_result

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import postprocess, vis

# CV bridge : OpenCV 와 ROS 를 이어주는 역할 
bridge = CvBridge()

# initialize result publisher
result_pub = rospy.Publisher('human_detection_result', human_detection_result, queue_size=10)


class Predictor(object):
    def __init__(self, model, exp, class_names = COCO_CLASSES, fp16 = False,
    device = 'cpu', legacy = False):
        self.model = model
        self.cls_name = class_names
        self.num_class = exp.num_classes
        self.confthre = 0.5
        self.nmsthre = 0.45
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy = legacy)
        self.test_size = exp.test_size


    def inference(self, img):
        # img_info dictionary 작성 

        img_info = {'id' : 0}

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0] , 
                self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0) # batch 차원 추가
        img = img.float()
        img.cpu() # cpu로 처리
        ### 참고 : apex를 이용해서 fp16을 사용하면 연산속도가 빨라진다고 함 -> 나중에 공부하기### 

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_class, self.confthre, self.nmsthre,class_agnostic=True
            )
        return outputs, img_info


    def visual(self, output, img_info, cls_conf = 0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_name)
        return vis_res



def camera_callback(predictor):    

    t0 = time.time()
    # Get ROS image using wait for message
    ros_image = rospy.wait_for_message("/camera/color/image_raw", Image)
    frame_bgr = bridge.imgmsg_to_cv2(ros_image, desired_encoding = 'bgr8')

    outputs, img_info = predictor.inference(frame_bgr)
    result_image = predictor.visual(outputs[0], img_info, predictor.confthre)

    cv2.putText(frame_bgr, "fps : {}".format(1.0 / (time.time()-t0)), (10,20), cv2.FONT_HERSHEY_SIMPLEX,
    0.7, (255,0,0), 2)
    cv2.imshow('YOLOX_NANO', result_image)

    cv2.waitKey(1)

    # 퍼블리시하는 부분
    try:
        outputs = outputs[0].tolist()

        hd_result = human_detection_result()
        
        hd_result.num_person = str(len(outputs))
        
        for num, res in enumerate(outputs):
            hd_result.xmin += str(int(res[0])) + ' '
            hd_result.ymin += str(int(res[1])) + ' '
            hd_result.xmax += str(int(res[2])) + ' '
            hd_result.ymax += str(int(res[3])) + ' '
            hd_result.conf += str(res[4]*res[5])[:4] + ' '

        result_pub.publish(hd_result)
    except:
        pass
     

def message_callback(message):
    global flag
    if message == String("On"):
        flag = 1
    else:
        flag = 0

if __name__=='__main__':

    print('start')

    # Set model 
    exp = get_exp('catkin_ws/src/yolox/scripts/exps/default/nano.py', 'nano')
    model = exp.get_model()
    model.eval()
    ckpt = torch.load('catkin_ws/src/yolox/scripts/90epoch_ckpt.pth', map_location = 'cpu')
    model.load_state_dict(ckpt["model"])
    
    # create object predicting results
    predictor = Predictor(model, exp, COCO_CLASSES)
    
    # ROS node init
    rospy.init_node('opencv_node', anonymous = True)
    
    # Set FLAG for On/Off controll 
    global flag 
    flag = 0

    while True:
        message_sub = rospy.Subscriber('/human_detection_msg', String, message_callback)
 
        if flag == 1:
            camera_callback(predictor)
    cv2.destroyAllWindows()
    
