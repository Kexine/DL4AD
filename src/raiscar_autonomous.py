#!/usr/bin/env python
import rospy

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

import torch
from torchvision import datasets, transforms

import numpy as np

import command_input_raiscar

from sensor_msgs.msg import Image

# Instantiate CvBridge
bridge = CvBridge()

class AutonomousDriver(object):
    def __init__(self):
        # In ROS, nodes are uniquely named. If two nodes with the same
        # node are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
        rospy.init_node('raiscar_autonomous', anonymous=True)
        rospy.Subscriber("usb_cam/image_raw",
                         Image,
                         self.callback)

        # -------------------- Initialize the NN
        # set the cuda device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device set to " + str(self.device))
    
        self.model = command_input_raiscar.Net().to(self.device)
        self.model.load_state_dict(torch.load("/home/minicar1/deep_learning_models/command_augmented_hwsmooth_50.pt"))

    def callback(self, data):
        # rospy.loginfo(data)
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError, e:
            print(e)
        else:
            cv2_img = cv2.resize(cv2_img, (200, 88))
            net_img = np.asarray(cv2_img,
                                 dtype=np.float32)  # np.transpose(cv2_img, (1,0,2)))
            net_img /= 255
            net_img = np.transpose(net_img,
                                   (2,0,1))

            net_img = torch.Tensor(net_img).unsqueeze(0).to(self.device)
            high_level_command = torch.Tensor([5.0]).unsqueeze(0).to(self.device)
            steering = self.model(net_img,
                                  high_level_command)

            print(steering.cpu().detach().numpy())

            # show the image
            # cv2.imshow("Narf", cv2_img)
            # cv2.waitKey()


if __name__ == '__main__':
    ad = AutonomousDriver()
    rospy.spin()
