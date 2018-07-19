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
import Branched_raiscar

from sensor_msgs.msg import Image
from sensor_msgs.msg import Joy
from raiscar.msg import MotorController

# Instantiate CvBridge
bridge = CvBridge()
camera_topic = "/middle_cam/image_raw"

class myvariables:
    hlcmd = 5.0
    acceleration = 0.0
    steer = 0.0

class AutonomousDriver(object):
    def __init__(self, VARS):
        # In ROS, nodes are uniquely named. If two nodes with the same
        # node are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
        self.VARS = VARS
        rospy.init_node('raiscar_autonomous', anonymous=True)
        self.control_pub = rospy.Publisher('controls', MotorController, queue_size=1000)
        rospy.Subscriber(camera_topic,
                         Image,
                         self.callback)
        rospy.Subscriber("/joy",
                         Joy,
                         self.controller_callback)

        # -------------------- Initialize the NN
        # set the cuda device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device set to " + str(self.device))
        self.control_signal = MotorController()

        # self.model = command_input_raiscar.Net().to(self.device)
        self.model = Branched_raiscar.Net().to(self.device)
        self.model.load_state_dict(torch.load("/home/minicar1/deep_learning_models/b_aug_hw_smoothed_clipped_110718_cont.pt"))


    def controller_callback(self, msg):
        #mapping the acceleration
        acceleration = 1-((1-msg.axes[3])/2)

        if msg.buttons[4] == 1 and msg.buttons[5] == 0:
            self.VARS.hlcmd = 3.0
        elif msg.buttons[4] == 0 and msg.buttons[5] == 1:
            self.VARS.hlcmd = 4.0
        else:
            self.VARS.hlcmd = 5.0

        print(self.VARS.hlcmd)

    def callback(self, data):
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError, e:
            print(e)
        else:
            cv2_img = cv2.resize(cv2_img, (200, 88))

            # convert the uint8 image to floats
            net_img = np.asarray(cv2_img,
                                 dtype=np.float32)
            net_img /= 255

            net_img = np.transpose(net_img,
                                   (2,0,1))
            #print(net_img.shape)

            # unsqueeze, to make a "batch" of size 1
            net_img = torch.Tensor(net_img).unsqueeze(0).to(self.device)

            # currently, we only have the high level command straight
            print(self.VARS.hlcmd)
            high_level_command = torch.Tensor([self.VARS.hlcmd]).unsqueeze(0).to(self.device)

            # actual applying of the model
            control_predict = self.model(net_img,
                                         high_level_command)
            control_predict = control_predict.cpu().detach().numpy()

            # we get a vector of size (1,2)
            control_predict[0,1] = 0.45
            self.control_signal.angle = control_predict[0,0]
            if acceleration >= 0.7 or acceleration <= 0.3:
                self.control_signal.speed = acceleration
            else:
                self.control_signal.speed = 0.5
            self.control_pub.publish(self.control_signal)

            #rospy.loginfo("\nSteer predicted: " + str(control_predict[0,0]) + "\nAcc. predicted: " + str(control_predict[0,1]))


            # show the image
            # cv2.imshow("Narf", cv2_img)
            # cv2.waitKey()


if __name__ == '__main__':
    VARS = myvariables()
    ad = AutonomousDriver(VARS)
    rospy.spin()
