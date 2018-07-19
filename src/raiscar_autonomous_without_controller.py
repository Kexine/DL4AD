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
from raiscar.msg import MotorController

# Instantiate CvBridge
bridge = CvBridge()
camera_topic = "/middle_cam/image_raw"
class AutonomousDriver(object):
    def __init__(self):
        # In ROS, nodes are uniquely named. If two nodes with the same
        # node are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.

        rospy.init_node('raiscar_autonomous', anonymous=True)
        self.control_pub = rospy.Publisher('controls', MotorController, queue_size=1000)
        rospy.Subscriber(camera_topic,
                         Image,
                         self.callback)

        # -------------------- Initialize the NN
        # set the cuda device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device set to " + str(self.device))
        self.control_signal = MotorController()

        # self.model = command_input_raiscar.Net().to(self.device)
        self.model = Branched_raiscar.Net().to(self.device)
        self.model.load_state_dict(torch.load("/home/minicar1/deep_learning_models/b_aug_hw_smoothed_clipped_110718_cont.pt"))


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
            print(net_img.shape)

            # unsqueeze, to make a "batch" of size 1
            net_img = torch.Tensor(net_img).unsqueeze(0).to(self.device)

            # currently, we only have the high level command straight
            high_level_command = torch.Tensor([5.0]).unsqueeze(0).to(self.device)

            # actual applying of the model
            control_predict = self.model(net_img,
                                         high_level_command)
            control_predict = control_predict.cpu().detach().numpy()



            # we get a vector of size (1,2)
            control_predict[0,1] = 0.45
            self.control_signal.angle = control_predict[0,0]
            self.control_signal.speed = control_predict[0,1]
            self.control_pub.publish(self.control_signal)

            rospy.loginfo("\nSteer predicted: " + str(control_predict[0,0]) + "\nAcc. predicted: " + str(control_predict[0,1]))


            # show the image
            # cv2.imshow("Narf", cv2_img)
            # cv2.waitKey()


if __name__ == '__main__':
    ad = AutonomousDriver()
    rospy.spin()
