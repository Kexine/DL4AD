#!/usr/bin/env python
import rospy

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

import torch
from torchvision import datasets, transforms

from sensor_msgs.msg import Image

# Instantiate CvBridge
bridge = CvBridge()

def callback(data):
    # rospy.loginfo(data)
    try:
         # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        cv2_img = cv2.resize(cv2_img, (200, 88))


        
        # show the image
        cv2.imshow("Narf", cv2_img)
        cv2.waitKey()
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('raiscar_autonomous', anonymous=True)

    rospy.Subscriber("usb_cam/image_raw", Image, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
