#!/usr/bin/env python

import argparse
import rosbag
import yaml
import subprocess
import std_msgs
import matplotlib.pyplot as plt
import cv2
from cv_bridge import CvBridge, CvBridgeError
import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bag",
                        help="path to RosBag")
    args = parser.parse_args()
    bag_path = args.bag
    bag = rosbag.Bag(bag_path, 'r', compression='bz2')
    bag_topics = bag.get_type_and_topic_info()[1].keys()

    # print("Topics in the bag: \n{}".format(bag_topics))

    '''
    bag_topics[0] -> right cam
    bag_topics[1] -> left cam
    bag_topics[2] -> joystick
    bag_topics[3] -> middle cam
    '''

    bridge = CvBridge()
    # dtype, n_channels = bridge.encoding_as_cvtype2('8UC3')




    for topics, msg, t in bag.read_messages(topics=bag_topics[0]):
        timestamp_verbose = datetime.datetime.fromtimestamp(t.to_time())

        print(timestamp_verbose.strftime('%Y-%m-%d %H:%M:%S'), topics)
        # cv_image = bridge.compressed_imgmsg_to_cv2(msg, "passthrough")
        # exit()
        print(type(msg))
        print(msg)
        cv_image = bridge.imgmsg_to_cv2(msg)

    # for topic, msg, t in bag.read_messages():
    #     if topic == '/group_middle_cam/node_middle_cam/image_raw/compressed':
    #         print(t,topic)
    #         print(type(msg))
    #
    #         exit()


    bag.close()

if __name__=="__main__":
    main()
