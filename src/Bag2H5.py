#!/usr/bin/env python
'''
Rosbag's wild clusterfuck of packages needed:
sudo apt-get install:
python-rosbag
python-genmsg
python-genpy
python-rosgraph
python-cv-bridge
python-rosmsg
python-sensor-msgs
'''
from __future__ import print_function
import argparse
import rosbag
import yaml
import subprocess
import std_msgs
import matplotlib.pyplot as plt
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from datetime import datetime
import rospy

import h5py


def msg_to_mat(msg, debayer=False):

    np_img = np.fromstring(msg.data, dtype=np.uint8)
    image_decode = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)
    bgr = np.zeros((image_decode.shape[0], image_decode.shape[1], 3), dtype=np.uint8)
    if debayer:
        cv2.cvtColor(image_decode, cv2.COLOR_BAYER_BG2BGR, bgr, 3)
        return bgr
    else:
        # cv2.cvtColor(image_decode, cv2.COLOR_BGR2RGB, bgr, 3)
        return image_decode


def rescale(cv2image, height=88, width=200):
    rescaled = cv2.resize(cv2image ,(width, height), interpolation = cv2.INTER_CUBIC)
    return rescaled

def get_direction_string(command):
        if command == 3:
            return 'Left'
        if command == 4:
            return 'Right'
        if command == 5:
            return 'Straight'
        if command == 2:
            return 'Follow Lane'



def get_complementary_cmd(topic, command):
    if topic=='/group_left_cam/node_left_cam/image_raw/compressed':
        if command == 2: # middle cam is follow lane
            return 4 # set left cam to right
        if command == 4: # middle cam is right
            return 4 # set left cam to right
        if command == 3: # middle cam is left
            return 3 # set left cam to left
    if topic=='/group_right_cam/node_right_cam/image_raw/compressed':
        if command == 2: # middle cam is follow lane
            return 3 # set left cam to right
        if command == 4: # middle cam is right
            return 4 # set left cam to right
        if command == 3: # middle cam is left
            return 3 # set left cam to left

    #
    #
    # if topic=='/group_right_cam/node_right_cam/image_raw/compressed':
    #     if command == 2: # middle cam is follow lane
    #         comp_cmd = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bag",
                        help="path to RosBag")
    parser.add_argument("-o", "--offset",
                        help="Set offset from playback start in seconds",
                        default = 60)
    args = parser.parse_args()
    bag_path = args.bag
    bag = rosbag.Bag(bag_path, 'r' )
    bag_topics = bag.get_type_and_topic_info()[1].keys()
    offset = float(args.offset)

    # print("Topics in the bag: \n{}".format(bag_topics))

    '''
    bag_topics[0] -> right cam
    bag_topics[1] -> left cam
    bag_topics[2] -> joystick
    bag_topics[3] -> middle cam
    '''

    # bridge = CvBridge()
    # dtype, n_channels = bridge.encoding_as_cvtype2('8UC3')

    dummy_file = h5py.File("../data/AgentHuman/SeqTrain/data_03663.h5")
    # print(type(dummy_file['rgb'][0]), )

    cv2.imshow('pic',dummy_file['rgb'][0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    f = h5py.File('myfile.hdf5','w')
    dset = f.create_dataset("rgb", (2000,88,200,3), np.uint8)
    dset = f.create_dataset("targets", (2000,1), 'f')


    info_dict = yaml.load(rosbag.Bag(bag_path, 'r')._get_yaml_info())
    st_time = datetime.now()
    current_buttons = None
    command = None
    start_time = rospy.Time(1529149890.831419 + offset) # t.secs=10
    for idx, (topics, msg, t) in enumerate(bag.read_messages(start_time=start_time)):
        # timestamp_verbose = datetime.fromtimestamp(t.to_time())
        # print(timestamp_verbose.strftime('%Y-%m-%d %H:%M:%S'), topics)
        #
        # for i, val in enumerate(bag.read_messages(bag_topics[2])):
        #     print(i,val[1].buttons)

        if topics=='/joy':
            '''
            if current_buttons.buttions[4] == 1 -> turn left
            if current_buttons.buttions[5] == 1 -> turn right
            '''
            current_buttons = msg
            COMMAND_DICT =  {2: 'Follow Lane', 3: 'Left', 4: 'Right', 5: 'Straight'}
            if current_buttons.buttons[4] == 1:
                # f["targets"][idx] = 3
                command = 3
                dir_string = get_direction_string(command)
            else:
                if current_buttons.buttons[5] == 1:
                    command = 4
                    dir_string = get_direction_string(command)

                else:
                    if current_buttons.buttons[5] == 1 and current_buttons.buttons[4] == 1:
                        command = 5
                        dir_string = get_direction_string(command)
                    else:
                        command = 2
                        dir_string = get_direction_string(command)




            # print(current_buttons.buttons)
            # if current_buttons.buttons[5]==1.0:
            # print(t, current_buttons.buttons)


        if topics=='/group_middle_cam/node_middle_cam/image_raw/compressed':
            # if current_buttons is not None:
            #     if current_buttons.buttons[5]==1.0:
            image_m = msg_to_mat(msg)
            if current_buttons is not None:
                print(msg.header.stamp, current_buttons.buttons, idx+1 )

            else:
                print(msg.header.stamp)
            font = cv2.FONT_HERSHEY_SIMPLEX
            dir_string = get_direction_string(command)
            if command is not None:
                cv2.putText(image_m,'{} {}'.format(command,dir_string),(10,30), font, 1.0 ,(0,0,255),2)
            cv2.imshow('middlecam',image_m)


        if topics=='/group_right_cam/node_right_cam/image_raw/compressed':
            # if current_buttons is not None:
            #     if current_buttons.buttons[5]==1.0:
            image_r = msg_to_mat(msg)
            if current_buttons is not None:
                print(msg.header.stamp, current_buttons.buttons, idx+1 )

            else:
                print(msg.header.stamp)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cmp_cmd = get_complementary_cmd(topics, command)
            dir_string = get_direction_string(cmp_cmd)

            if command is not None:
                cv2.putText(image_r,'{} {}'.format(cmp_cmd,dir_string),(10,30), font, 1.0 ,(0,0,255),2)
            cv2.imshow('rightcam', image_r)

        if topics=='/group_left_cam/node_left_cam/image_raw/compressed':
            # if current_buttons is not None:
            #     if current_buttons.buttons[5]==1.0:
            image_l = msg_to_mat(msg)
            if current_buttons is not None:
                print(msg.header.stamp, current_buttons.buttons, idx+1 )

            else:
                print(msg.header.stamp)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cmp_cmd = get_complementary_cmd(topics, command)
            dir_string = get_direction_string(cmp_cmd)

            if cmp_cmd is not None:
                cv2.putText(image_l,'{} {}'.format(cmp_cmd,dir_string),(10,30), font, 1.0 ,(0,0,255),2)
            cv2.imshow('leftcam', image_l)



            # font = cv2.FONT_HERSHEY_SIMPLEX
            # if command is not None:
            #     cv2.putText(image,'{} {}'.format(command,dir_string),(10,30), font, 1.0 ,(0,0,255),2)
            #
            # cv2.imshow('pic',image)
            cv2.waitKey(0)
            # rescaled_image = rescale(image)










            # if command is None:
            #     f["targets"][idx] = -1
            # else:
            #     f["targets"][idx] = command
            # f["rgb"][idx,...] = rescaled_image

            # if idx>=500:
            #     break




    # time_elapsed = datetime.now() - start_time

    # print('\nTime elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    # f.close()
    #
    #
    # f = h5py.File('myfile.hdf5','r')
    #
    # cv2.putText(image,'{} {}'.format(command,dir_string),(10,30), font, 1.0 ,(0,0,255),2)
    # cv2.imshow('pic',f['rgb'][1])
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # f.close()
    bag.close()

if __name__=="__main__":
    main()
