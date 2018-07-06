#!/usr/bin/env python
'''
Rosbag's wild clusterfuck of packages needed:
sudo apt-get install python-rosbag python-genmsg python-genpy python-rosgraph python-cv-bridge python-rosmsg python-sensor-msgs
'''
from __future__ import print_function,division
import argparse
import rosbag
import yaml
import subprocess
import std_msgs
import cv2
import numpy as np
import rospy
import os
import h5py
import progressbar
import random
import math

CMD_FOLLOW = 2
CMD_LEFT = 3
CMD_RIGHT = 4
CMD_STRAIGHT = 5

BTN_L1 = 4
BTN_R1 = 5
AXIS_LEFT_STICK = 0
AXIS_RIGHT_STICK = 3



'''
Due to random camera initialization of ROS, we have two camera topic setups:
1. Train Bag:

LEFT_CAM_TOPIC = '/group_left_cam/node_left_cam/image_raw/compressed'
MIDDLE_CAM_TOPIC = '/group_right_cam/node_right_cam/image_raw/compressed'
RIGHT_CAM_TOPIC = '/group_middle_cam/node_middle_cam/image_raw/compressed'


2. Eval & Test Bag
LEFT_CAM_TOPIC = '/group_right_cam/node_right_cam/image_raw/compressed'
MIDDLE_CAM_TOPIC = '/group_middle_cam/node_middle_cam/image_raw/compressed'
RIGHT_CAM_TOPIC = '/group_left_cam/node_left_cam/image_raw/compressed'


NOTE: this configuration does not apply to all future data aquisition runs!
TODO: assign fixed serial numbers to the roslaunch in a way that the same camera
is always at its assigned position!

'''


LEFT_CAM_TOPIC = '/group_left_cam/node_left_cam/image_raw/compressed'
MIDDLE_CAM_TOPIC = '/group_right_cam/node_right_cam/image_raw/compressed'
RIGHT_CAM_TOPIC = '/group_middle_cam/node_middle_cam/image_raw/compressed'

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
        if command == CMD_LEFT:
            return 'Left'
        if command == CMD_RIGHT:
            return 'Right'
        '''
        actual command for straight is 5, but 2 is also interpreted as Straight
        since in the phyisical system no follow lane command is used
        '''
        if command == CMD_STRAIGHT or command==2:
            return 'Straight'
        # if command == CMD_FOLLOW:
        #     return 'Follow Lane'


def get_complementary_cmd(topic, command):
    if topic==LEFT_CAM_TOPIC:
        if command == CMD_FOLLOW or command == CMD_STRAIGHT: # middle cam is follow lane
            return CMD_RIGHT # set left cam to right
        if command == CMD_RIGHT: # middle cam is right
            return CMD_RIGHT # set left cam to right
        if command == CMD_LEFT: # middle cam is left
            return CMD_LEFT # set left cam to left

    if topic==RIGHT_CAM_TOPIC:
        if command == CMD_FOLLOW or command == CMD_STRAIGHT: # middle cam is follow lane
            return CMD_LEFT # set left cam to right
        if command == CMD_RIGHT: # middle cam is right
            return CMD_RIGHT # set left cam to right
        if command == CMD_LEFT: # middle cam is left
            return CMD_LEFT # set left cam to left

    return float('nan')




def make_dirs(destination, ENABLE_TEST_BAG):
    middle_destination = destination + 'middle/'
    right_destination = destination + 'right/'
    left_destination = destination + 'left/'

    if not os.path.isdir(destination):
        print("Creating directory..." )
        os.mkdir(destination)

    if not os.path.isdir(middle_destination):
        print("Creating directory..." )
        os.mkdir(middle_destination)
    # if ENABLE_TEST_BAG==False:
    if not os.path.isdir(right_destination):
        print("Creating directory..." )
        os.mkdir(right_destination)

    if not os.path.isdir(left_destination):
        print("Creating directory..." )
        os.mkdir(left_destination)
    # else:
    #     right_destination = None
    #     left_destination = None

    return middle_destination, right_destination, left_destination


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bag",
                        help="path to RosBag")
    parser.add_argument("-o", "--offset",
                        help="Set offset from playback start in seconds",
                        default = 0)
    parser.add_argument("-l", "--location",
                        help="location, e.g.: campus",
                        default = 'campus')
    parser.add_argument("-d", "--destination",
                        help="destination directory of the h5 files,e.g. : ~/DL4AD/data/custom/campus/")

    parser.add_argument("-s", "--show",
                        help="set to True or False, shows camera input",
                        action='store_true')

    parser.add_argument("-e", "--enableTestBag",
                        help="Set to true to also store the original image files",
                        action='store_true')

    parser.add_argument("--only_middle",
                        help="Set to true to only get middle camera images",
                        action='store_true')
    parser.add_argument("-w","--window",
                        help="Set size of rolling window for steering smoothing",
                        default=10)

    args = parser.parse_args()
    destination = args.destination
    location = args.location
    bag_path = args.bag
    offset = float(args.offset)
    SHOW_CAM = args.show
    ENABLE_TEST_BAG = args.enableTestBag
    ONLY_MIDDLE = args.only_middle

    STEERING_OFFSET = 0.45
    TRACE_SIZE = int(args.window)

    # create directories and return paths
    middle_destination, right_destination, left_destination = make_dirs(destination,ENABLE_TEST_BAG)

    bag = rosbag.Bag(bag_path, 'r' )

    # get amount of files needed to maintain 200 images per h5 fily
    n_files = int(bag.get_message_count(RIGHT_CAM_TOPIC)/200)

    font = cv2.FONT_HERSHEY_SIMPLEX

    '''
    bag_topics[0] -> right cam
    bag_topics[1] -> left cam
    bag_topics[2] -> joystick
    bag_topics[3] -> middle cam
    '''

    # initialize to going straight
    current_buttons = None
    command = CMD_STRAIGHT
    cmp_cmd = float('nan')
    analog_steer = 0.0
    analog_gas = 0.0

    start_time = rospy.Time(bag.get_start_time() + offset) # t.secs=10

    COMMAND_DICT =  {CMD_FOLLOW: 'Follow Lane', CMD_LEFT: 'Left', CMD_RIGHT: 'Right', CMD_STRAIGHT: 'Straight', None: None}
    cnt_middle = 0
    cnt_right = 0
    cnt_left = 0

    file_cnt_m = 0
    file_cnt_r = 0
    file_cnt_l = 0

    ctrl_idx = 0
    steer_trace = np.zeros(TRACE_SIZE)

    bar = progressbar.ProgressBar(max_value = bag.get_message_count())

    for idx, (topics, msg, t) in enumerate(bag.read_messages(start_time=start_time)):
        if topics=='/joy':
            '''
            if current_buttons.buttions[4] == 1 -> turn left
            if current_buttons.buttions[5] == 1 -> turn right
            '''
            current_buttons = msg

            l1_pressed = current_buttons.buttons[BTN_L1] == 1
            r1_pressed = current_buttons.buttons[BTN_R1] == 1

            if l1_pressed and r1_pressed:
                command = CMD_STRAIGHT

            elif l1_pressed:
                command = CMD_LEFT
            elif r1_pressed:
                command = CMD_RIGHT
            else:
                command = CMD_STRAIGHT  # this used to be follow lane but we currently don't use that

            dir_string = COMMAND_DICT[command]

            # turning left is positive, turning right is negative
            steer_trace[ctrl_idx % TRACE_SIZE ] =  current_buttons.axes[AXIS_LEFT_STICK]
            ctrl_idx += 1

            # analog_steer = current_buttons.axes[AXIS_LEFT_STICK]



            # left_stick_up_down = current_buttons.axes[1]

            # right_stick_left_right =  current_buttons.axes[2]
            analog_gas =  current_buttons.axes[AXIS_RIGHT_STICK]

        '''
        When we recorded the old graveyard dataset, the middle and right camera
        where switchted, so I switch them here too, but only in the topics string
        old order:                  new order:
        1. middle                   1. right
        2. right                    2. middle
        3. left                     3. left
        '''

        if topics == MIDDLE_CAM_TOPIC:
            if cnt_middle >= 200 or cnt_middle == 0:
                f_m = h5py.File(middle_destination + '{}_{}_{:05d}.h5'.format(location,
                                                                              'middle',
                                                                              file_cnt_m),
                                'w')
                if ENABLE_TEST_BAG:
                    dset = f_m.create_dataset("rgb_original", (200,480,640,3), np.uint8)
                dset = f_m.create_dataset("rgb", (200,88,200,3), np.uint8)
                dset = f_m.create_dataset("targets", (200,3), 'f')
                cnt_middle = 0
                file_cnt_m += 1

            middle_image_original = msg_to_mat(msg)
            rescaled_image = rescale(middle_image_original)

            analog_steer = np.mean(steer_trace)

            # print("steer trace {}".format(steer_trace))
            # print("averaged steer {}".format(analog_steer))
            # print("ctrl index {}".format(ctrl_idx))

            if math.isnan(command):
                f_m["targets"][cnt_middle] = float('nan')
            else:
                targets_m = np.array([command,
                                      analog_steer,
                                      analog_gas])
                f_m["targets"][cnt_middle] = targets_m
            f_m["rgb"][cnt_middle,...] = rescaled_image
            if ENABLE_TEST_BAG:
                f_m["rgb_original"][cnt_middle,...] = msg_to_mat(msg)

            if SHOW_CAM==True:
                cv2.putText(middle_image_original,
                            '{} {:.8f} {:.8f}'.format(f_m['targets'][cnt_middle][0],
                                                      f_m['targets'][cnt_middle][1],
                                                      f_m['targets'][cnt_middle][2]),
                            (10,30),
                            font, 0.5,(0,0,255),2)
                cv2.imshow('pic_m',middle_image_original)
            cnt_middle +=1

        if  ONLY_MIDDLE==False:
            if topics == RIGHT_CAM_TOPIC:
                if cnt_right >= 200 or cnt_right == 0:
                    f_r = h5py.File(right_destination + '{}_{}_{:05d}.h5'.format(location,
                                                                                 'right',
                                                                                 file_cnt_r),
                                    'w')
                    if ENABLE_TEST_BAG:
                        dset = f_r.create_dataset("rgb_original", (200,480,640,3), np.uint8)
                    dset = f_r.create_dataset("rgb", (200,88,200,3), np.uint8)
                    dset = f_r.create_dataset("targets", (200,3), 'f')
                    cnt_right = 0
                    file_cnt_r += 1

                right_image_original = msg_to_mat(msg)
                rescaled_image = rescale(right_image_original)
                cmp_cmd = get_complementary_cmd(topics, command)

                analog_steer = np.mean(steer_trace)

                if math.isnan(cmp_cmd):
                    f_r["targets"][cnt_right] = float('nan')
                else:
                    targets_r = np.array([cmp_cmd, analog_steer + STEERING_OFFSET, analog_gas ])
                    f_r["targets"][cnt_right] = targets_r
                f_r["rgb"][cnt_right,...] = rescaled_image
                if ENABLE_TEST_BAG:
                    f_r["rgb_original"][cnt_right, ...] = right_image_original

                if SHOW_CAM==True:
                    cv2.putText( right_image_original ,'{} {:.8f} {:.8f}'.format(f_r['targets'][cnt_right][0],
                        f_r['targets'][cnt_right][1],f_r['targets'][cnt_right][2]),
                        (10,30), font, 0.5,(0,0,255),2)
                    cv2.imshow('pic_r',right_image_original)
                cnt_right +=1



            if topics == LEFT_CAM_TOPIC:
                if cnt_left >= 200 or cnt_left == 0:
                    f_l = h5py.File(left_destination + '{}_{}_{:05d}.h5'.format(location,'left',file_cnt_l),'w')
                    if ENABLE_TEST_BAG:
                        dset = f_l.create_dataset("rgb_original", (200,480,640,3), np.uint8)
                    dset = f_l.create_dataset("rgb", (200,88,200,3), np.uint8)
                    dset = f_l.create_dataset("targets", (200,3), 'f')
                    cnt_left = 0
                    file_cnt_l += 1

                left_image_original = msg_to_mat(msg)
                rescaled_image = rescale(left_image_original)

                cmp_cmd = get_complementary_cmd(topics, command)

                analog_steer = np.mean(steer_trace)

                if math.isnan(cmp_cmd):
                    f_l["targets"][cnt_left] = float('nan')
                else:
                    targets_l = np.array([cmp_cmd, analog_steer - STEERING_OFFSET, analog_gas ])
                    f_l["targets"][cnt_left] = targets_l
                f_l["rgb"][cnt_left,...] = rescaled_image
                if ENABLE_TEST_BAG:
                    f_l["rgb_original"][cnt_left, ...] = left_image_original

                if SHOW_CAM==True:
                    cv2.putText( left_image_original ,'{} {:.8f} {:.8f}'.format(f_l['targets'][cnt_left][0],
                        f_l['targets'][cnt_left][1],f_l['targets'][cnt_left][2]),
                        (10,30), font, 0.5,(0,0,255),2)
                    cv2.imshow('pic_l',left_image_original)

                cnt_left +=1

        bar.update(idx)
        if SHOW_CAM:# and (cnt_middle==cnt_left) and (cnt_middle==cnt_right):
            cv2.waitKey(0)
            # cv2.destroyAllWindows()


    # remove last h5 file if not complete
    f_m = h5py.File(middle_destination + '{}_middle_{:05d}.h5'.format(location, file_cnt_m-1),'r')
    if f_m['rgb'][-1].all() == 0:
        os.remove(middle_destination + '{}_middle_{:05d}.h5'.format(location,file_cnt_m-1))
        if ENABLE_TEST_BAG==False:
            os.remove(right_destination + '{}_right_{:05d}.h5'.format(location,file_cnt_m-1))
            os.remove(left_destination + '{}_left_{:05d}.h5'.format(location,file_cnt_m-1))
            print("removing:\n{}\n{}\n{}".format(
            '{}_middle_{:05d}.h5'.format(location,file_cnt_m-1),
            '{}_right_{:05d}.h5'.format(location,file_cnt_m-1),
            '{}_left_{:05d}.h5'.format(location,file_cnt_m-1)
        ))
    f_m.close

    # # show random picture
    # file_idx = random.randint(0, int(file_cnt_m-1))
    # f_m = h5py.File(middle_destination + '{}_middle_{:05d}.h5'.format(location,file_idx),'r')
    # f_l = h5py.File(left_destination + '{}_left_{:05d}.h5'.format(location,file_idx),'r')
    # f_r = h5py.File(right_destination + '{}_right_{:05d}.h5'.format(location,file_idx),'r')
    #
    #
    # pic_idx =random.randint(0,199)
    #
    # m_img = f_m['rgb_original'][pic_idx]
    # cv2.putText( m_img ,'{}'.format(f_m['targets'][pic_idx]) ,(10,30), font, 0.5,(0,0,255),2)
    # cv2.imshow('pic_m',m_img)
    #
    # r_img = f_r['rgb_original'][pic_idx]
    # cv2.putText( r_img ,'{}'.format(f_r['targets'][pic_idx]) ,(10,30), font, 0.5,(0,0,255),2)
    # cv2.imshow('pic_r',r_img)
    #
    # l_img = f_l['rgb_original'][pic_idx]
    # cv2.putText( l_img ,'{}'.format(f_l['targets'][pic_idx]) ,(10,30), font, 0.5,(0,0,255),2)
    # cv2.imshow('pic_l',l_img)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    bag.close()




if __name__=="__main__":
    main()
