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
        '''
        actual command for straight is 5, but 2 is also interpreted as Straight
        since in the phyisical system no follow lane command is used
        '''
        if command == 5 or command==2:
            return 'Straight'
        # if command == 2:
        #     return 'Follow Lane'



def get_complementary_cmd(topic, command):
    if topic=='/group_left_cam/node_left_cam/image_raw/compressed':
        if command == 2 or command == 5: # middle cam is follow lane
            return 4 # set left cam to right
        if command == 4: # middle cam is right
            return 4 # set left cam to right
        if command == 3: # middle cam is left
            return 3 # set left cam to left

    if topic=='/group_right_cam/node_right_cam/image_raw/compressed':
        if command == 2 or command == 5: # middle cam is follow lane
            return 3 # set left cam to right
        if command == 4: # middle cam is right
            return 4 # set left cam to right
        if command == 3: # middle cam is left
            return 3 # set left cam to left

    return float('nan')




def make_dirs(destination):

    middle_destination = destination + 'middle/'
    right_destination = destination + 'right/'
    left_destination = destination + 'left/'

    if not os.path.isdir(destination):
        print("Creating directory..." )
        os.mkdir(destination)

    if not os.path.isdir(middle_destination):
        print("Creating directory..." )
        os.mkdir(middle_destination)

    if not os.path.isdir(right_destination):
        print("Creating directory..." )
        os.mkdir(right_destination)


    if not os.path.isdir(left_destination):
        print("Creating directory..." )
        os.mkdir(left_destination)


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

    args = parser.parse_args()
    destination = args.destination
    location = args.location
    bag_path = args.bag
    offset = float(args.offset)
    SHOW_CAM = args.show

    STEERING_OFFSET = 0.45



    # create directories and return paths
    middle_destination, right_destination, left_destination = make_dirs(destination)

    bag = rosbag.Bag(bag_path, 'r' )

    # get amount of files needed to maintain 200 images per h5 fily
    n_files = int(bag.get_message_count('/group_middle_cam/node_middle_cam/image_raw/compressed')/200)

    font = cv2.FONT_HERSHEY_SIMPLEX


    '''
    bag_topics[0] -> right cam
    bag_topics[1] -> left cam
    bag_topics[2] -> joystick
    bag_topics[3] -> middle cam
    '''

    current_buttons = None
    command = float('nan')
    cmp_cmd = float('nan')

    start_time = rospy.Time(bag.get_start_time() + offset) # t.secs=10

    COMMAND_DICT =  {2: 'Follow Lane', 3: 'Left', 4: 'Right', 5: 'Straight', None: None}
    cnt_middle = 0
    cnt_right = 0
    cnt_left = 0

    file_cnt_m = 0
    file_cnt_r = 0
    file_cnt_l = 0

    widgets = [ progressbar.widgets.Bar(),
           ' ', progressbar.widgets.Timer(),
           ' ', progressbar.widgets.AdaptiveETA(),
           ' ']
    bar = progressbar.ProgressBar(max_value = bag.get_message_count(),
                              widgets = widgets)

    for idx, (topics, msg, t) in enumerate(bag.read_messages(start_time=start_time)):
        if topics=='/joy':
            '''
            if current_buttons.buttions[4] == 1 -> turn left
            if current_buttons.buttions[5] == 1 -> turn right
            '''
            current_buttons = msg

            if current_buttons.buttons[4] == 1:
                command = 3
                dir_string = COMMAND_DICT[command]
            else:
                if current_buttons.buttons[5] == 1:
                    command = 4
                    dir_string = COMMAND_DICT[command]

                else:
                    if current_buttons.buttons[5] == 1 and current_buttons.buttons[4] == 1:
                        command = 5
                        dir_string = COMMAND_DICT[command]
                    else:
                        command = 5
                        dir_string = COMMAND_DICT[command]

            # turning left is positive, turning right is negative
            analog_steer = current_buttons.axes[0]
            # left_stick_up_down = current_buttons.axes[1]

            # right_stick_left_right =  current_buttons.axes[2]
            analog_gas =  current_buttons.axes[3]



        if topics == '/group_middle_cam/node_middle_cam/image_raw/compressed':
            if cnt_middle >= 200 or cnt_middle == 0:
                f_m = h5py.File(middle_destination + '{}_{}_{:05d}.h5'.format(location,'middle',file_cnt_m),'w')
                dset = f_m.create_dataset("rgb_original", (200,480,640,3), np.uint8)
                dset = f_m.create_dataset("rgb", (200,88,200,3), np.uint8)
                dset = f_m.create_dataset("rgb_original", (640,800,200,3), np.uint8)
                dset = f_m.create_dataset("targets", (200,3), 'f')
                cnt_middle = 0
                file_cnt_m += 1

<<<<<<< HEAD
            middle_image_original = msg_to_mat(msg)
            rescaled_image = rescale(middle_image_original)
=======
            middle_image_origin = msg_to_mat(msg)
            rescaled_image = rescale(middle_image_origin)
>>>>>>> 2c59425c59d0d6d0c96034cc594781458ed8cb8f

            # save middle cam information

            if math.isnan(command):
                f_m["targets"][cnt_middle] = float('nan')
            else:
                targets_m = np.array([command, analog_steer, analog_gas ])
                f_m["targets"][cnt_middle] = targets_m
            f_m["rgb"][cnt_middle,...] = rescaled_image
<<<<<<< HEAD
            f_m["rgb_original"][cnt_middle,...] = middle_image_original
=======
            f_m["rgb_original"][cnt_middle,...] = middle_image_origin
>>>>>>> 2c59425c59d0d6d0c96034cc594781458ed8cb8f

            if SHOW_CAM==True:
                cv2.putText( middle_image_original ,'{} {:.8f} {:.8f}'.format(f_m['targets'][cnt_middle][0],
                    f_m['targets'][cnt_middle][1],f_m['targets'][cnt_middle][2]),
                    (10,30), font, 0.5,(0,0,255),2)
                cv2.imshow('pic_m',middle_image_original)

            cnt_middle +=1


        if topics == '/group_right_cam/node_right_cam/image_raw/compressed':
            if cnt_right >= 200 or cnt_right == 0:
                f_r = h5py.File(right_destination + '{}_{}_{:05d}.h5'.format(location,'right',file_cnt_r),'w')
                dset = f_r.create_dataset("rgb_original", (200,480,640,3), np.uint8)
                dset = f_r.create_dataset("rgb", (200,88,200,3), np.uint8)
                dset = f_r.create_dataset("targets", (200,3), 'f')
                cnt_right = 0
                file_cnt_r += 1

            right_image_original = msg_to_mat(msg)
            rescaled_image = rescale(right_image_original)
            cmp_cmd = get_complementary_cmd(topics, command)

            if math.isnan(cmp_cmd):
                f_r["targets"][cnt_right] = float('nan')
            else:
                targets_r = np.array([cmp_cmd, analog_steer + STEERING_OFFSET, analog_gas ])
                f_r["targets"][cnt_right] = targets_r
            f_r["rgb"][cnt_right,...] = rescaled_image
            f_r["rgb_original"][cnt_right, ...] = right_image_original

            if SHOW_CAM==True:
                cv2.putText( right_image_original ,'{} {:.8f} {:.8f}'.format(f_r['targets'][cnt_right][0],
                    f_r['targets'][cnt_right][1],f_r['targets'][cnt_right][2]),
                    (10,30), font, 0.5,(0,0,255),2)
                cv2.imshow('pic_r',right_image_original)
            cnt_right +=1



        if topics == '/group_left_cam/node_left_cam/image_raw/compressed':
            if cnt_left >= 200 or cnt_left == 0:
                f_l = h5py.File(left_destination + '{}_{}_{:05d}.h5'.format(location,'left',file_cnt_l),'w')
                dset = f_l.create_dataset("rgb_original", (200,480,640,3), np.uint8)
                dset = f_l.create_dataset("rgb", (200,88,200,3), np.uint8)
                dset = f_l.create_dataset("targets", (200,3), 'f')
                cnt_left = 0
                file_cnt_l += 1

            left_image_original = msg_to_mat(msg)
            rescaled_image = rescale(left_image_original)

            cmp_cmd = get_complementary_cmd(topics, command)

            if math.isnan(cmp_cmd):
                f_l["targets"][cnt_left] = float('nan')
            else:
                targets_l = np.array([cmp_cmd, analog_steer - STEERING_OFFSET, analog_gas ])
                f_l["targets"][cnt_left] = targets_l
            f_l["rgb"][cnt_left,...] = rescaled_image
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
        os.remove(right_destination + '{}_right_{:05d}.h5'.format(location,file_cnt_m-1))
        os.remove(left_destination + '{}_left_{:05d}.h5'.format(location,file_cnt_m-1))
        print("removing:\n{}\n{}\n{}".format(
        '{}_middle_{:05d}.h5'.format(location,file_cnt_m-1),
        '{}_right_{:05d}.h5'.format(location,file_cnt_m-1),
        '{}_left_{:05d}.h5'.format(location,file_cnt_m-1)
        ))
    f_m.close

    # show random picture
    file_idx = random.randint(0, int(file_cnt_m-1))
    f_m = h5py.File(middle_destination + '{}_middle_{:05d}.h5'.format(location,file_idx),'r')
    f_l = h5py.File(left_destination + '{}_left_{:05d}.h5'.format(location,file_idx),'r')
    f_r = h5py.File(right_destination + '{}_right_{:05d}.h5'.format(location,file_idx),'r')


    pic_idx =random.randint(0,199)

    m_img = f_m['rgb_original'][pic_idx]
    cv2.putText( m_img ,'{}'.format(f_m['targets'][pic_idx]) ,(10,30), font, 0.5,(0,0,255),2)
    cv2.imshow('pic_m',m_img)

    r_img = f_r['rgb_original'][pic_idx]
    cv2.putText( r_img ,'{}'.format(f_r['targets'][pic_idx]) ,(10,30), font, 0.5,(0,0,255),2)
    cv2.imshow('pic_r',r_img)

    l_img = f_l['rgb_original'][pic_idx]
    cv2.putText( l_img ,'{}'.format(f_l['targets'][pic_idx]) ,(10,30), font, 0.5,(0,0,255),2)
    cv2.imshow('pic_l',l_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    bag.close()




if __name__=="__main__":
    main()
