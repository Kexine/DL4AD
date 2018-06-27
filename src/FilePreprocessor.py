#!/usr/bin/env python3

import argparse
import random
import os
import numpy as np





def move_set(file_list, source, destination):
    for idx,val in enumerate(file_list):
        src_path = source + '/' + val
        dst_path = destination + '/' + val
        print(" {} -> {}".format(src_path, dst_path))
        os.rename(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data",
                        help="path of data to split into validation and training set",
                        )
    parser.add_argument("-f", "--fraction",
                        help="Train/Val fraction",
                        default = 0.8)

    args = parser.parse_args()
    data_path = args.data
    fraction = float(args.fraction)

    destination_dir_train = data_path + '/train'
    destination_dir_val = data_path + '/val'

    # get file names
    file_names = sorted(os.listdir(data_path))
    # get total file list length
    total_length = len(file_names)

    # get length of training set
    train_length = int(fraction * total_length)
    # get length of validation set
    val_length = total_length - train_length

    # get always the same split
    np.random.seed(42)
    np.random.shuffle(file_names)

    train_indices = np.arange(train_length)

    val_start = train_length

    train_set = np.take(file_names, train_indices)

    val_indices = np.arange(val_start, total_length)

    val_set = np.take(file_names, val_indices)

    print("-------------------------------------------------------------")
    print("Total amount of files: {}\t| Split fraction: {}".format(total_length, fraction))
    print("Traning files: {}\t\t| Validation files: {} ".format(train_length,val_length))
    print("-------------------------------------------------------------")


    if not os.path.isdir(destination_dir_train):
        print("Creating Training directory..." )
        os.mkdir(destination_dir_train)
        print("Moving training data into directory: {}".format(destination_dir_train))
        move_set(train_set, data_path, destination_dir_train)

    if not os.path.isdir(destination_dir_val):
        print("Creating Validation directory..." )
        os.mkdir(destination_dir_val)
        print("Moving validation data into directory: {}".format(destination_dir_val))
        move_set(val_set, data_path, destination_dir_val)



if __name__ == '__main__':
    main()
