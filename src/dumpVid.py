#!/usr/bin/env python3

import argparse
# from ImageHandling import ImageBrowser
from Extractor import H5Dataset, target_idx, target_idx_raiscar
import torch
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import command_input, Branched, command_input_raiscar, Branched_raiscar
import cv2
from Tester import renderGas, renderSteering

try:
    import progressbar
except ModuleNotFoundError:
    progressbar = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

AGENT_COLOR = (0x00,0x34,0xd1)
HUMAN_COLOR = (0xd1,0x9c,0x00)


if __name__=="__main__":
    # ---------- Argument parsing
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--type",
                        help="Type of the Network: command_input or branched",
                        default="command_input")
    parser.add_argument("-d", "--dataset",
                        help="Directory of the data",
                        default='../data/AgentHuman/SeqVal')
    parser.add_argument("-s", "--saveto",
                        help="save path of the vid",
                        default="~/")
    args = parser.parse_args()

    net_type = args.type
    dataset_path = args.dataset
    save_path = args.saveto

    raiscar = not (net_type in ['command_input', 'branched'])

    test_set = H5Dataset(root_dir = dataset_path,
                         transform= transforms.ToTensor(), raiscar=raiscar)



    # arrays for storing predictions and ground truth
    length = len(test_set)
    pred = np.empty((length, 2))
    truth = np.empty((length, 2))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(save_path + 'out.avi',fourcc, 20.0, (640,480))

    print("dumping video...")
    if progressbar is not None:
        bar = progressbar.ProgressBar(max_value = len(test_set))
    with torch.no_grad():
        for idx in range(len(test_set)):
            if net_type in ['command_input', 'branched']:
                data, target = test_set[idx]
                orig_image = cv2.resize(data.numpy().transpose((1,2,0)),
                                        (640, 480))
            else:
                data, target, orig_image = test_set[idx]

            data, target = data.to(device), target.to(device)

            data.unsqueeze_(0)
            target.unsqueeze_(0)
            if net_type in ['command_input', 'branched']:
                command = target[:,target_idx['command']]
                speed = target[:,target_idx['speed']]

                truth[idx,0] = target[:,target_idx['steer']].cpu().numpy()
                truth[idx,1] = target[:,target_idx['gas']].cpu().numpy()
            else:
                command = target[:,target_idx_raiscar['command']]
                # pred[idx:] = model(data, command).cpu().numpy()

                truth[idx,0] = target[:,target_idx_raiscar['steer']].cpu().numpy()
                truth[idx,1] = target[:,target_idx_raiscar['gas']].cpu().numpy()

            # Also: get a video of the output!
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(orig_image,"Human", (15,470),font ,0.5,HUMAN_COLOR,2)
            cv2.putText(orig_image,"{}".format(command[0]), (120,470),font,0.8,HUMAN_COLOR,2)
            renderGas(orig_image, truth[idx][1], (20,450))

            renderSteering(orig_image, truth[idx,0], color=HUMAN_COLOR, pos= (320,480))

            # write original image to video
            out.write(orig_image)

            # Update the progressbar
            if progressbar is not None:
                bar.update(idx)

    # release videowriter
    out.release()

    cv2.destroyAllWindows()
