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

try:
    import progressbar
except ModuleNotFoundError:
    progressbar = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

AGENT_COLOR = (0x00,0x34,0xd1)
HUMAN_COLOR = (0xd1,0x9c,0x00)

def renderGas(img,
              gas,
              pos,  # this is a x,y-tuple!
              max_height = 150,
              width = 20):
    # helper variable to see the middle of the bounding rectangle
    middle_y = pos[1] - int(max_height / 2)

    # draw the rectangle containing the gas value
    pt1 = pos
    pt2 = (pos[0] + width,
           pos[1] - max_height)
    cv2.rectangle(img, pt1, pt2, (0xFF, 0xFF, 0xFF), 1)

    # draw the gas rectangle inside the other one
    if gas > 0:
        clr = (0, 0, 0xFF)
    else:
        clr = (0xFF, 0, 0)

    pt1 = (pos[0] + 2,
           middle_y)
    pt2 = (pos[0] + width - 2,
           middle_y - int(gas * max_height / 2))
    cv2.rectangle(img, pt1, pt2, clr, -1)

    # additional line to see where gas is 0 0
    cv2.line(img,
             (pos[0] - 3, middle_y),
             (pos[0] + width + 3, middle_y),
             (0xFF, 0xFF, 0xFF))


def renderSteering(orig_image, raw_value,color, pos):
    cv2.circle(orig_image,(320,480), 68, (0xFF,0xFF,0xFF), 1)

    # TODO: map [-1,+1] joystick output to radiant [-pi, +pi]
    # negative is left, positive Right
    # raw_value = -1.0
    rad = raw_value*np.pi

    x, y = pos
    dx = (np.cos(rad - np.pi/2)) * 65
    dy = (np.sin(rad - np.pi/2)) * 65


    # print("old vlaue {}, new value {}".format(raw_value, rad))
    # print("dx {}, dy {}".format(dx,dy))

    cv2.arrowedLine(orig_image, (x,y), (x+int(dx),y+int(dy)), color, 2)


if __name__=="__main__":
    # ---------- Argument parsing
    parser = argparse.ArgumentParser()

    # parser.add_argument("type",
    #                     help="command|branched",
    #                     default="command")
    parser.add_argument("-t", "--type",
                        help="Type of the Network: command_input or branched",
                        default="command_input")
    parser.add_argument("-m","--model",
                        help="The file from the trained model")
    # parser.add_argument("-c", "--csv",
    #                     help="A CSV file with the target and prediction values")
    parser.add_argument("-d", "--dataset",
                        help="Directory of the test data",
                        default='../data/AgentHuman/SeqVal')
    args = parser.parse_args()

    net_type = args.type
    dataset_path = args.dataset
    model_path = args.model

    # just for safety so we don't accidently delete our model
    assert ".pt" in model_path, "Are you sure this is a model? It needs the extension '.pt'!"

    # ---------- Initialization
    raiscar = not (net_type in ['command_input', 'branched'])
    test_set = H5Dataset(root_dir = dataset_path,
                         transform= transforms.ToTensor(), raiscar=raiscar)

    print("Loading model...")
    if net_type == 'command_input':
        model = command_input.Net().to(device)
    elif net_type == 'branched':
        model = Branched.Net().to(device)
    elif net_type == 'command_input_raiscar':
        model = command_input_raiscar.Net().to(device)
    elif net_type == 'branched_raiscar':
        model = Branched_raiscar.Net().to(device)
    else:
        print("Please specify one of these models: \
        command_input|branched|command_input_raiscar|branched_raiscar")
        exit()

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # arrays for storing predictions and ground truth
    length = len(test_set)
    pred = np.empty((length, 2))
    truth = np.empty((length, 2))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(model_path.replace(".pt", ".avi"),fourcc, 20.0, (640,480))

    print("Applying model...")
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
                pred[idx,:] = model(data, speed, command).cpu().numpy()

                truth[idx,0] = target[:,target_idx['steer']].cpu().numpy()
                truth[idx,1] = target[:,target_idx['gas']].cpu().numpy()
            else:
                command = target[:,target_idx_raiscar['command']]
                pred[idx:] = model(data, command).cpu().numpy()

                truth[idx,0] = target[:,target_idx_raiscar['steer']].cpu().numpy()
                truth[idx,1] = target[:,target_idx_raiscar['gas']].cpu().numpy()

            # Also: get a video of the output!
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(orig_image,"Human", (15,470),font ,0.5,HUMAN_COLOR,2)
            cv2.putText(orig_image,"Agent", (575,470),font ,0.5,AGENT_COLOR,2)
            cv2.putText(orig_image,"{}".format(command[0]), (120,470),font,0.8,HUMAN_COLOR,2)
            renderGas(orig_image, truth[idx][1], (20,450))
            renderGas(orig_image, pred[idx][1], (580,450))

            renderSteering(orig_image, truth[idx,0], color=HUMAN_COLOR, pos= (320,480))
            renderSteering(orig_image, pred[idx,0], color=AGENT_COLOR, pos = (320,480))

            # write original image to video
            out.write(orig_image)

            # Update the progressbar
            if progressbar is not None:
                bar.update(idx)

    # release videowriter
    out.release()

    cv2.destroyAllWindows()

    error = pred - truth

    mse = np.mean(error ** 2, axis=0)

    mean = error.mean(axis=0)
    median = error[int(error.shape[0]/2), :]

    # calculate R-squared error
    non_zero_indices = np.where(truth != 0)
    R_squared = np.ones(truth.shape[1]) - np.mean(error[np.where(truth != 0)]**2 / truth[np.where(truth!=0)]**2)

    print("truth norm: {}".format(np.mean(truth**2, axis=0)))

    print("R²: {}".format(R_squared))

    print("MSE:\t{}\nMean error:\t{}\nMedian error:\t{}".format(mse,
                                                                error.mean(axis=0),
                                                                median))

    print("storing to csv...")
    df = pd.DataFrame({'pred_steer': pred[:,0],
                       'pred_gas': pred[:,1],
                       'truth_steer': truth[:,0],
                       'truth_gas': truth[:,1],
                       'abs_error_steer': error[:,0],
                       'abs_error_gas': error[:,1],
                       'rel_error_steer': error[:,0]/truth[:,0],
                       'rel_error_gas': error[:,1]/truth[:,1]})
    df.to_csv(model_path.replace(".pt", "_TEST.csv"),
              sep="\t",
              index=False)

    # browser = ImageBrowser([test_set])
    # browser.show()
