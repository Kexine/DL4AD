#!/usr/bin/env python3

import argparse
# from ImageHandling import ImageBrowser
from Extractor import H5Dataset, target_idx, target_idx_raiscar
import torch
from torchvision import datasets, transforms
import numpy as np
import pandas as pd

import command_input, Branched, command_input_raiscar, Branched_raiscar

try:
    import progressbar
except ModuleNotFoundError:
    progressbar = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    # ---------- Initialization
    test_set = H5Dataset(root_dir = dataset_path,
                         transform= transforms.ToTensor())

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

    print("Applying model...")
    if progressbar is not None:
        bar = progressbar.ProgressBar(max_value = len(test_set))
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_set):
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

            if progressbar is not None:
                bar.update(idx)

    truth_prime = truth - np.mean(truth, axis=0)
    # truth_prime[np.where(truth == 0)] = np.finfo(float).eps
    error = pred - truth

    mse = np.mean(error ** 2, axis=0)

    mean = error.mean(axis=0)
    median = error[int(error.shape[0]/2), :]

    # calculate R-squared error
    non_zero_indices = np.where(truth != 0)
    R_squared = np.ones(truth.shape[1]) - np.mean(error[np.where(truth != 0)]**2 / truth[np.where(truth!=0)]**2)

    print("truth norm: {}".format(np.mean(truth_prime**2, axis=0)))

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
