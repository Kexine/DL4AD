#!/usr/bin/env python3

import argparse
from ImageHandling import ImageBrowser
from Extractor import H5Dataset, target_idx
import torch
from torchvision import datasets, transforms
import numpy as np
import pandas as pd

import command_input

try:
    import progressbar
except ModuleNotFoundError:
    progressbar = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(object):
    """ An agent for the imitation learing stuff."""
    def __init__(self, model_path, agent_type):
        if agent_type == "command_input":
            self.model = command_input.Net()  # .to(device)
        else:
            raise NotImplementedError()

        self.model.load_state_dict(torch.load(model_path))

        self.model.eval()

    def get_control(self, img, speed, command):
        with torch.no_grad():
            return self.model(img, speed, command)


if __name__=="__main__":
    # ---------- Argument parsing
    parser = argparse.ArgumentParser()

    parser.add_argument("type",
                        help="command|branched",
                        default="command")
    parser.add_argument("-m","--model",
                        help="The file from the trained model")
    parser.add_argument("-c", "--csv",
                        help="A CSV file with the target and prediction values")
    parser.add_argument("-d", "--dataset",
                        help="Directory of the test data",
                        default='../data/AgentHuman/SeqVal')
    args = parser.parse_args()

    dataset_path = args.dataset
    model_path = args.model

    # ---------- Initialization
    basic_transform = transforms.ToTensor()

    test_set = H5Dataset(root_dir = dataset_path,
                         transform= transforms.ToTensor())

    print("Loading model...")
    agent = Agent(model_path, 'command_input')

    # arrays for storing predictions and ground truth
    pred = np.empty((len(test_set), 2))
    truth = np.empty((len(test_set), 2))

    print("Applying model...")
    if progressbar is not None:
        bar = progressbar.ProgressBar(max_value = len(test_set))
    for idx, (data, target) in enumerate(test_set):
        data.unsqueeze_(0)
        target.unsqueeze_(0)

        speed = target[:,target_idx['speed']]
        command = target[:,target_idx['command']]

        pred[idx,:] = agent.get_control(data, speed, command).numpy()
        truth[idx,0] = target[:,target_idx['steer']].numpy()
        truth[idx,1] = target[:,target_idx['gas']].numpy()

        if progressbar is not None:
            bar.update(idx)

    mse = np.sum((pred - truth)**2,
                 axis=0)

    error = np.diff((truth, pred), axis = 0)
    mean = error.mean(axis=0)
    median = error[int(error.shape[0]/2), :]

    print("MSE:\t{}\nMean:\t{}\nMedian:\t{}".format(mse,
                                                    error.mean(axis=0),
                                                    median))

    print("storing to csv...")
    df = pd.DataFrame({'pred_steer': pred[:,0],
                       'pred_gas': pred[:,1],
                       'truth_steer': truth[:,0],
                       'truth_gas': truth[:,1]})
    df.to_csv(model_path.replace(".pt","") + ".csv",
              sep="\t",
              index=False)

    # browser = ImageBrowser([test_set])
