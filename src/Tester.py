#!/usr/bin/env python3

import argparse
from ImageHandling import ImageBrowser
from Extractor import H5Dataset, target_idx
import torch
from torchvision import datasets, transforms

import command_input

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(object):
    """ An agent for the imitation learing stuff."""
    def __init__(self, model_path, agent_type):
        if agent_type == "command_input":
            self.model = command_input.Net().to(device)
        else:
            raise NotImplementedError()

        self.model.load_state_dict(torch.load(model_path))

    def get_control(self, img, speed, command):
        with torch.no_grad():
            print(img.shape)
            return self.model(img, speed, command)


if __name__=="__main__":
    # ---------- Argument parsing
    parser = argparse.ArgumentParser()

    parser.add_argument("type",
                        help="command|branched",
                        default="command")
    parser.add_argument("model",
                        help="The file from the trained model")
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

    print("Applying model...")
    for data, target in test_set:
        speed = target[target_idx['speed']]
        command = target[target_idx['command']]

        print(agent.get_control(data, speed, command))

    browser = ImageBrowser([test_set])
