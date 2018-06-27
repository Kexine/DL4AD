#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Learning in Autonomous Driving
Project 4: Exercise sheet 3, Task 1
Michael Floßmann, Kshitij Sirohi, Hendrik Vloet
"""

# basic python stuff
import argparse
import pandas as pd
import os

# pytorch stuff
import torch
import torch.optim as optim
from torchvision import transforms

# custom stuff
from Extractor import H5Dataset, target_idx
from customTransforms import *
from CustomLoss import WeightedMSELoss

import command_input
import Branched

net_types = ['command_input',
             'branched',
             'command_input_raiscar',
             'branched_raiscar']

try:
    import progressbar
    progress_widgets = [progressbar.widgets.DynamicMessage('loss'),
                        ' ', progressbar.widgets.Percentage(),
                        ' ', progressbar.widgets.Bar(),
                        ' ', progressbar.widgets.Timer(),
                        ' ', progressbar.widgets.AdaptiveETA(samples = 200),
                        ' ', progressbar.widgets.CurrentTime()]
except ModuleNotFoundError:
    progressbar = None
    print("Progressbar not found. Please consider installing the module Progressbar2 for sweet-ass progressbars")


def load_model(model, model_path):
    '''
    Check if a pre trained model exists and load it if found
    '''
    print("Checking if some model exists... ", end="")

    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model was found and loaded!")
    else:
        print("No model found, starting training with new model!")


def get_output(net_type,
               model,
               data,
               target):
    if net_type == 'command_input':
        return model(data,
                     target[:, target_idx['speed']],
                     target[:, target_idx['command']])
    elif net_type == 'branched':
        return  model(data,
                      target[:,target_idx['speed']])
    elif net_type == 'command_input_raiscar':
        raise NotImplementedError
    elif net_type == 'branched_raiscar':
        raise NotImplementedError


def get_target(net_type,
               target):
    if net_type == 'command_input':
        return target[:,[target_idx['steer'],
                         target_idx['gas']]]
    elif net_type == 'branched':
        
    elif net_type == 'command_input_raiscar':
        raise NotImplementedError
    elif net_type == 'branched_raiscar':
        raise NotImplementedError


def evaluate(, net_type,
             model,
             eval_loader,
             loss_function,
             weights):
    model.eval()
    with torch.no_grad():
        loss = 0
        model = model.to(device)

        if progressbar is not None:
            eval_bar = progressbar.ProgressBar(max_value = len(eval_loader),
                                               widgets = progress_widgets)

        # actual evaluation
        for eval_idx, (data, target) in enumerate(eval_loader):
            data = data.to(device)
            target = target.to(device)

            output = get_output(net_type, model, data, target)

            output_target = model.extract_output(target)

            current_loss = loss_function(output.double(),
                                         output_target.double()).item()
            loss += current_loss

            if progressbar is not None:
                eval_bar.update(eval_idx, loss=loss/(eval_idx + 1))
            else:
                print("\rEvaluation in progress {:.0f}%/100%".format((eval_idx+1)/len(eval_loader)*100),
                      end="",
                      flush=True)

    avg_loss = loss/len(eval_loader)
    return avg_loss


def main(net_type,
         model_path,
         traindata_path,
         valdata_path,
         eval_rate,
         batch_size,
         no_transforms,
         amount_epochs):
    # -------------------- Get the cuda device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device: {}".format(device))

    # -------------------- Prepare the datasets
    basic_transforms = transforms.Compose([JustNormalize(std=1)])  # TODO: maybe we don't need the Compose()?

    # the transformations with the magnitudes to apply to the train data
    if no_transforms:
        train_transforms = basic_transforms
    else:
        train_transforms = RandomApplyFromList([ContrastNBrightness(1.5,0.5),
                                                GaussianBlur(1.5),
                                                SaltNPepper(0.1),
                                                GaussianNoise(0, 0.1),
                                                RegionDropout((10, 10),10)],
                                               normalize = True,
                                               std=1)

    train_set = H5Dataset(root_dir = traindata_path,
                          transform = train_transforms)
    eval_set = H5Dataset(root_dir = valdata_path,
                         transform = basic_transforms)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size = batch_size,
                                               shuffle=True,
                                               pin_memory=False,
                                               num_workers=4,
                                               drop_last=True)

    eval_loader = torch.utils.data.DataLoader(eval_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4,
                                              drop_last=True)

    # -------------------- Prepare the model

    if net_type == 'command_input':
        model = command_input.Net().to(device)
    elif net_type == 'branched':
        model = Branched.Net().to(device)
    elif net_type == 'command_input_raiscar':
        raise NotImplementedError
    elif net_type == 'branched_raiscar':
        raise NotImplementedError

    load_model(model, model_path)

    # -------------------- Prepare the optimizer + loss function

    # define the weights
    weights = torch.eye(2)
    weights[0,0] = 0.75
    weights[1,1] = 0.25

    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    loss_function = WeightedMSELoss(weights.to(device))

    # -------------------- Prepare the dataframe for logging the loss
    loss_df = pd.DataFrame([], columns=['train_loss', 'eval_loss', 'epoch'])

    # -------------------- A bit of flavour status
    print(100*"-")
    print("{:<50}".format("Batch Size: " + str(batch_size)), end="")
    print("{:>50}".format("Eval Rate: " + str(eval_rate)))
    print("{:<50}".format(str(len(train_set)) + " Training Samples"), end="")
    print("{:>50}".format(str(len(eval_set)) + " Eval Samples"))
    print("{:<50}".format(str(len(train_loader)) + " Training Batches"), end="")
    print("{:>50}".format(str(len(eval_loader)) + " Eval Batches"))
    print(100*"-", end="\n\n\n")

    # -------------------- Start actual training
    for epoch in range(1, amount_epochs + 1):
        try:
            print('{:#^100}'.format(' Epoch ' + str(epoch) + ' '))

            if progressbar is not None:
                bar = progressbar.ProgressBar(max_value = len(train_loader),
                                              widgets = progress_widgets)

            # initialize the train loss calculation
            train_loss = 0
            amount_trains = 0

            # -------------------- Start training for this epoch
            for batch_idx, (data, target) in enumerate(train_loader):
                # Move the input and target data on the GPU
                data = data.to(device)
                target = target.to(device)

                # Zero out gradients from previous step
                optimizer.zero_grad()

                output = get_output(net_type, model, data, target)

                # Get the target outputs that we're training on
                output_target = target[:,[target_idx['steer'],
                                          target_idx['gas']]]

                loss = loss_function(output,
                                     output_target)  # DONE: weights are now in the initializer

                # Backward pass (gradient computation)
                loss.backward()
                # Adjusting the parameters according to the loss function
                optimizer.step()

                # store the training loss
                train_loss += loss.item()
                amount_trains += 1

                if progressbar is not None:
                    bar.update(batch_idx, loss=train_loss / (amount_trains))

                # -------------------- Evaluate
                if batch_idx & eval_rate == eval_rate - 1:
                    print("\n\n{:-^100}".format("Evaluation"))
                    eval_loss = evaluate(model,
                                         eval_loader,
                                         loss_function, weights)
                    print(("{:-^100}".format("Eval loss: {:.5f}".format(eval_loss))))

                    model.train()

                    # -------------------- Store the loss
                    loss_df = loss_df.append(pd.DataFrame([[train_loss/amount_trains, eval_loss, epoch]],
                                                          columns=['train_loss', 'eval_loss', 'epoch']),
                                             ignore_index=True)

                    train_loss = 0
                    amount_trains = 0

                    # # ---------- Also, save the model here
                    # save_model(model, model_path)

        except KeyboardInterrupt:
            print("Abort detected! Saving the model and exiting (Please don't hit C-c again >.<)")
            save_model(model, model_path)
            with open(model_path.replace(".pt", "_loss.csv"), 'w') as f:
                loss_df.to_csv(f, sep="\t", header=True, index=True)
                # TODO: can also be done with appending instead of overwriting
            break

        save_model(model, model_path)

        with open(model_path.replace(".pt", "_loss.csv"), 'w') as f:
            loss_df.to_csv(f, sep="\t", header=True, index=True)
            # TODO: can also be done with appending instead of overwriting


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("net_type",
                        help="What kind of Net do you want to train?\n" + \
                        "|".join(net_types))
    parser.add_argument("-m", "--model",
                        help="A (existing?) model file to store to",
                        default=None)
    parser.add_argument("-t", "--train",
                        help="Directory of the train data",
                        default='../data/AgentHuman/SeqTrain/train')
    parser.add_argument("-v", "--val",
                        help="Directory of the validation data",
                        default='../data/AgentHuman/SeqTrain/val')
    parser.add_argument("-r", "--evalrate",
                        help="Evaluate every [N] training batches",
                        default=263,  # this is basically 10 evals each epoch
                        type=int)
    parser.add_argument("-b", "--batchsize",
                        help="Size of batches",
                        default=200,
                        type=int)
    parser.add_argument("-e", "--epochs",
                        help="Train for how many epochs? (15)",
                        type=int,
                        default = 15)
    parser.add_argument("--no_transforms",
                        help="Don't apply any transforms on the test set.",
                        action='store_true')


    args = parser.parse_args()

    net_type = args.net_type
    model_path = args.model
    traindata_path = args.train
    valdata_path = args.val
    eval_rate = args.evalrate
    batch_size = args.batchsize
    no_transforms = args.no_transforms
    amount_epochs = args.epochs

    # create the default model path based on 
    if model_path is None:
        model_path = dt.datetime.now().strftime("../model/%Y-%m-%d_%H%M{}.pt".format(net_type))

    assert net_type in net_types, "Please choose a proper Net!"

    main(net_type,
         model_path,
         traindata_path,
         valdata_path,
         eval_rate,
         batch_size,
         no_transforms,
         amount_epochs)
