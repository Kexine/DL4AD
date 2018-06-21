#!/usr/bin/env python3

import pandas as pd
import numpy as np


def main():
    import argparse
    import time
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile',
                        help='path to csv file with training and eval loss')

    args = parser.parse_args()


    csv_path = args.csvfile

    df = pd.read_csv(csv_path,
                     delim_whitespace=True)

    values_each_epoch = {}
    for i, row in enumerate(df.iterrows()):
        values = row[1]
        epoch = values['epoch']

        if epoch not in values_each_epoch:
            values_each_epoch[epoch] = 1
        else:
            values_each_epoch[epoch] += 1

    x_values = np.array([])
    for epoch, amount_values in values_each_epoch.items():
        x_values = np.append(x_values, np.linspace(epoch, epoch + 1, amount_values,
                                                   endpoint=False))

    plt.figure()
    plt.plot(x_values, df['train_loss'], label='training')
    plt.plot(x_values, df['eval_loss'], label='eval')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig('training.png')

    plt.show()


if __name__ == '__main__':
    main()
