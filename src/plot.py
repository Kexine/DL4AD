#!/usr/bin/env python3

import pandas as pd
import numpy as np

def prettify_csv(train_file,
                 eval_file,
                 eval_rate):
    train_df = pd.read_csv(train_file,
                           sep='\t')
    eval_df = pd.read_csv(eval_file,
                          sep='\t')

    data_key = train_df.keys()[1]

    # train_indices, train_data = train_df[data_key]

    train_indices = np.append([0], [i + 1 for i in train_df[data_key].index])
    train_data = np.append([data_key], train_df[data_key].values)

    eval_key = eval_df.keys()[1]
    epoch_key = eval_df.keys()[0]

    eval_indices = [eval_rate*(i + 1) for i in eval_df[eval_key].index]
    eval_data = np.append([eval_key], eval_df[eval_key].values)
    eval_epoch = np.append([epoch_key], eval_df[epoch_key].values)

    return {'train_idx': train_indices,
            'train_data': train_data,
            'eval_idx': eval_indices,
            'eval_epoch': eval_epoch,
            'eval_data': eval_data}

def main():
    import argparse
    import time
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--traincsv',
                        help='path to csv file with training loss')
    parser.add_argument('-v', '--valcsv',
                        help='path to csv file with validation loss')

    args = parser.parse_args()


    train_csv_path = args.traincsv
    eval_csv_path = args.valcsv

    data = prettify_csv(train_csv_path, eval_csv_path, 0.8)
    print(data['eval_epoch'])

    plt.figure()
    plt.plot(data['train_data'], label='training')
    plt.xlabel('Index')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig('training.png')

    plt.figure()
    plt.plot(data['eval_data'], label='validation', c='r')
    plt.xlabel('Index')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig('validation.png')

    plt.show()









if __name__ == '__main__':
    main()
