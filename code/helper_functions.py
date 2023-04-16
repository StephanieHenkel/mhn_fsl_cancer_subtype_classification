"""
This file contains helper functions, which are used in the training file.

created by Stephanie Henkel
"""

import re
import os
import random
import json
import copy

import torch
import numpy as np
import pandas as pd


def create_query_epochs(max_epochs):
    """
    Creates a numpy array of epochs for which the fine-tuned model is evaluated. In this way a number of epochs
    until which is trained can be seen as a hyperparameter.
    Args:
        max_epochs:max number of epochs

    Returns: numpy array of epochs

    """
    if max_epochs < 10:  # e.g. for 0-shot or when i wanted to test only one really low number of epochs
        epoch_splits = np.array([max_epochs])
    else:
        epoch_splits = np.linspace(0, max_epochs, 3, endpoint=False, dtype=np.int32)[1:]
        epoch_splits = np.append(epoch_splits, max_epochs)
    return epoch_splits - 1


def hyperparameter_dict_from_info_file(model_info_dir):
    """
    Creates a dictionary with hyperparameters as keys and corresponding values from a file,
    which is saved under model_info_dir.
    Args:
        model_info_dir: directory of the file which gives all the hyperparameter information of the model

    Returns: hyperparameter dictionary

    """
    with open(model_info_dir, 'r') as f:
        hyperparameter_dict = json.loads(f.read())

    # correct the type of the values in the dictionary
    new_dict = {}
    for key, value_str in hyperparameter_dict.items():
        if "None" == value_str:
            value = None
        elif "tensor" in value_str:
            value = torch.tensor([int(v) for v in re.findall(r'\d+', value_str)])
        elif "." in value_str:
            value = float(value_str)
        else:
            value = int(value_str)
        new_dict[key] = value

    return new_dict


def save_results_to_csv(logdir, results):
    """
    Saves the given results as well as their mean and standard deviation to a csv file to the given directory.
    Args:
        logdir: directory where the csv file is saved to
        results:

    Returns:

    """
    final_performances_df = pd.DataFrame({key: pd.Series(value) for key, value in
                                          results.items()})
    mean_per_metric = final_performances_df.mean(axis=0)
    mean_per_metric.name = "mean"
    std_per_metric = final_performances_df.std(axis=0)
    std_per_metric.name = "std"
    final_performances_df = pd.concat([final_performances_df, pd.DataFrame(mean_per_metric).T])
    final_performances_df = pd.concat([final_performances_df, pd.DataFrame(std_per_metric).T])
    final_performances_df.to_csv(os.path.join(logdir, 'results.csv'), index=True)

    return


def set_seed(seed: int = 42):
    """
    Set seed for all underlying (pseudo) random number sources.

    :param seed: seed to be used
    :return: None
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_net = None

    def __call__(self, val_loss, network):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_net = copy.deepcopy(network.state_dict())
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_net = copy.deepcopy(network.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"Early Stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
