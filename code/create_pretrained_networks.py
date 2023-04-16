"""
The purpose of this file is to create pre-trained models of type 'network_type'
and pooling_function function 'pooling_function' with the given combination of hyperparameter settings.

created by Stephanie Henkel
"""

from training import create_pretrained_model
from helper_functions import set_seed
from Load_DINO_Split import load_cancer_csvs, HistopathologySlideBags
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from torch.optim import AdamW

from time import gmtime, strftime
import numpy as np
import torch
import itertools
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Give the most important directories.')
    parser.add_argument('dataset_path', type=str, help='A string giving the full path of the pretraining data')

    args = parser.parse_args()
    dataset_path = args.dataset_path

    set_seed()
    seed = 24

    aggregation = ["mean", "gated_attention", "hopfield_pooling"][0]
    network_type = ["FeedForward", "Hopfield"][0]
    logdir = f"runs/pretraining/{network_type}/{aggregation}"
    pretrain_params = {"network_type": network_type, "pooling_function": aggregation, "num_epochs": 150}

    # set hyperparameter search space
    pretrain_hyperparameters = {}
    pretrain_hyperparameters["batch_size"] = [16, 32, 64]
    pretrain_hyperparameters["lr_train"] = [1e-3, 1e-4, 5e-4]
    pretrain_hyperparameters["weight_decay"] = [0.0, 0.01, 0.001]
    pretrain_hyperparameters["instance_dropout"] = [0.0, 0.5]
    pretrain_hyperparameters["bag_dropout"] = [0.1]
    pretrain_hyperparameters["instance_hidden"] = [torch.Tensor([64]).long(), torch.Tensor([256]).long()]
    pretrain_hyperparameters["bag_hidden"] = [torch.Tensor([64, 32]).long(), torch.Tensor([128]).long()]

    if network_type == "Hopfield":
        pretrain_hyperparameters["d_model"] = [1024]
        pretrain_hyperparameters["num_heads"] = [1]
        pretrain_hyperparameters["scaling"] = [0.03, None]
        pretrain_hyperparameters["hopf_drop"] = [0.0]

    keys, values = zip(*pretrain_hyperparameters.items())
    hyperparams_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    num_hypersettings = len(hyperparams_permutations)

    # ------------------------  create train, val and test set  ------------------------
    dataset_df, label_dict = load_cancer_csvs(dataset_path, fine_tune_set=None)
    train_df, rest_df = train_test_split(dataset_df, stratify=dataset_df["label"].values, test_size=0.2, random_state=seed)
    val_df, test_df = train_test_split(rest_df, stratify=rest_df["label"].values, test_size=0.5, random_state=seed)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_set = HistopathologySlideBags(train_df, label_dict, max_bag_size=2000)
    val_set = HistopathologySlideBags(val_df, label_dict, max_bag_size=np.infty)
    test_set = HistopathologySlideBags(test_df, label_dict, max_bag_size=np.infty)

    # create_weights
    device = torch.device(r'cuda:0' if torch.cuda.is_available() else r'cpu')
    train_weights = torch.tensor(class_weight.compute_class_weight("balanced",
                                                                   classes=np.unique(train_df["label"].values),
                                                                   y=train_df["label"].values),
                                 dtype=torch.float32).to(device)

    for model_number, curr_hyperparams in enumerate(hyperparams_permutations):
        print(f"\n{'-' * 36} Hyperparameter setting {model_number + 1}/{num_hypersettings} {'-'*36}")

        # to hinder running the constellation with no instance hidden layer twice
        if curr_hyperparams["instance_hidden"] is None and curr_hyperparams["instance_dropout"] != 0:
            continue

        time_stamp = strftime("%Y%m%d_%H%M%S", gmtime())
        hyperparam_logdir = os.path.join(logdir, time_stamp)
        network_state_dict = create_pretrained_model(train_set, val_set, test_set, train_weights, pretrain_params,
                                                     curr_hyperparams, hyperparam_logdir, dataset_df["label"], AdamW,
                                                     pretrain_params["num_epochs"])
