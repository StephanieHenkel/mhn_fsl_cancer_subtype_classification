"""
This file contains a class and functions to write the results into tensorboard files.

created by Stephanie Henkel
"""

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import os
import json
from pathlib import Path
from time import gmtime, strftime

from helper_functions import hyperparameter_dict_from_info_file


class TensorboardWriter:
    def __init__(self, logdir, hyperparams):
        self.base_logdir = logdir
        self.writer = None
        self.hparams_writer = None
        self.results = {}
        self.hyperparams = hyperparams

    def create_writer(self, logdir=None):
        if logdir is None:
            save_dir = self.base_logdir
        else:
            save_dir = os.path.join(self.base_logdir, logdir)
        self.writer = SummaryWriter(save_dir)
        self.save_model_info(save_dir)

    def save_model_info(self, model_dir):
        str_dict = dict(zip(self.hyperparams.keys(), [str(val) for val in self.hyperparams.values()]))
        with open(os.path.join(model_dir, 'model_info.txt'), 'w') as f:
            f.write(json.dumps(str_dict))

    def create_hyperparams_writer(self, logdir):
        save_dir = os.path.join(self.base_logdir, logdir)
        self.hparams_writer = SummaryWriter(save_dir)

    def add_writer_results(self, metric_values, step):
        for metric in metric_values.keys():
            self.writer.add_scalar(metric, metric_values[metric], step)

    def add_hyperparams_writer_results(self, metric_values):
        for metric in metric_values.keys():
            if metric.split("/")[0] in ["loss", "accuracy", "balanced_accuracy", "roc_auc", "prec", "recall", "f1_score",
                                        "overall_balanced_accuracy"]:
                if metric in self.results.keys():
                    self.results[metric].append(metric_values[metric])
                else:
                    self.results[metric] = [metric_values[metric]]

    def close_writer(self):
        self.writer.flush()
        self.writer.close()

    def close_hparams_writer(self, hyperparams):
        if self.hparams_writer is not None:
            metric_mean_std = {}
            for metric in self.results.keys():
                metric_mean_std[f"mean_{metric}"] = np.mean(self.results[metric])
                metric_mean_std[f"std_{metric}"] = np.std(self.results[metric])

            self.hparams_writer.add_hparams(hyperparams, metric_mean_std)

            self.hparams_writer.flush()
            self.hparams_writer.close()


def create_writer_dict_for_model_grid(logdir, model_directories, hyperparams_permutations, query_epochs):
    writer_dict = {}
    query_epochs_hyperparams = {}

    for curr_model_dir in model_directories:
        outer_curr_model_dir = os.path.join(logdir, Path(curr_model_dir).parts[-3])
        pretrain_hyperparams = hyperparameter_dict_from_info_file(
                    os.path.join(Path(curr_model_dir).parents[1], "model_info.txt"))

        writer_dict[curr_model_dir] = {}
        query_epochs_hyperparams[curr_model_dir] = {}
        for hparam_idx in range(len(hyperparams_permutations)):
            writer_dict[curr_model_dir][hparam_idx] = {}
            query_epochs_hyperparams[curr_model_dir][hparam_idx] = {}

            for ep in query_epochs:
                query_epochs_hyperparams[curr_model_dir][hparam_idx][ep] = pretrain_hyperparams.copy()
                query_epochs_hyperparams[curr_model_dir][hparam_idx][ep].update(hyperparams_permutations[hparam_idx])
                query_epochs_hyperparams[curr_model_dir][hparam_idx][ep]["ft_epoch_stop"] = int(ep + 1)

                hyperparam_logdir = os.path.join(outer_curr_model_dir, strftime("%Y%m%d_%H%M%S", gmtime()), f"epoch_{ep}")
                writer_dict[curr_model_dir][hparam_idx][ep] = TensorboardWriter(hyperparam_logdir,
                                                                                query_epochs_hyperparams[curr_model_dir][hparam_idx][ep])
                writer_dict[curr_model_dir][hparam_idx][ep].create_hyperparams_writer(logdir="cv_score")

    return writer_dict, query_epochs_hyperparams


def close_writer_dict_for_model_grid(writer_dict, query_epochs_hyperparams):
    for curr_model_dir in writer_dict.keys():
        for hparam_idx in writer_dict[curr_model_dir].keys():
            for ep in writer_dict[curr_model_dir][hparam_idx].keys():
                writer = writer_dict[curr_model_dir][hparam_idx][ep]
                writer.close_hparams_writer(query_epochs_hyperparams[curr_model_dir][hparam_idx][ep])