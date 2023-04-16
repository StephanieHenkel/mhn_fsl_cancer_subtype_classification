"""
The purpose of this file is to run nested cross-validation for a network type 'network_type'
and a pooling function 'pooling_function' with the given combination of hyperparameter settings and
chosen pretrained models.

created by Stephanie Henkel
"""

from training import nested_cross_validate
from torch.optim import AdamW
import argparse

from helper_functions import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Give the most important directories.')
    parser.add_argument('dataset_path_pretrain', type=str, help='A string giving the full path of the pretraining data')
    parser.add_argument('dataset_path_finetune', type=str, help='A string giving the full path of the fine-tuning data')
    parser.add_argument('pretrained_models_basepath', type=str,
                        help='A string giving the full path of the pretrained models')

    args = parser.parse_args()
    dataset_path_pretrain = args.dataset_path_pretrain
    dataset_path_finetune = args.dataset_path_finetune
    pretrained_models_basepath = args.pretrained_models_basepath
    set_seed()

    aggregation = ["mean", "gated_attention", "hopfield_pooling"][0]
    network_type = ["FeedForward", "Hopfield"][1]

    hyperparams = {}

    params = {"network_type": network_type, "pooling_function": aggregation, "n_shots": 10, "n_tasks": 30}
    params["epochs_fine-tune"] = [0 if params["n_shots"] == 0 else 120][0]

    if params["n_shots"] != 0:
        hyperparams["ft_batch_size"] = [4, 16]
        hyperparams["ft_learning_rate"] = [1e-4]

        hyperparams["ft_weight_decay"] = [0.01]

        if network_type == "Hopfield":
            hyperparams["hopfield_droput"] = [0.0, 0.5]

    results = nested_cross_validate(outer_folds=5, inner_folds=3, params=params, hyperparams=hyperparams,
                                    optimizer=AdamW, dataset_path_pretrain=dataset_path_pretrain,
                                    dataset_path_finetune=dataset_path_finetune,
                                    pretrained_models_basepath=pretrained_models_basepath, seed=24)
