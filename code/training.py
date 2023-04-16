"""
This file contains all functions used for training and evaluation for pretraining as well as for fine-tuning.

created by Stephanie Henkel
"""

import os
import gc
import itertools
import glob
from time import gmtime, strftime
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold

from Models import FeedForward, HistoHopfield, calculate_metrics
from Load_DINO_Split import collate, load_cancer_csvs, create_support_dataloader, create_query_dataloader
from helper_functions import create_query_epochs, save_results_to_csv, hyperparameter_dict_from_info_file, EarlyStopping
from Tensorboard_Logging import TensorboardWriter, create_writer_dict_for_model_grid, close_writer_dict_for_model_grid
sns.set()


def iterate_dataset(network, data_loader, log_writer, step=None, optimizer=None, loss_weight=None, name=None):
    """
    This function iterates over the given data loader with the given network and evaluates the predictions.
    Args:
        network: the network which is trained or evaluated
        data_loader: the data loader, which gives the data in a for loop
        log_writer: writer to log the predictions to tensorboard
        step: - pre-training: the corresponding epoch in which this is performed
               - fine-tuning: the corresponding task number
        optimizer:
        loss_weight: required for weighted loss
        name: to name the saved metrics corresponding to what is done

    Returns: performance, metrics
    """
    device = torch.device(r'cuda:0' if torch.cuda.is_available() else r'cpu')
    train = optimizer is not None
    if train:
        network.train()

    else:
        network.eval()
        if isinstance(network, HistoHopfield):
            network.precompute_keys()

    loss_list = []
    y_pred_list = []
    target_list = []

    for x, y, bag_lengths, batch_indices in data_loader:
        x, y = x.to(device=device), y.to(device=device)
        bag_lengths = bag_lengths.tolist()

        network_input = (x, bag_lengths)
        if isinstance(network, HistoHopfield):
            network_input = (x, bag_lengths, batch_indices)

        with torch.set_grad_enabled(train):
            y_pred = network.forward(network_input)
            loss = CrossEntropyLoss(weight=loss_weight)(y_pred, y)

            loss_list.append(loss.detach().cpu().item())
            y_pred_list.append(y_pred.detach().cpu())
            target_list.append(y.cpu())

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 0.2)  # gradient clipping
            optimizer.step()

    # Compute performance measures of current model
    cat_preds = torch.cat(y_pred_list, dim=0)
    cat_targets = torch.cat(target_list, dim=0)

    performance = calculate_metrics(cat_preds, cat_targets)
    performance["loss"] = np.mean(loss_list)

    metrics = {}
    for metric in performance.keys():
        metrics[f"{metric}/{name}"] = performance[metric]

    if log_writer is not None:
        log_writer.add_writer_results(metrics, step)

    return performance, metrics


def run_training(network: Module, optimizer, data_loader_train, data_loader_val=None, num_epochs: int = 1,
                 writer=None, model_save_dir=None, train_weights=None):
    """
    Train the given network by gradient descent using backpropagation and early stopping on the validation data.
    Args:
        network: either the Hopfield or the feedforward network
        optimizer: optimizer to train the network
        data_loader_train: dataloader for training data
        data_loader_val: dataloader for validation data
        num_epochs: max number of epochs to train
        writer: tensorboard writer to log the evaluation results
        model_save_dir: directory where to save the final trained model to
        train_weights: weights for weighted loss

    Returns: the best trained model given by early stopping

    """
    pbar_epoch = tqdm(range(num_epochs))
    early_stopping = EarlyStopping(patience=20, verbose=False)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    pbar_epoch.set_description("TRAIN Epoch -- Accuracy: , Loss:  |||| VAL Epoch -- Accuracy: , Loss: ")

    best_loss = np.infty
    best_train_performance = None
    best_val_performance = None

    for epoch in pbar_epoch:
        # Train network.
        performance_train, train_metrics = iterate_dataset(network, data_loader_train, writer, epoch, optimizer,
                                                           train_weights, "training")

        # Validate for early stopping
        performance_val, eval_metrics = iterate_dataset(network, data_loader_val, writer, epoch, None, train_weights,
                                                        "evaluation")

        pbar_epoch.set_description(
            f"TRAIN Epoch -- Accuracy: {round(performance_train['balanced_accuracy'], 3)}, "
            f"Loss: {round(performance_train['loss'], 3)} |||| "
            f"VAL Epoch -- Accuracy: {round(performance_val['balanced_accuracy'], 3)}, "
            f"Loss: {round(performance_val['loss'], 3)}")

        scheduler.step(performance_val["loss"])
        early_stopping(performance_val["loss"], network=network)

        if performance_val["loss"] < best_loss:
            best_loss = performance_val["loss"]
            best_train_performance = train_metrics
            best_val_performance = eval_metrics

        if early_stopping.early_stop:
            print(f"Early stopping at Epoch {epoch + 1}")
            break

    writer.close_writer()
    writer.add_hyperparams_writer_results(best_train_performance)
    writer.add_hyperparams_writer_results(best_val_performance)

    if not os.path.exists(os.path.dirname(model_save_dir)):
        os.makedirs(os.path.dirname(model_save_dir))
    torch.save(early_stopping.best_net, model_save_dir)

    return early_stopping.best_net


def create_pretrained_model(train_set, val_set, test_set, train_weights, params, hyperparams, logdir, train_labels,
                            optimizer, num_epochs):
    """
    This function is used to create the pretrained model by training a specific network type with the given
    hyperparameters and saving the trained model and its evaluation results to a logdir.
    Args:
        train_set: training dataset
        val_set: validation dataset for early stopping
        test_set: test dataset for final evaluation
        train_weights: weights for weighted loss
        params: parameter
        hyperparams: hyperparameter
        logdir: directory to log the evaluation results and the model
        train_labels: labels of the training data
        optimizer:
        num_epochs: max number of epochs to train the model

    Returns: state dictionary of final trained model

    """
    data_loader_train = DataLoader(train_set, batch_size=hyperparams["batch_size"], shuffle=True, drop_last=True,
                                   collate_fn=collate)
    data_loader_val = DataLoader(val_set, batch_size=128, shuffle=False, collate_fn=collate)
    data_loader_test = DataLoader(test_set, batch_size=128, shuffle=False, collate_fn=collate)

    train_writer = TensorboardWriter(logdir, hyperparams)
    train_writer.create_writer("training")
    train_writer.create_hyperparams_writer(os.path.join("training", "cv_score"))

    test_writer = TensorboardWriter(logdir, hyperparams)
    test_writer.create_writer("testing")
    test_writer.create_hyperparams_writer(os.path.join("testing", "cv_score"))

    # ------------------------  Init the model ------------------------
    network = init_model(params, hyperparams, len(train_labels.unique()))
    network.label2type = train_set.label2type
    if params["network_type"] == "Hopfield":
        network.set_fixed_stored_df(train_set.data, 100)

    optimizer_train = optimizer(network.parameters(), lr=hyperparams["lr_train"],
                                weight_decay=hyperparams["weight_decay"])

    # ------------------------  Training stage ------------------------
    model_save_dir = os.path.join(logdir, "model", "train_model.pth")
    network_state_dict = run_training(network=network, optimizer=optimizer_train, data_loader_train=data_loader_train,
                                      data_loader_val=data_loader_val, num_epochs=num_epochs, writer=train_writer,
                                      model_save_dir=model_save_dir, train_weights=train_weights)

    train_writer.close_hparams_writer(hyperparams)

    # test network
    network.load_state_dict(network_state_dict)
    performance_test, test_metrics = iterate_dataset(network, data_loader_test, test_writer, 1, None, train_weights,
                                                     "test")

    test_writer.add_hyperparams_writer_results(test_metrics)
    test_writer.close_writer()
    test_writer.close_hparams_writer(hyperparams)

    return network_state_dict


def fine_tune_and_test(model_dir, params, hyperparams, label_dict, all_labels, fine_tune_df, query_dataloader,
                       optimizer, writer, inner_cv, seed, model_save_dir=None):
    """
    This function is used in the nested cross-validation to fine-tune and test a network type with the given
    hyperparameters on a fine-tune and query datasets.
    Args:
        model_dir: directory of the pretrained model
        params: parameter
        hyperparams: hyperparameter
        label_dict: dictionary with cancer type as first keys, corresponding subtypes as second keys
                    and subtype indices as values.
        all_labels:
        fine_tune_df: data frame of the dataset used for fine-tuning
        query_dataloader: dataloader of the query dataset
        optimizer:
        writer: tensorboard writer to log the evaluation results for the query dataset
        inner_cv: True if this function is used for inner cv
                  False if used for outer cv
        seed:
        model_save_dir: directory where to save the fine-tuned models

    Returns: the mean evaluation results of the query dataset,
             with the mean over all tasks for each metric and epoch

    """
    if inner_cv:
        num_epochs = params["epochs_fine-tune"]

        results = {}
        for ep in create_query_epochs(num_epochs):
            results[ep] = {}

    else:  # BEST outer model
        num_epochs = hyperparams["ft_epoch_stop"]
        results = {num_epochs - 1: {}}

    pbar_tasks = tqdm(range(params["n_tasks"]), total=params["n_tasks"])
    pbar_tasks.set_description("FINE-TUNE")

    all_support_performances = {}

    for task_n in pbar_tasks:
        # load pretrained model
        network = init_model(params, hyperparameter_dict_from_info_file(os.path.join(Path(model_dir).parents[1], "model_info.txt")), len(all_labels[0].unique()))
        network.train_on_full = False
        device = torch.device(r'cuda:0' if torch.cuda.is_available() else r'cpu')
        network.load_state_dict(torch.load(model_dir, map_location=device))

        # ------------------------  Fine-tuning stage ------------------------
        network.change_classifier(len(all_labels[1].unique()))
        network.label2type = query_dataloader.dataset.label2type

        if params["network_type"] == "Hopfield":
            network.set_fixed_stored_df(fine_tune_df, np.infty)
            network.hopf_drop = hyperparams["hopfield_droput"]

        if params["n_shots"] == 0:  # zero shot
            performance_query, query_metrics = iterate_dataset(network, query_dataloader, writer[-1], task_n, None,
                                                               None, "query")

            for metric in performance_query.keys():
                if metric not in results[-1].keys():
                    results[-1][metric] = [performance_query[metric]]
                else:
                    results[-1][metric].append(performance_query[metric])

        else:
            optimizer_fine_tune = optimizer(filter(lambda p: p.requires_grad, network.parameters()),
                                            lr=hyperparams["ft_learning_rate"],
                                            weight_decay=hyperparams["ft_weight_decay"])
            # sample for each task a new support set with n-shot samples per class to fine-tune on
            support_dataloader = create_support_dataloader(fine_tune_df, hyperparams["ft_batch_size"],
                                                           params, label_dict, loaded_bags=True, seed=seed)

            for epoch in range(num_epochs):
                # fine-tune network on support set
                performance, _ = iterate_dataset(network, support_dataloader, None, None, optimizer_fine_tune, None,
                                                 name="finetune")

                # evaluate network on query
                if epoch in results.keys():
                    performance_query, query_metrics = iterate_dataset(network, query_dataloader, writer[epoch], task_n,
                                                                       None, None, "query")

                    for metric in performance_query.keys():
                        if metric not in results[epoch].keys():
                            results[epoch][metric] = [performance_query[metric]]
                        else:
                            results[epoch][metric].append(performance_query[metric])

                    writer[epoch].add_hyperparams_writer_results(query_metrics)

                if epoch not in all_support_performances.keys():
                    all_support_performances[epoch] = {}
                for metric in performance.keys():
                    if metric in all_support_performances[epoch].keys():
                        all_support_performances[epoch][metric].append(performance[metric])
                    else:
                        all_support_performances[epoch][metric] = [performance[metric]]

        # -------------------------------- save model -----------------------------------
        if model_save_dir is not None:
            model_save_dir_curr = os.path.join(model_save_dir, f"{task_n}th_fine_tuned_model.pth")
            if not os.path.exists(os.path.dirname(model_save_dir_curr)):
                os.makedirs(os.path.dirname(model_save_dir_curr))
            torch.save(network.state_dict(), model_save_dir_curr)

        query_acc_string = ""
        query_loss_string = ""
        for e in results.keys():
            query_acc_string += f"{round(np.mean(results[e]['balanced_accuracy']), 3)} | "
            query_loss_string += f"{round(np.mean(results[e]['loss']), 3)} | "

        # average of metric of last epoch over tasks
        support_mean_acc_str = round(np.mean(all_support_performances[epoch]['balanced_accuracy']), 3) if \
            params["n_shots"] != 0 else "--"
        support_mean_loss_str = round(np.mean(all_support_performances[epoch]['loss']), 3) if \
            params["n_shots"] != 0 else "--"

        pbar_tasks.set_description(f"FINE-TUNE || SUPPORT -- Accuracy: {support_mean_acc_str}, "
                                   f"Loss: {support_mean_loss_str} |||| QUERY -- Accuracies: {query_acc_string}, "
                                   f"Loss: {query_loss_string}")
        if params["n_shots"] == 0:
            break

    writer_max_epoch = max(writer.keys())
    for e in all_support_performances.keys():
        fine_tune_metrics_mean = {}
        fine_tune_metrics_std = {}
        for metric in all_support_performances[e].keys():
            fine_tune_metrics_mean[f"{metric}/fine_tune_mean"] = np.mean(all_support_performances[e][metric])
            fine_tune_metrics_std[f"{metric}/fine_tune_std"] = np.std(all_support_performances[e][metric])

        writer[writer_max_epoch].add_writer_results(fine_tune_metrics_mean, e)
        writer[writer_max_epoch].add_writer_results(fine_tune_metrics_std, e)

    for epoch_writer in writer.values():
        epoch_writer.close_writer()

    # mean over all tasks for each metric and epoch(50, 100, 150)
    results_mean = {}
    for e in results.keys():
        results_mean[e] = {}
        epoch_results = results[e]
        for metric in epoch_results.keys():
            results_mean[e][metric] = np.mean(epoch_results[metric])

    return results_mean


def init_model(params, hyperparams, output_size):
    """
    This function initializes the model with the given model type and its parameters
    :param params: the model parameters
    :param hyperparams: the hyperparameters
    :param output_size: the number of possible classes the model should be able to predict
    :return: the initialized network
    """
    device = torch.device(r'cuda:0' if torch.cuda.is_available() else r'cpu')
    if params["network_type"] == "FeedForward":
        network = FeedForward(output_size=output_size, instance_hidden_size=hyperparams["instance_hidden"],
                              bag_hidden=hyperparams["bag_hidden"], instance_dropout=hyperparams["instance_dropout"],
                              bag_dropout=hyperparams["bag_dropout"],
                              aggregation=params["pooling_function"]).to(device=device)
    elif params["network_type"] == "Hopfield":
        network = HistoHopfield(output_size=output_size, num_heads=hyperparams["num_heads"],
                                instance_hidden_size=hyperparams["instance_hidden"],
                                bag_hidden=hyperparams["bag_hidden"], d_model=hyperparams["d_model"],
                                aggregation=params["pooling_function"], scaling=hyperparams["scaling"],
                                instance_dropout=hyperparams["instance_dropout"],
                                bag_dropout=hyperparams["bag_dropout"],
                                hopf_drop=hyperparams["hopf_drop"]).to(device=device)
    else:
        raise ValueError(f"Not a valid network type {params['network_type']}")

    return network


def cross_validate(run_folds, optimizer, outer_train_ids, dataset_df, label_dict, all_labels,
                   model_directories, hyperparams_permutations, params, seed, logdir):
    """
    This function performs cross validation
    Args:
        run_folds: the number of folds
        optimizer:
        outer_train_ids: indices for the training set of the outer cv
        dataset_df:
        label_dict: dictionary with cancer type as first keys, corresponding subtypes as second keys
                    and subtype indices as values.
        all_labels: [pretraining labels, fine-tune labels]
        model_directories: list of the full directories of all pretrained models
        hyperparams_permutations: list of dictionaries, where each dictionary represents one hyperparameter constellation
        params:
        seed:
        logdir: directory where the evaluation results are saved to

    Returns: best_pretrain_model
             best_hyperparams: dict of the best hyperparameter constellation

    """
    inner_cv_performances_list = []

    k_fold = StratifiedKFold(n_splits=run_folds, shuffle=True, random_state=seed)
    query_epochs = create_query_epochs(params["epochs_fine-tune"])
    writer_dict, query_epochs_hyperparams = create_writer_dict_for_model_grid(logdir, model_directories,
                                                                              hyperparams_permutations, query_epochs)

    # set fold
    for fold, (inner_train_ids, inner_test_ids) in enumerate(
            k_fold.split(np.arange(len(all_labels[1][outer_train_ids])),
                         y=all_labels[1][outer_train_ids].values)):

        print("\n")
        fine_tune_df = dataset_df.iloc[outer_train_ids[inner_train_ids]].reset_index(drop=True)
        fine_tune_df.loc[:, "bag"] = fine_tune_df["bag_dir"].progress_map(
            lambda x: np.load(x).astype(np.float32))
        query_df = dataset_df.iloc[outer_train_ids[inner_test_ids]].reset_index(drop=True)
        query_dataloader = create_query_dataloader(query_df, label_dict)

        num_hypersettings = len(hyperparams_permutations)
        # set model
        for model_index, curr_model_dir in enumerate(model_directories):

            # choose fine-tune hyperparameters
            for num_setting, curr_hyperparams in enumerate(hyperparams_permutations):
                print(f"\n\n{'-' * 36} INNER FOLD {fold+1}/{run_folds} || MODEL {model_index+1}/"
                      f"{len(model_directories)} || Hyperparameter setting {num_setting + 1}/{num_hypersettings}"
                      f" {'-' * 36}")

                curr_dict = hyperparameter_dict_from_info_file(os.path.join(Path(curr_model_dir).parents[1], "model_info.txt"))
                curr_dict.update(curr_hyperparams)

                for writer in writer_dict[curr_model_dir][num_setting].values():
                    writer.create_writer(logdir=f"inner_fold_{fold}")

                performance = fine_tune_and_test(model_dir=curr_model_dir, params=params, hyperparams=curr_dict,
                                                 label_dict=label_dict, all_labels=all_labels,
                                                 fine_tune_df=fine_tune_df, query_dataloader=query_dataloader,
                                                 optimizer=optimizer, writer=writer_dict[curr_model_dir][num_setting],
                                                 inner_cv=True, seed=seed, model_save_dir=None)

                for ep in performance.keys():
                    epoch_performance = performance[ep]
                    for metric in epoch_performance.keys():
                        inner_cv_performances_list.append([fold, curr_model_dir, num_setting, ep, metric, epoch_performance[metric]])

    close_writer_dict_for_model_grid(writer_dict, query_epochs_hyperparams)

    inner_cv_performances_df = pd.DataFrame(inner_cv_performances_list, columns=["fold", "model_dir", "add_hparams_idx",
                                                                                 "epoch", "metric", "performance"])
    grouped_df = inner_cv_performances_df.groupby(["model_dir", "add_hparams_idx", "epoch", "metric"]).performance.apply(
        lambda x: x.tolist()).reset_index()

    grouped_df["mean"] = grouped_df.performance.apply(lambda x: np.mean(x))
    grouped_df["std"] = grouped_df.performance.apply(lambda x: np.std(x))

    max_row_idx = grouped_df[grouped_df["metric"] == "balanced_accuracy"]["mean"].idxmax()
    best_pretrain_model = grouped_df.iloc[max_row_idx]["model_dir"]
    best_hyperparam_idx = grouped_df.iloc[max_row_idx]["add_hparams_idx"]

    best_hyperparams = hyperparameter_dict_from_info_file(os.path.join(Path(best_pretrain_model).parents[1], "model_info.txt"))
    best_hyperparams.update(hyperparams_permutations[best_hyperparam_idx])
    best_hyperparams["ft_epoch_stop"] = grouped_df.iloc[max_row_idx]["epoch"] + 1

    gc.collect()

    return best_pretrain_model, best_hyperparams


def nested_cross_validate(outer_folds, inner_folds, params, hyperparams, optimizer, dataset_path_pretrain,
                          dataset_path_finetune, pretrained_models_basepath, seed):
    """
    This function performs nested cross-validation with #outer_folds and #inner_folds
    Args:
        outer_folds: number of outer cv folds
        inner_folds: number of inner cv folds
        params: dictionary of fix parameters such as the network_type or the aggregation type
        hyperparams: dictionary where each hyperparameter has a list of possible values
        optimizer:
        dataset_path_pretrain: base path where all pretraining data is saved
        dataset_path_finetune: base path where all fine-tuning data is saved
        pretrained_models_basepath: base path where all pretrained models are saved
        seed:

    Returns: the mean and standard deviation of the balanced accuracy over all outer folds

    """

    model_directories = glob.glob(os.path.join(
        pretrained_models_basepath, params['network_type'], "Best", params['pooling_function'], "*", "*", "*.pth"),
        recursive=True)

    logdir = os.path.join("runs", "nested_cv", params['network_type'], params['pooling_function'], str(params['n_shots']),
                          "_shot")
    outer_k_fold = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)

    keys, values = zip(*hyperparams.items())
    hyperparams_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    nested_cv_performance = {}
    possible_fine_tune_sets = ["COAD", "STAD"]
    for i in range(len(possible_fine_tune_sets)):
        print("-" * 100)
        print(f"{'-' * 42} DATASET CONSTELLATION {i + 1}/{len(possible_fine_tune_sets)} {'-' * 42}")

        curr_nested_cv_performance = {}
        base_set_logdir = os.path.join(logdir, possible_fine_tune_sets[i])

        pretrain_df, _ = load_cancer_csvs(dataset_path_pretrain, fine_tune_set=None)
        fine_tune_df, fine_tune_label_dict = load_cancer_csvs(dataset_path_finetune, fine_tune_set=possible_fine_tune_sets[i])

        for fold, (outer_train_ids, outer_test_ids) in enumerate(outer_k_fold.split(np.arange(fine_tune_df.shape[0]),
                                                                                    y=fine_tune_df["label"])):
            outer_fold_logdir = os.path.join(base_set_logdir, f"outer_fold_{fold}")
            print("-" * 100)
            print(f"{'-' * 42} OUTER FOLD {fold + 1}/{outer_folds} {'-' * 42}")

            best_pretrain_model, best_hyperparams = cross_validate(run_folds=inner_folds, optimizer=optimizer,
                                                                   outer_train_ids=outer_train_ids,
                                                                   dataset_df=fine_tune_df,
                                                                   label_dict=fine_tune_label_dict,
                                                                   all_labels=[pretrain_df["label"],
                                                                               fine_tune_df["label"]],
                                                                   model_directories=model_directories,
                                                                   hyperparams_permutations=hyperparams_permutations,
                                                                   params=params, seed=seed, logdir=outer_fold_logdir)

            print(f"{'-' * 42} BEST SETTING {fold + 1}/{outer_folds} {'-' * 42}")
            print(f"{best_hyperparams}")
            best_hyperparam_logdir = os.path.join(outer_fold_logdir, "Best", strftime("%Y%m%d_%H%M%S", gmtime()))

            writer_key = best_hyperparams["ft_epoch_stop"] - 1
            best_pretrain_model_hyperp = hyperparameter_dict_from_info_file(os.path.join(Path(best_pretrain_model).parents[1], "model_info.txt"))
            total_hyperparams = best_pretrain_model_hyperp | best_hyperparams

            best_writer = {writer_key: TensorboardWriter(best_hyperparam_logdir, total_hyperparams)}
            best_writer[writer_key].create_writer()
            best_model_save_dir = os.path.join(best_hyperparam_logdir, "model")

            outer_fine_tune_df = fine_tune_df.iloc[outer_train_ids].reset_index(drop=True)
            outer_fine_tune_df.loc[:, "bag"] = outer_fine_tune_df["bag_dir"].progress_map(
                lambda x: np.load(x).astype(np.float32))
            query_df = fine_tune_df.iloc[outer_test_ids].reset_index(drop=True)
            query_dataloader = create_query_dataloader(query_df, fine_tune_label_dict)

            performance = fine_tune_and_test(model_dir=best_pretrain_model, params=params, hyperparams=best_hyperparams,
                                             label_dict=fine_tune_label_dict,
                                             all_labels=[pretrain_df["label"], fine_tune_df["label"]],
                                             fine_tune_df=outer_fine_tune_df, query_dataloader=query_dataloader,
                                             optimizer=optimizer, writer=best_writer, inner_cv=False, seed=seed,
                                             model_save_dir=best_model_save_dir)

            performance = performance[writer_key]

            for metric in performance.keys():
                if metric not in nested_cv_performance.keys():
                    nested_cv_performance[metric] = [performance[metric]]
                else:
                    nested_cv_performance[metric].append(performance[metric])

                if metric not in curr_nested_cv_performance.keys():
                    curr_nested_cv_performance[metric] = [performance[metric]]
                else:
                    curr_nested_cv_performance[metric].append(performance[metric])

        save_results_to_csv(base_set_logdir, curr_nested_cv_performance)
        print(f"\n{'-' * 60}")

    print(f"{'-' * 7} 'NESTED CROSS VALIDATION' RESULTS {'-' * 7}")
    for metric in nested_cv_performance.keys():
        print(f"{metric}: Average - {round(np.mean(nested_cv_performance[metric]), 3)}, Std - "
              f"{round(np.std(nested_cv_performance[metric]), 3)}")

    save_results_to_csv(logdir, nested_cv_performance)

    return np.mean(nested_cv_performance["balanced_accuracy"]), np.std(nested_cv_performance["balanced_accuracy"])
