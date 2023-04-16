"""
This file contains several functions to handle the data as well as a dataset class.

created by Stephanie Henkel
"""

import pandas as pd
import numpy as np
import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader

from constants import *

import os
import glob
import collections
import tqdm
tqdm.tqdm.pandas()


class HistopathologySlideBags(data_utils.Dataset):
    def __init__(self, data, label_dict, max_bag_size=2000, already_loaded_bags=False):
        self.data = data
        self.label_dict = label_dict
        self.load_to_memory = torch.cuda.is_available()
        self.max_bag_size = max_bag_size
        self.unique_labels = None

        self.label2type = {}
        self.already_loaded_bags = already_loaded_bags
        self.device = torch.device(r'cuda:0' if torch.cuda.is_available() else r'cpu')

        self.load_data()

    def __len__(self):
        return len(self.data)

    def load_data(self):
        if self.load_to_memory and not self.already_loaded_bags:
            self.data.loc[:, "bag"] = self.data["bag_dir"].progress_map(lambda x: np.load(x).astype(np.float32))

        self.unique_labels = sum([len(subtypes_dict) for subtypes_dict in self.label_dict.values()])

        for cancer_type in self.label_dict.keys():
            type_dict = self.label_dict[cancer_type]
            for subtype in type_dict.keys():
                self.label2type[type_dict[subtype]] = cancer_type

    def __getitem__(self, index):
        if self.load_to_memory:
            bag = self.data["bag"][index]
        else:
            bag = np.load(self.data["bag_dir"][index]).astype(np.float32)

        label = self.data["label"][index].astype(np.int64)
        bag_size = self.data["bag_size"][index]

        if bag_size > self.max_bag_size:
            keep_indices = np.random.choice(bag_size, size=self.max_bag_size, replace=False)
            bag = bag[keep_indices]
            bag_size = self.max_bag_size

        return bag, label, bag_size, index


def load_cancer_csvs(embeddings_dir, fine_tune_set=None):  # DONE
    """
    This function loads the csv-files of all required cancer types into one
    data frame with their subtypes as labels.

    Args:
        embeddings_dir: directory of data with cancer types as folders
        fine_tune_set: None, "STAD" or "COAD"

    Returns:
        data: data frame with columns --> cancer_type | label | bag_dir (former directory) | bag_size (former bag_length)
        label_dict: dictionary with cancer type as first keys, corresponding subtypes as second
                    keys and subtype indices as values.

    """
    all_csv_dirs = glob.glob(os.path.join(embeddings_dir, "*", "*.csv"), recursive=True)

    if fine_tune_set:
        cancer_type_csv_dirs = [csv_dir for csv_dir in all_csv_dirs if fine_tune_set in csv_dir]
    else:
        cancer_type_csv_dirs = all_csv_dirs

    label_dict = {}
    subtype_index = 0
    data = pd.DataFrame()
    cancer_type_csv_dirs.sort()

    for csv in cancer_type_csv_dirs:
        curr_df = pd.read_csv(csv)

        # filter out samples which fulfill criteria
        curr_df = curr_df[curr_df["bag_length"] < MAX_BAG_LENGTH]
        curr_df = curr_df[curr_df["bag_length"] >= MIN_BAG_LENGTH]
        label_counts = curr_df["label"].value_counts()
        subtypes = label_counts.keys()[label_counts >= MIN_SAMPLES]

        subtypes = np.asarray(subtypes)
        subtypes.sort()

        if len(subtypes) > 1:
            curr_df = curr_df[curr_df["label"].isin(subtypes)]
            cur_cancer_type = curr_df["cancer_type"].unique().item()
            label_dict[cur_cancer_type] = dict(zip(subtypes, np.arange(subtype_index, subtype_index + len(subtypes))))
            subtype_index += len(subtypes)
            curr_df["label"] = curr_df["label"].map(label_dict[cur_cancer_type])  # subtype name to label index
            curr_df["directory"] = curr_df["directory"].map(lambda x: os.path.join(embeddings_dir, x))
            data = pd.concat([data, curr_df[["cancer_type", "label", "directory", "bag_length"]].rename(
                columns={"directory": "bag_dir", "bag_length": "bag_size"})], ignore_index=True)

    return data, label_dict


def collate(batch):
    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch, 0, out=out)  # in default collate they use torch.stack
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    elif isinstance(elem, int):
        return torch.tensor(batch)


def create_support_dataloader(fine_tune_df, batch_size, params, label_dict, loaded_bags=True, seed=243):
    if params["n_shots"] != 0:
        support_df = fine_tune_df.groupby("label").apply(lambda x: x.sample(params["n_shots"], random_state=seed)).droplevel("label")
        support_set = HistopathologySlideBags(support_df.reset_index(drop=True), label_dict, max_bag_size=np.infty,
                                              already_loaded_bags=loaded_bags)
        support_dataloader = DataLoader(support_set, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate)
    else:
        support_dataloader = None

    return support_dataloader


def create_query_dataloader(query_df, label_dict):
    query_set = HistopathologySlideBags(query_df, label_dict, max_bag_size=np.infty)
    query_loader = DataLoader(query_set, batch_size=128, shuffle=False, collate_fn=collate)
    return query_loader







