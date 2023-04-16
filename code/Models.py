"""
This file defines all network types, their initialization and evaluation metrics.

created by Stephanie Henkel
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, accuracy_score

from Pooling_Functions import choose_pooling_function
from constants import SEED

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def initialize_weights_he(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0.0, std=np.sqrt(2 / m.in_features))
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif isinstance(m, nn.ModuleList):
        for elem in m:
            if isinstance(elem, nn.Linear):
                nn.init.normal_(elem.weight.data, mean=0.0, std=np.sqrt(2 / elem.in_features))
                if elem.bias is not None:
                    elem.bias.data.fill_(0)

    elif isinstance(m, nn.Parameter):
        nn.init.normal_(m.data, mean=0.0, std=np.sqrt(2 / m.in_features))
        if m.bias is not None:
            m.bias.data.fill_(0)


class Classifier(nn.Module):
    def __init__(self, input_size=1920, layer_dims=[32], dropout=0.0):
        super(Classifier, self).__init__()
        self.num_layers = len(layer_dims)
        self.layer_dims = layer_dims

        layers_list = [nn.Linear(input_size, self.layer_dims[0])]
        batch_norm_list = []

        for i in range(self.num_layers - 1):
            layers_list.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            batch_norm_list.append(nn.BatchNorm1d(self.layer_dims[i]))

        self.layers = nn.ModuleList(layers_list)

        self.batch_norms = nn.ModuleList(batch_norm_list)
        self.dropout = nn.Dropout(dropout)
        self.apply(initialize_weights_he)
        self.act_fn = nn.ReLU()

        # initialize last linear layer different since no RELU is applied
        nn.init.xavier_normal(self.layers[-1].weight.data, gain=1.0)

        self.freezed_layers = False

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            module.train(mode)
            if self.freezed_layers and mode:
                if name not in [f'layers']:
                    module.train(False)
                else:
                    for layer_ind, layer in module.named_children():
                        if layer_ind != str((self.num_layers - 1)):
                            layer.train(False)

    def forward(self, x):
        h = x
        for i in range(self.num_layers):
            h = self.layers[i](h)  # linear layer

            if i < (self.num_layers - 1):
                h = self.batch_norms[i](h)  # batch norm
                h = self.act_fn(h)
                h = self.dropout(h)

        return h


class InstanceEmbedder(nn.Module):
    def __init__(self, input_size=1920, layer_dims=[32], dropout=0.0):
        super(InstanceEmbedder, self).__init__()

        self.num_layers = len(layer_dims)
        self.layer_dims = layer_dims.tolist()

        layers_list = [nn.Linear(input_size, self.layer_dims[0])]
        layer_norm_list = [nn.LayerNorm(self.layer_dims[0])]

        for i in range(self.num_layers - 1):
            layers_list.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            layer_norm_list.append(nn.LayerNorm(self.layer_dims[i + 1]))

        self.layers = nn.ModuleList(layers_list)

        self.layer_norms = nn.ModuleList(layer_norm_list)
        self.dropout = nn.Dropout(dropout)
        self.apply(initialize_weights_he)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        h = x
        for i in range(self.num_layers):
            h = self.layers[i](h)  # linear layer
            h = self.layer_norms[i](h)  # layer norm
            h = self.act_fn(h)
            h = self.dropout(h)

        return h


class FeedForward(nn.Module):
    def __init__(self, input_size=1920, output_size=5, instance_hidden_size=None, bag_hidden=None,
                 instance_dropout=0.0, bag_dropout=0.0, aggregation="max", label2type=None):
        super(FeedForward, self).__init__()

        self.start_batch_norm = nn.BatchNorm1d(num_features=input_size, affine=False)

        if instance_hidden_size is None:
            self.agg_input_dim = input_size
            self.instance_embedder = lambda y: y
        else:
            self.agg_input_dim = int(instance_hidden_size)
            self.instance_embedder = InstanceEmbedder(input_size=input_size, layer_dims=instance_hidden_size,
                                                      dropout=instance_dropout)

        self.pooling_function = choose_pooling_function(aggregation, self.agg_input_dim)

        self.bag_hidden = bag_hidden.tolist()
        classifier_dims = self.bag_hidden + [output_size]

        self.classifier = Classifier(input_size=self.agg_input_dim, layer_dims=classifier_dims,
                                     dropout=bag_dropout)
        self.train_on_full = True
        self.fine_tune = False
        self.freezed_layers = False
        self.label2type = label2type
        self.output_size = output_size

    def change_classifier(self, new_output_size):
        self.fine_tune = True
        self.output_size = new_output_size
        # adapt last layer
        device = torch.device(r'cuda:0' if torch.cuda.is_available() else r'cpu')
        self.classifier.layers[-1] = nn.Linear(self.classifier.layers[-1].in_features, new_output_size).to(device=device)
        nn.init.xavier_normal(self.classifier.layers[-1].weight.data, gain=1.0)

        if not self.train_on_full:
            for name, param in self.named_parameters():
                if f'classifier.layers.{self.classifier.num_layers - 1}' not in name:
                    param.requires_grad = False

            self.freezed_layers = True
            self.classifier.freezed_layers = True

    def train(self, mode: bool = True):
        # this adapted train function only sets non freezed layers into training mode
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            module.train(mode)
            if self.freezed_layers and mode:
                # freeze all but classifier
                if name != "classifier":
                    module.train(False)

        if self.fine_tune:
            self.start_batch_norm.train(False)
        return self

    def forward(self, x):
        h = x[0]  # torch.rand(sum(bag_lengths), self.output_size)
        bag_lengths = x[1]

        h = self.start_batch_norm(h)
        h = self.instance_embedder(h)
        h = self.pooling_function.aggregate(h, bag_lengths)
        pred = self.classifier(h)

        return pred


class HistoHopfield(nn.Module):
    def __init__(self, input_size=1920, output_size=5, num_heads=1, instance_hidden_size=None, bag_hidden=None, d_model=1024,
                 aggregation="max", scaling=None, label2type=None, instance_dropout=0.0, bag_dropout=0.0,
                 hopf_drop=0.0):
        super(HistoHopfield, self).__init__()

        self.start_batch_norm = nn.BatchNorm1d(num_features=input_size, affine=False)

        if instance_hidden_size is None:
            self.agg_input_dim = input_size
            self.instance_embedder = lambda y: y
        else:
            self.agg_input_dim = int(instance_hidden_size)
            self.instance_embedder = InstanceEmbedder(input_size=input_size, layer_dims=instance_hidden_size,
                                                      dropout=instance_dropout)

        self.pooling_function = choose_pooling_function(aggregation, self.agg_input_dim)

        self.train_on_full = True
        self.fine_tune = False
        self.freezed_layers = False
        self.label2type = label2type
        self.output_size = output_size

        self.n_fix_stored = 1  # will be set later in set_fixed_stored_df
        self.possible_stored_pattern = None
        self.fixed_stored_df = None
        self.precomputed_key_data = None
        self.precomputed_labels = None

        self.seed = SEED
        self.device = torch.device(r'cuda:0' if torch.cuda.is_available() else r'cpu')

        # --------------- define Hopfield layer ---------------
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = int(self.d_model / self.num_heads)
        if scaling is None:
            self.scaling = 1.0 / np.sqrt(self.d_k)
        else:
            self.scaling = scaling

        self.bag_hidden = bag_hidden.tolist()
        self.pre_hopfield_layer = InstanceEmbedder(input_size=self.agg_input_dim, layer_dims=bag_hidden,
                                                   dropout=bag_dropout)

        hopfield_input = self.bag_hidden[-1]
        self.hopf_drop = hopf_drop

        self.W_q = nn.Linear(hopfield_input, self.d_model, bias=False)
        self.W_k = nn.Linear(hopfield_input, self.d_model, bias=False)
        # He initialization (=kaiming_normal_)
        torch.nn.init.kaiming_normal_(self.W_q.weight, mode='fan_in', nonlinearity='linear')
        torch.nn.init.kaiming_normal_(self.W_k.weight, mode='fan_in', nonlinearity='linear')
        self.softmax = torch.nn.Softmax(dim=-1)

    def change_classifier(self, new_output_size):
        self.output_size = new_output_size
        self.fine_tune = True

        if not self.train_on_full:
            # freeze all but last layer
            for name, param in self.named_parameters():
                if "W_q" not in name and "W_k" not in name:
                    param.requires_grad = False

            self.freezed_layers = True

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            module.train(mode)
            if self.freezed_layers and mode:
                if name not in ["W_q", "W_k", "softmax"]:
                    module.train(False)

        if self.fine_tune:
            self.start_batch_norm.train(False)

        return self

    @torch.no_grad()
    def prepare_labels(self, labels):
        prepared_labels = torch.zeros((len(labels), self.output_size))
        for i in range(len(labels)):
            prepared_labels[i, labels[i]] = 1

        return prepared_labels

    def create_stored_df(self, df, n_stored=1):
        stored_df = df.groupby("label").apply(
            lambda x: x.sample(n_stored, random_state=self.seed)).droplevel("label").reset_index(drop=True)
        if not torch.cuda.is_available():
            stored_df.loc[:, "bag"] = stored_df["bag_dir"].map(lambda x: np.load(x).astype(np.float32))
        return stored_df

    def get_stored_data(self, df):
        stored_data = torch.from_numpy(np.vstack(df["bag"])).to(device=self.device)
        stored_bag_lengths = df["bag_size"].to_list()
        stored_labels = df["label"].to_list()
        stored_labels = self.prepare_labels(stored_labels).to(device=self.device)

        return [stored_data, stored_bag_lengths, stored_labels]

    def set_fixed_stored_df(self, df, n_stored=1):
        self.possible_stored_pattern = df
        self.n_fix_stored = min(df["label"].value_counts().min(), n_stored)

        # randomly sample n_fix_stored keys from each label (=n_fix_stored x num_labels)
        self.fixed_stored_df = self.create_stored_df(df, self.n_fix_stored)
        return

    @torch.no_grad()
    def precompute_keys(self):
        new_order = np.concatenate(
            [np.arange(len(self.fixed_stored_df))[i:: self.n_fix_stored] for i in range(self.n_fix_stored)])

        [stored_data, stored_bag_lengths, self.precomputed_labels] = self.get_stored_data(self.fixed_stored_df.loc[new_order])
        self.precomputed_key_data = self.forward_data(stored_data, stored_bag_lengths, self.W_k).view(self.n_fix_stored,
                                                                                                      self.num_heads,
                                                                                                      self.output_size,
                                                                                                      self.d_k)
        return

    @torch.no_grad()
    def sample_stored_pattern(self, batch_indices):
        # take all except current query samples as possible key
        leftover_df = self.possible_stored_pattern[~self.possible_stored_pattern.index.isin(batch_indices.tolist())]

        # randomly sample 1 key for each label
        stored_df = self.create_stored_df(leftover_df, 1)

        return self.get_stored_data(stored_df)

    def forward_data(self, data, bag_lengths, matrix):
        data = self.start_batch_norm(data)
        data = self.instance_embedder(data)
        data = self.pooling_function.aggregate(data, bag_lengths)
        data = self.pre_hopfield_layer(data)
        k = matrix(data)
        return k

    def forward(self, x):
        [query_images, query_bag_lengths, batch_indices] = x

        if self.training:
            [stored_data, stored_bag_lengths, stored_labels] = self.sample_stored_pattern(batch_indices)
            # size(k) = heads, output_size, d_k
            k = self.forward_data(stored_data, stored_bag_lengths, self.W_k).view(-1, self.num_heads, self.d_k).transpose(0, 1)
        else:
            # size(k) = n_fix_stored, heads, output_size, d_k
            k, stored_labels = self.precomputed_key_data, self.precomputed_labels

        # when we think of transformer attention with batch_size=1 and tgt_len = num_state, src_len=num_stored
        q = self.forward_data(query_images, query_bag_lengths, self.W_q).view(-1, self.num_heads, self.d_k).transpose(0, 1)  # heads, num_state, d_k
        q = self.scaling * q

        if self.training:
            q_k = torch.bmm(q, k.transpose(1, 2))  # (heads, num_state, num_stored), multiplies for each head seperately
            assoc_matrix = self.softmax(q_k)
            assoc_matrix = nn.functional.dropout(assoc_matrix, p=self.hopf_drop).mean(dim=0)
            prediction = assoc_matrix @ stored_labels
        else:
            # I keep the n_fix_stored separated, because if e.g. a query is of class 2 with medium signal
            # but one key of another class e.g. class 3 has somehow a high similarity (outlier), then it will
            # miss-classify this sample nonetheless the similarity chose the highest class on majority vote of key-sets
            q_k = torch.stack([torch.bmm(q, k[i].transpose(1, 2)) for i in range(k.shape[0])])  # (n_fix_stored, heads, num_state, output_size)
            assoc_matrix = self.softmax(q_k).mean(dim=1)  # mean over heads
            predictions = torch.bmm(assoc_matrix, stored_labels.view(self.n_fix_stored, self.output_size, -1)).transpose(0, 1)  # state, n_fix_stored, output
            y_hat_per_sample = predictions.argmax(dim=-1)  # the max label of all stored per n_fix_stored sampling , (num_state, n_fix_stored)
            major_y_hat = torch.mode(y_hat_per_sample).values  # the major label vote per state pattern over all n_fix_stored

            new_preds = torch.zeros((predictions.shape[0], self.output_size)).to(self.device)  # (num_state, num_classes)
            for i in range(predictions.shape[0]):  # run over state_patterns
                index = y_hat_per_sample[i] == major_y_hat[i]
                new_preds[i] = predictions[i][index].mean(dim=0)  # average over those predictions, which predicted the majority vote for this state pattern
                # the statement above excludes in this way the outliers discussed above
            prediction = new_preds

        return prediction


@torch.no_grad()
def calculate_metrics(y_pred, y_true):
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    y_hat = torch.argmax(y_pred, dim=1)

    # accuracy and balanced accuracy over all samples (over all cancer types together)
    all_metrics = {"accuracy": accuracy_score(y_true, y_hat),
                   "balanced_accuracy": balanced_accuracy_score(y_true, y_hat)}

    return all_metrics
        

    