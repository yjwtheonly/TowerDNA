import torch
import random
import numpy as np
from torch_geometric.nn import GCNConv, GINConv
import torch.nn.functional as F
from model.layers import AttentionModule2, TensorNetworkModule
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree


class SimGNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """

    def __init__(self, args, number_of_labels = 6):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SimGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram:
            self.feature_count = self.args.tensor_neurons + self.args.bins  # 32
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.normalize = torch.nn.BatchNorm1d(self.number_labels)
        self.calculate_bottleneck_features()
        if self.args.gnn_operator == "gcn":
            self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == "gin":
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.number_labels, self.args.filters_1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_1, self.args.filters_1),
                torch.nn.BatchNorm1d(self.args.filters_1),
            )

            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_1, self.args.filters_2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_2, self.args.filters_2),
                torch.nn.BatchNorm1d(self.args.filters_2),
            )

            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_2, self.args.filters_3),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3),
            )

            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
        else:
            raise NotImplementedError("Unknown GNN-Operator.")

        self.attention = AttentionModule2(self.args)

        self.tensor_network = TensorNetworkModule(self.args, self.args.filters_3)
        if self.args.num_NTN_mlp == 1:
            self.fully_connected_first = torch.nn.Linear(
                self.feature_count, self.args.bottle_neck_neurons  # 32 -> 16
            )
            self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons,1)
        elif self.args.num_NTN_mlp == 2:
            raise NotImplementedError("Not implemented yet.")
        
    def calculate_histogram(
        self, abstract_features_1, abstract_features_2, batch_1, batch_2
    ):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for target graphs.
        :param abstract_features_2: Feature matrix for source graphs.
        :param batch_1: Batch vector for source graphs, which assigns each node to a specific example
        :param batch_1: Batch vector for target graphs, which assigns each node to a specific example
        :return hist: Histsogram of similarity scores.
        """
        abstract_features_1, mask_1 = to_dense_batch(abstract_features_1, batch_1)
        abstract_features_2, mask_2 = to_dense_batch(abstract_features_2, batch_2)

        B1, N1, _ = abstract_features_1.size()
        B2, N2, _ = abstract_features_2.size()

        mask_1 = mask_1.view(B1, N1)
        mask_2 = mask_2.view(B2, N2)
        num_nodes = torch.max(mask_1.sum(dim=1), mask_2.sum(dim=1))

        scores = torch.matmul(
            abstract_features_1, abstract_features_2.permute([0, 2, 1])
        ).detach()

        hist_list = []
        for i, mat in enumerate(scores):
            mat = torch.sigmoid(mat[: num_nodes[i], : num_nodes[i]]).view(-1)
            hist = torch.histc(mat, bins=self.args.bins)
            hist = hist / torch.sum(hist)
            hist = hist.view(1, -1)
            hist_list.append(hist)

        return torch.stack(hist_list).view(-1, self.args.bins)

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Abstract feature matrix.
        """
        features = self.normalize(features)
        features = self.convolution_1(features, edge_index)
        features = F.relu(features)
        features = F.dropout(features, p=self.args.dropout, training=self.training)
        features = self.convolution_2(features, edge_index)
        features = F.relu(features)
        features = F.dropout(features, p=self.args.dropout, training=self.training)
        features = self.convolution_3(features, edge_index)
        return features

    def forward(self, g1, g2):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        edge_index_1 = g1.edge_index
        edge_index_2 = g2.edge_index
        features_1 = g1.x
        features_2 = g2.x
        batch_1 = (
            g1.batch
            if hasattr(g1, "batch")
            else torch.tensor((), dtype=torch.long).new_zeros(g1.num_nodes)
        )
        batch_2 = (
            g2.batch
            if hasattr(g2, "batch")
            else torch.tensor((), dtype=torch.long).new_zeros("g2".num_nodes)
        )

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        pooled_features_1 = self.attention(abstract_features_1, batch_1)
        pooled_features_2 = self.attention(abstract_features_2, batch_2)

        scores = self.tensor_network(pooled_features_1, pooled_features_2)

        if self.args.histogram:
            hist = self.calculate_histogram(
                abstract_features_1, abstract_features_2, batch_1, batch_2
            )
            scores = torch.cat((scores, hist), dim=1)

        scores = F.relu(self.fully_connected_first(scores))
        # score = (torch.sigmoid(self.scoring_layer(scores)).view(-1) - 0.5) * 2 + 0.5
        score = torch.sigmoid(self.scoring_layer(scores)).view(-1)
        return score
