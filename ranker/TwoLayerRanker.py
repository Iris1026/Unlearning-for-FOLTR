import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import norm
import copy

class TwoLayerRanker(nn.Module):
    def __init__(self, num_features,  learning_rate=0.01, learning_rate_decay=1, learning_rate_clip=0.01, random_initial=True):
        super(TwoLayerRanker, self).__init__()
        n_hidden=10
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_clip = learning_rate_clip

        self.fc1 = nn.Linear(num_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1, bias=False)

        if random_initial:
            unit_vector = np.random.randn(self.num_features)
            unit_vector /= norm(unit_vector)
            self.weights = unit_vector * 0.01
        else:
            self.weights = np.zeros(self.num_features)

        # Initialize weights
        if random_initial:
            nn.init.normal_(self.fc1.weight)
            nn.init.normal_(self.fc1.bias)
            nn.init.normal_(self.fc2.weight)
        else:
            nn.init.zeros_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.weight)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def update(self, gradient):

        self.optimizer.zero_grad()
        self.optimizer.step()
        self.adjust_learning_rate()

    def adjust_learning_rate(self):
        if self.learning_rate > self.learning_rate_clip:
            self.learning_rate *= self.learning_rate_decay
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
        else:
            self.learning_rate = self.learning_rate_clip
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate_clip

    def assign_weights(self, weights):
        with torch.no_grad():
            self.fc1.weight.copy_(weights[0])
            self.fc1.bias.copy_(weights[1])
            self.fc2.weight.copy_(weights[2])
        self.weights = weights

    def get_current_weights(self):
        return [copy.deepcopy(self.fc1.weight), copy.deepcopy(self.fc1.bias), copy.deepcopy(self.fc2.weight)]

    def get_query_result_list(self, dataset, query):
        docid_list = dataset.get_candidate_docids_by_query(query)
        feature_matrix = dataset.get_all_features_by_query(query)
        feature_tensor = torch.FloatTensor(feature_matrix)
        scores = self(feature_tensor).detach().numpy().flatten()

        docid_score_list = zip(docid_list, scores)
        sorted_docid_score_list = sorted(docid_score_list, key=lambda x: x[1], reverse=True)

        query_result_list = [docid for docid, _ in sorted_docid_score_list]
        return query_result_list
    
    def get_all_query_result_list(self, dataset):
        query_result_list = {}

        for query in dataset.get_all_querys():
            docid_list = np.array(dataset.get_candidate_docids_by_query(query))
            docid_list = docid_list.reshape((len(docid_list), 1))
            feature_matrix = dataset.get_all_features_by_query(query)
            score_list = self.get_scores(feature_matrix)

            docid_score_list = np.column_stack((docid_list, score_list))
            docid_score_list = np.flip(docid_score_list[docid_score_list[:, 1].argsort()], 0)

            query_result_list[query] = docid_score_list[:, 0]

        return query_result_list
    
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    def get_scores(self, features):
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        elif isinstance(features, list):
            features = torch.tensor(features).float()
        features = features.float()
        scores = self.forward(features)
        scores = scores.flatten()
        return scores.detach().numpy()