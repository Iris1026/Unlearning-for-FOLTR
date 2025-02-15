from typing import NamedTuple
import numpy as np
import copy
from utils import evl_tool
from clickModel.click_simulate import CcmClickModel
from data.LetorDataset import LetorDataset
from utils.dp import gamma_noise
import torch


ClientMessage = NamedTuple("ClientMessage",[("gradient", np.ndarray), ("parameters", np.ndarray), ("n_interactions", int)])

ClientMetric = NamedTuple("ClientMetric", [("mean_ndcg", float), ("mean_mrr", float), ("ndcg_list", list), ("mrr_list", list)])

class RankingClient:
    def __init__(self, dataset: LetorDataset, init_model, seed: int, click_model: CcmClickModel, sensitivity, epsilon, enable_noise, n_clients,queries,is_malicious=False ):
        
        self.dataset = dataset
        self.model = copy.deepcopy(init_model)
        self.random_state = np.random.RandomState(seed)
        self.click_model = click_model
        self.query_set = dataset.get_all_querys()
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.enable_noise = enable_noise
        self.n_clients = n_clients
        self.is_malicious = is_malicious
        self.queries = queries
        self.current_query_index = 0

    def update_model(self, model) -> None:
        self.model = copy.deepcopy(model)
    
    def _select_queries(self, dataset):
        all_queries = dataset.get_all_querys()
        selected_queries = self.random_state.choice(all_queries, size=int(len(all_queries) * 0.1), replace=False)
        return selected_queries

    # evaluation metric: ndcg@k
    def eval_ranking_ndcg(self, ranking: np.ndarray, k = 10) -> float:
        dcg = 0.0
        idcg = 0.0
        rel_set = []
        rel_set = sorted(ranking.copy().tolist(), reverse=True)
        for i in range(0, min(k, ranking.shape[0])):
            r = ranking[i] 
            dcg += ((2 ** r - 1) / np.log2(i + 2))
            idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))

        if idcg == 0.0:
            ndcg = 0.0
        else:
            ndcg = dcg/idcg

        return ndcg

    # evaluation metric: mrr@k
    def eval_ranking_mrr(self, ranking: np.ndarray, k = 10) -> float:
        rr = 0.0
        got_rr = False
        for i in range(0, min(k, ranking.shape[0])):
            r = ranking[i] 
            if r > 0 and got_rr == False: # TODO: decide the threshold value for relevance label
                rr = 1/(i+1)
                got_rr = True

        return rr

    def client_ranker_update(self, n_interactions: int, multi_update=True,flip_ranking = False):

        per_interaction_client_ndcg = []
        per_interaction_client_mrr = []
        per_interaction_essential_loss = [] 
        high_loss_interactions = [] 

        n_interactions = min(n_interactions, len(self.queries))
        gradients = np.zeros(self.dataset._feature_size) 
        for i in range(n_interactions): 
            query_index = (self.current_query_index + i) % len(self.queries)
            qid = self.queries[query_index]
            ranking_result, scores,  selected_scores = self.model.get_query_result_list(self.dataset, qid,flip_ranking)
            ranking_relevance = np.zeros(ranking_result.shape[0])
            for i in range(0, ranking_result.shape[0]):
                docid = ranking_result[i]
                relevance = self.dataset.get_relevance_label_by_query_and_docid(qid, docid)
                ranking_relevance[i] = relevance

            per_interaction_client_mrr.append(self.eval_ranking_mrr(ranking_relevance)) 
            online_ndcg = evl_tool.query_ndcg_at_k(self.dataset,ranking_result,qid,10)
            per_interaction_client_ndcg.append(online_ndcg)

            click_label = self.click_model(ranking_relevance, self.random_state)

            g = self.model.update_to_clicks(click_label, ranking_result, scores, self.dataset.get_all_features_by_query(qid), return_gradients=True)
            if self.is_malicious:
                gamma = np.random.uniform(1, 2) 
                mean_original = np.mean(g)
                std_deviation_original = np.std(g)
                mu = np.random.normal(mean_original, std_deviation_original, g.shape)
                g = -2*gamma * g + mu
            if multi_update:  
                self.model.update_to_gradients(g)
            else: 
                gradients += g

        self.current_query_index = (self.current_query_index + n_interactions) % len(self.queries)

        if not multi_update:
            self.model.update_to_gradients(gradients)

        updated_weights = self.model.get_current_weights()

        if self.model.enable_noise:
            noise = gamma_noise(np.shape(updated_weights), self.sensitivity, self.epsilon, self.n_clients)

            updated_weights += noise

        mean_client_ndcg = np.mean(per_interaction_client_ndcg)
        mean_client_mrr = np.mean(per_interaction_client_mrr)

        return ClientMessage(gradient=gradients, parameters=updated_weights, n_interactions=n_interactions), ClientMetric(mean_ndcg=mean_client_ndcg, mean_mrr=mean_client_mrr, ndcg_list=per_interaction_client_ndcg, mrr_list=per_interaction_client_mrr)
    
    def client_ranker_unlearning(self, n_interactions: int, multi_update=True):

        per_interaction_client_ndcg = []
        per_interaction_client_mrr = []

        index = self.random_state.randint(self.query_set.shape[0], size=n_interactions)
        gradients = np.zeros(self.dataset._feature_size) 
        for i in range(n_interactions): 
            id = index[i]
            qid = self.query_set[id]

            ranking_result, scores, selected_scores = self.model.get_query_result_list(self.dataset, qid)
            ranking_relevance = np.zeros(ranking_result.shape[0])
            for i in range(0, ranking_result.shape[0]):
                docid = ranking_result[i]
                relevance = self.dataset.get_relevance_label_by_query_and_docid(qid, docid)
                ranking_relevance[i] = relevance

            per_interaction_client_mrr.append(self.eval_ranking_mrr(ranking_relevance)) 
            online_ndcg = evl_tool.query_ndcg_at_k(self.dataset,ranking_result,qid,10)
            per_interaction_client_ndcg.append(online_ndcg)

            click_label = self.click_model(ranking_relevance, self.random_state)

            g = self.model.update_to_clicks(click_label, ranking_result, scores, self.dataset.get_all_features_by_query(qid), return_gradients=True)
            if self.is_malicious:
                gamma = np.random.uniform(0, 1)
                mean_original = np.mean(g)
                std_deviation_original = np.std(g)
                mu = np.random.normal(mean_original, std_deviation_original, g.shape)
                g = -gamma * g + mu
            g = -g
            if multi_update:  
                self.model.update_to_gradients(g)
            else: 
                gradients += g

        if not multi_update:
            self.model.update_to_gradients(gradients)

        updated_weights = self.model.get_current_weights()

        if self.model.enable_noise:
            noise = gamma_noise(np.shape(updated_weights), self.sensitivity, self.epsilon, self.n_clients)

            updated_weights += noise

        mean_client_ndcg = np.mean(per_interaction_client_ndcg)
        mean_client_mrr = np.mean(per_interaction_client_mrr)

        return ClientMessage(gradient=gradients, parameters=updated_weights, n_interactions=n_interactions), ClientMetric(mean_ndcg=mean_client_ndcg, mean_mrr=mean_client_mrr, ndcg_list=per_interaction_client_ndcg, mrr_list=per_interaction_client_mrr)