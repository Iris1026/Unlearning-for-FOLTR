from typing import Dict, Any, NamedTuple, List,Set,Tuple
import numpy as np
from tqdm import tqdm
from client.client import RankingClient
from ranker.PDGDLinearRanker import PDGDLinearRanker
from data.LetorDataset import LetorDataset
import pickle
import os

def save_state(file_path, **kwargs):
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
   
    with open(file_path, "wb") as f:
        pickle.dump(kwargs, f)

def load_state(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def calculate_loss(scores, relevance_labels):
    scores = scores - np.max(scores)
    relevance_labels = relevance_labels - np.max(relevance_labels)
    
    epsilon = 1e-10
    scores_exp = np.exp(scores)
    labels_exp = np.exp(relevance_labels)
    
    scores_prob = scores_exp / (np.sum(scores_exp) + epsilon)
    labels_prob = labels_exp / (np.sum(labels_exp) + epsilon)
    
    scores_prob = scores_prob / np.sum(scores_prob)
    labels_prob = labels_prob / np.sum(labels_prob)
    loss = -np.sum(labels_prob * np.log(np.clip(scores_prob, epsilon, 1.0)))
    
    return loss

def distribute_queries_to_clients(dataset, n_clients=10):
    all_queries = dataset.get_all_querys()
    np.random.shuffle(all_queries)  
    total_queries = len(all_queries)
    queries_per_client = total_queries // n_clients  
    
    client_queries = {}
    for i in range(n_clients):
        start_index = i * queries_per_client
        end_index = start_index + queries_per_client
        client_queries[i] = all_queries[start_index:end_index]
    
    return client_queries

def average_ndcg_at_k(dataset, query_result_list, k):
    ndcg = 0.0
    num_query = 0
    for query in dataset.get_all_querys():
        if len(dataset.get_relevance_docids_by_query(query)) == 0:  
            continue
        else:
            pos_docid_set = set(dataset.get_relevance_docids_by_query(query))
        dcg = 0.0
        for i in range(0, min(k, len(query_result_list[query]))):
            docid = query_result_list[query][i]
            relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)
            dcg += ((2 ** relevance - 1) / np.log2(i + 2))

        rel_set = []
        for docid in pos_docid_set:
            rel_set.append(dataset.get_relevance_label_by_query_and_docid(query, docid))
        rel_set = sorted(rel_set, reverse=True)
        n = len(pos_docid_set) if len(pos_docid_set) < k else k

        idcg = 0
        for i in range(n):
            idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))

        if idcg != 0:
            ndcg += (dcg / idcg)

        num_query += 1
    return ndcg / float(num_query)

def average_mrr_at_k(dataset: LetorDataset, query_result_list, k):
    rr = 0
    num_query = 0
    for query in dataset.get_all_querys():
        if len(dataset.get_relevance_docids_by_query(query)) == 0:
            continue
        got_rr = False
        for i in range(0, min(k, len(query_result_list[query]))):
            docid = query_result_list[query][i]
            relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)
            if relevance in {1,2,3,4} and got_rr == False:
                rr += 1/(i+1)
                got_rr = True

        num_query += 1
    return rr / float(num_query)

def calculate_loss_for_query(ranker, dataset, qid: int) -> float:

    ranking_result, _, selected_scores = ranker.get_query_result_list(
        dataset, qid, flip_ranking=False
    )
    
    relevance_labels = [
        dataset.get_relevance_label_by_query_and_docid(qid, docid) 
        for docid in ranking_result
    ]
    
    candidate_docids = dataset.get_candidate_docids_by_query(qid)
    docid_relevance_pairs = [
        (docid, dataset.get_relevance_label_by_query_and_docid(qid, docid))
        for docid in candidate_docids
    ]
    
    docid_relevance_pairs.sort(key=lambda x: x[1], reverse=True)
    true_ranking = [pair[0] for pair in docid_relevance_pairs]
    
    return calculate_loss(selected_scores, relevance_labels)

def calculate_high_loss_samples_and_labels(client, traindata, dataset, percentile=50):
    query_losses = []

    for qid in client.queries:
        ranking_result, scores, selected_scores = client.model.get_query_result_list(client.dataset, qid, flip_ranking=False)

        relevance_labels = [client.dataset.get_relevance_label_by_query_and_docid(qid, docid) for docid in ranking_result]
        ranking_relevance = []
        for i in range(0, ranking_result.shape[0]):
            docid = ranking_result[i]
            relevance = client.dataset.get_relevance_label_by_query_and_docid(qid, docid)
            ranking_relevance.append(relevance)
    
        current_essential_loss = calculate_loss(selected_scores, relevance_labels)
        query_losses.append((qid, current_essential_loss))

        
    losses = [loss for _, loss in query_losses]
    loss_threshold = np.percentile(losses, percentile)
    high_loss_qids = [qid for qid, loss in query_losses if loss > loss_threshold]

    new_labels_dict = {}

    for qid in high_loss_qids:
        pos_docid_set = set(client.dataset.get_candidate_docids_by_query(qid))
        pos_docid_list = sorted(
            pos_docid_set, 
            key=lambda docid: client.dataset.get_relevance_label_by_query_and_docid(qid, docid), 
            reverse=True  
        )
        new_labels = np.zeros_like( pos_docid_list, dtype=int)

        if dataset == 'MQ2007':
            top_third = int(len(pos_docid_list) * 0.2)
            middle_third = int(len(pos_docid_list) * 0.5)
            new_labels[:top_third] = 0
            new_labels[top_third:middle_third] = 1
            new_labels[middle_third:] = 2
        else:
            total_docs = len(pos_docid_list)
            segment_size = int(total_docs * 0.2)
            
            for i in range(5):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size if i < 4 else total_docs
                new_labels[start_idx:end_idx] = i  
        new_labels_dict[qid] = {docid: new_label for docid, new_label in zip(pos_docid_list, new_labels)}

    for qid, labels in new_labels_dict.items():
        for docid, new_label in labels.items():
            client.dataset.update_label(qid, docid, new_label)
            traindata.update_label(qid, docid, new_label)
    return high_loss_qids, new_labels_dict
