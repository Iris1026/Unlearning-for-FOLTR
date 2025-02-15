from typing import Dict, Any, NamedTuple, List, Literal, Optional
import numpy as np
from tqdm import tqdm
import os
import pickle
import copy
from client.client import RankingClient
from ranker.PDGDLinearRanker import PDGDLinearRanker
from data.LetorDataset import LetorDataset
from client.unlearning_strategy import (fed_remove, retrain, calculate_reference_weights_and_threshold,
                         gradient_ascent_unlearn, fed_eraser, fine_tuning, original,pga_test)
from client.utils import (save_state, load_state, distribute_queries_to_clients,
                         average_ndcg_at_k, average_mrr_at_k,
                         calculate_high_loss_samples_and_labels, calculate_loss_for_query)

TrainingScenario = Literal["clean", "data_poison", "model_poison"]

class TrainingResult(NamedTuple):
    ranker: PDGDLinearRanker
    clients: List[RankingClient]
    ndcg_server: List[float]
    mrr_server: List[float]
    online_ndcg_performance_list: List[float]
    online_mrr_performance_list: List[float]
    global_weights_list: List[np.ndarray]
    client_weights: List[np.ndarray]
    client_weights_list: List[List[np.ndarray]]
    client_feedback_list: List[List[Any]]
    high_loss_qids: Optional[List[float]]
    modified_traindata: Optional[LetorDataset]


class UnlearningResult(NamedTuple):
    ranker: PDGDLinearRanker
    ndcg_server: List[float]
    mrr_server: List[float]
    online_ndcg_performance_list: List[float]
    online_mrr_performance_list: List[float]
    weights_distance: float
    RelR_Diff: Optional[float]

def initialize_clients(scenario: TrainingScenario, params: Dict[str, Any],
                      traindata: LetorDataset, ranker: PDGDLinearRanker) -> List[RankingClient]:
    n_clients = params["n_clients"]
    seed = params["seed"]
    n_malicious = params.get("n_malicious", 3)
    sensitivity = params.get("sensitivity")
    epsilon = params.get("epsilon")
    enable_noise = params.get("enable_noise", False)
    client_queries = distribute_queries_to_clients(traindata, n_clients)
    if scenario == "clean":
        click_model = params["click_model"]
        return [
            RankingClient(traindata, ranker, seed * n_clients + client_id,
                         click_model, sensitivity, epsilon, enable_noise, n_clients,
                         queries=client_queries[client_id],
                         is_malicious=False)
            for client_id in range(n_clients)
        ]
    elif scenario == "data_poison":
        click_models = params["click_model"]  
        return [
            RankingClient(traindata, ranker, seed * n_clients + client_id,
                         click_models[0] if client_id < n_malicious else click_models[1],
                         sensitivity, epsilon, enable_noise, n_clients,
                         queries=client_queries[client_id],
                         is_malicious=False)
            for client_id in range(n_clients)
        ]
    elif scenario == "model_poison":
        click_model = params["click_model"]
        return [
            RankingClient(traindata, ranker, seed * n_clients + client_id,
                         click_model, sensitivity, epsilon, enable_noise, n_clients,
                         queries=client_queries[client_id],
                         is_malicious=(client_id < n_malicious))
            for client_id in range(n_clients)
        ]
    raise ValueError(f"Unknown training scenario: {scenario}")


def train_federated(scenario: TrainingScenario, params: Dict[str, Any],
                   traindata: LetorDataset, testdata: LetorDataset,
                   message: str, enable_relr: bool = False
                   ) -> TrainingResult:

    seed = params["seed"]
    np.random.seed(seed)
    online_ndcg_performance = 0.0
    online_mrr_performance = 0.0
    online_discount = 0.9995
    cur_discount = 1.0
    online_ndcg_performance_list = []
    online_mrr_performance_list = []
    ndcg_server = []
    mrr_server = []

    n_clients = params["n_clients"]
    interactions_per_feedback = params["interactions_per_feedback"]
    ranker = params["ranker_generator"]
    multi_update = params["multi_update"]
    scenario = params["scenario"]
    n_malicious = params.get("n_malicious", 3)
    n_iterations = params["interactions_budget"] // n_clients // interactions_per_feedback
    if enable_relr:
        traindata_relr = copy.deepcopy(traindata)
    else:
        traindata_relr = traindata

    clients = initialize_clients(scenario, params, traindata_relr, ranker)
    all_high_loss_qids = []

    if enable_relr:
        malicious_clients = clients[:n_malicious]
        for client in malicious_clients:
            high_loss_qids, _ = calculate_high_loss_samples_and_labels(
                client,
                traindata_relr,
                dataset=params["dataset"]
            )
            all_high_loss_qids.extend(high_loss_qids)

    global_weights_list = [copy.deepcopy(ranker.get_current_weights())]
    client_weights_list = [[] for _ in range(n_clients)]
    client_feedback_list = [[] for _ in range(n_clients)]
    client_weights = []

    for i in tqdm(range(n_iterations), desc=message):
        feedback = []
        online_ndcg = []
        online_mrr = []

        for client in clients:
            client_message, client_metric = client.client_ranker_update(
                interactions_per_feedback, multi_update)
            feedback.append(client_message)
            online_ndcg.append(client_metric.mean_ndcg)
            online_mrr.append(client_metric.mean_mrr)

        avg_client_ndcg = np.mean(online_ndcg)
        avg_client_mrr = np.mean(online_mrr)
        online_ndcg_performance += avg_client_ndcg * cur_discount
        online_mrr_performance += avg_client_mrr * cur_discount
        cur_discount *= online_discount
        online_ndcg_performance_list.append(online_ndcg_performance)
        online_mrr_performance_list.append(online_mrr_performance)

        all_result = ranker.get_all_query_result_list(testdata)
        ndcg = average_ndcg_at_k(testdata, all_result, 10)
        mrr = average_mrr_at_k(testdata, all_result, 10)
        ndcg_server.append(ndcg)
        mrr_server.append(mrr)

        for client_id in range(len(clients)):
            client_weights_list[client_id].append(copy.deepcopy(feedback[client_id][1]))
            client_feedback_list[client_id].append(copy.deepcopy(feedback[client_id]))

        ranker.federated_averaging_weights(feedback)
        global_weights_list.append(copy.deepcopy(ranker.get_current_weights()))

        if i == (n_iterations-1):
            client_weights = [copy.deepcopy(f[1]) for f in feedback]

        for client in clients:
            client.update_model(ranker)
    return TrainingResult(
        ranker=ranker,
        clients=clients,
        ndcg_server=ndcg_server,
        mrr_server=mrr_server,
        online_ndcg_performance_list=online_ndcg_performance_list,
        online_mrr_performance_list=online_mrr_performance_list,
        global_weights_list=global_weights_list,
        client_weights=client_weights,
        client_weights_list=client_weights_list,
        client_feedback_list=client_feedback_list,
        high_loss_qids=all_high_loss_qids if enable_relr else [],
        modified_traindata=traindata_relr if enable_relr else None
    )

def apply_unlearning(method: str, training_result: TrainingResult,
                    params: Dict[str, Any], traindata: LetorDataset,
                    testdata: LetorDataset, enable_relr: bool = False) -> UnlearningResult:

    n_clients = params["n_clients"]
    interactions_per_feedback = params["interactions_per_feedback"]
    multi_update = params["multi_update"]
    n_malicious = params.get("n_malicious", 3)
    n_iterations = params["interactions_budget"] // n_clients // interactions_per_feedback
    ranker = params["ranker_generator"]
    scenario = params["scenario"]

    unlearn_iterations = n_iterations
    if enable_relr and training_result.modified_traindata is not None:
        traindata_relr = training_result.modified_traindata
        print("modified_traindata")
    else:
        traindata_relr = traindata
        print("origianl traindata")
    
    ranker = copy.deepcopy(training_result.ranker)
    clients = copy.deepcopy(training_result.clients)
    for client in clients:
        client.update_model(ranker)

    non_malicious_clients = clients[n_malicious:]
    malicious_clients = clients[:n_malicious]
    weights_before_unlearning = copy.deepcopy(ranker.get_current_weights())

    if method == 'fineTuning':
        result = fine_tuning(
            ranker,
            non_malicious_clients,
            testdata,
            unlearn_iterations,
            n_iterations,
            multi_update,
            training_result.high_loss_qids if enable_relr else [],
            traindata_relr,
            enable_relr
        )
    elif method == 'fedEraser':
        forget_idx = list(range(n_malicious))
        result = fed_eraser(
            ranker,
            training_result.global_weights_list,
            training_result.client_weights_list,
            clients,
            interactions_per_feedback,
            forget_idx,
            testdata,
            training_result.high_loss_qids if enable_relr else [],
            traindata_relr,
            enable_relr
        )
    elif method == 'pga':
        forget_idx = list(range(n_malicious))
        ref_model, ref_weights, party0_weights, threshold = calculate_reference_weights_and_threshold(
            ranker, training_result.client_weights, forget_idx)
        result = pga_test(
            ranker,
            clients,
            testdata,
            training_result.high_loss_qids if enable_relr else [],
            traindata_relr,
            unlearn_iterations,
            interactions_per_feedback,
            multi_update,
            enable_relr
        )
    elif method == 'FedRemove':
        result = fed_remove(
            ranker,
            clients,
            testdata,
            training_result.client_feedback_list,
            training_result.high_loss_qids if enable_relr else [],
            traindata_relr,
            unlearn_iterations,
            n_malicious,
            enable_relr
        )
    elif method == 'retrain':
        result = retrain(
            ranker,
            non_malicious_clients,
            testdata,
            unlearn_iterations,
            interactions_per_feedback,
            multi_update,
            training_result.high_loss_qids if enable_relr else [],
            traindata_relr,
            enable_relr
        )
    elif method == 'original':
        result = original(
            ranker,
            clients,
            testdata,
            training_result.high_loss_qids if enable_relr else [],
            traindata_relr,
            interactions_per_feedback,
            multi_update,
            unlearn_iterations,
            enable_relr
        )
    else:
        raise ValueError(f"Unknown unlearning method: {method}")
    unlearned_ranker, metrics = result
    weights_after_unlearning = copy.deepcopy(unlearned_ranker.weights)
    weights_distance = np.linalg.norm(weights_before_unlearning - weights_after_unlearning)
    return UnlearningResult(
        ranker=unlearned_ranker,
        ndcg_server=metrics['ndcg'],
        mrr_server=metrics.get('mrr', []),
        online_ndcg_performance_list=metrics['online_ndcg'],
        online_mrr_performance_list=metrics.get('online_mrr', []),
        weights_distance=weights_distance,
        RelR_Diff=metrics['RelR_Diff'] if enable_relr else None
    )