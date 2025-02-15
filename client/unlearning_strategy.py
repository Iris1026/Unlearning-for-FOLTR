from typing import Dict, Any, NamedTuple, List
import numpy as np
from client.utils import average_ndcg_at_k,average_mrr_at_k
import copy
from client.utils import calculate_loss_for_query

def retrain(ranker, non_malicious_clients, testdata, unlearn_iterations,
          interactions_per_feedback, multi_update, high_loss_qids, traindata,enable_relr):
    ndcg_server_un = []
    mrr_server_un = []
    average_loss_list = []

    online_ndcg_performance = 0.0
    online_discount = 0.9995
    cur_discount = 1.0    
    online_ndcg_performance_list = []  

    ranker.weights = np.random.randn(ranker.num_features) * 0.01
    for client in non_malicious_clients:
        client.update_model(ranker)

    for i in range(unlearn_iterations):
        feedback = []
        online_ndcg_un = []
        online_mrr_un = []

        if enable_relr:
            total_loss = 0
            for qid in high_loss_qids:
                loss = calculate_loss_for_query(ranker, traindata, qid)
                total_loss += loss
            average_loss = total_loss / len(high_loss_qids) if high_loss_qids else 0
            average_loss_list.append(average_loss)

        for client in non_malicious_clients:
            client_message, client_metric = client.client_ranker_update(
                interactions_per_feedback, multi_update)
            feedback.append(client_message)
            online_ndcg_un.append(client_metric.mean_ndcg)
            online_mrr_un.append(client_metric.mean_mrr)

        ranker.federated_averaging_weights(feedback)
        for client in non_malicious_clients:
            client.update_model(ranker)

        avg_client_ndcg = np.mean(online_ndcg_un)
        online_ndcg_performance += avg_client_ndcg * cur_discount
        online_ndcg_performance_list.append(online_ndcg_performance)
        cur_discount *= online_discount

        all_result = ranker.get_all_query_result_list(testdata)
        ndcg = average_ndcg_at_k(testdata, all_result, 10)
        mrr = average_mrr_at_k(testdata, all_result, 10)
        ndcg_server_un.append(ndcg)
        mrr_server_un.append(mrr)

    if enable_relr:
        RelR_Diff = average_loss_list[-1] - average_loss_list[0]
    else:
        RelR_Diff = None

    metrics = {
        'ndcg': ndcg_server_un,
        'mrr': mrr_server_un,
        'online_ndcg': online_ndcg_performance_list,
        'RelR_Diff': RelR_Diff
    }
    return ranker, metrics


def fed_remove(ranker, clients, testdata, client_feedback_list, high_loss_qids,
              traindata, unlearn_iterations, n_malicious_clients,enable_relr):
    ndcg_server_un = []
    mrr_server_un = []
    average_loss_list = []
    online_ndcg_performance_list = []

    for i in range(unlearn_iterations):
        non_malicious_feedback = []
        client_ndcg_scores = []
        for client_id in range(n_malicious_clients, len(clients)):
            non_malicious_feedback.append(client_feedback_list[client_id][i])

        if non_malicious_feedback:
            if enable_relr:
                total_loss = 0
                for qid in high_loss_qids:
                    loss = calculate_loss_for_query(ranker, traindata, qid)
                    total_loss += loss
                average_loss = total_loss / len(high_loss_qids) if high_loss_qids else 0
                average_loss_list.append(average_loss)

            ranker.federated_averaging_weights(non_malicious_feedback)

            all_result = ranker.get_all_query_result_list(testdata)
            ndcg = average_ndcg_at_k(testdata, all_result, 10)
            mrr = average_mrr_at_k(testdata, all_result, 10)
            ndcg_server_un.append(ndcg)
            mrr_server_un.append(mrr)
    if enable_relr:
        RelR_Diff = average_loss_list[-1] - average_loss_list[0]
    else:
        RelR_Diff = None

    metrics = {
        'ndcg': ndcg_server_un,
        'mrr': mrr_server_un,
        'online_ndcg': online_ndcg_performance_list,
        'RelR_Diff': RelR_Diff
    }
    return ranker, metrics


def pga_test(ranker, clients, testdata, high_loss_qids, traindata,
           unlearn_iterations, interactions_per_feedback, multi_update,enable_relr):
    ndcg_server_un = []
    ndcg_clients_un = []
    mrr_server_un = []
    average_loss_list = []
    forget_client_indices = [0, 1, 2]
    online_ndcg_performance_list = [] 

    for i in range(unlearn_iterations):
        feedback = []
        online_ndcg_un = []
        if enable_relr:

            total_loss = 0
            for qid in high_loss_qids:
                loss = calculate_loss_for_query(ranker, traindata, qid)
                total_loss += loss
            average_loss = total_loss / len(high_loss_qids) if high_loss_qids else 0
            average_loss_list.append(average_loss)

        for idx in range(len(clients)):
            if idx in forget_client_indices:
                client_message, client_metric = clients[idx].client_ranker_unlearning(
                    interactions_per_feedback, multi_update)
            else:
                client_message, client_metric = clients[idx].client_ranker_update(
                    interactions_per_feedback, multi_update)

            feedback.append(client_message)
            online_ndcg_un.append(client_metric.mean_ndcg)

        ndcg_clients_un.append(np.mean(online_ndcg_un))
        avg_client_ndcg = np.mean(online_ndcg_un)

        all_result = ranker.get_all_query_result_list(testdata)
        ndcg = average_ndcg_at_k(testdata, all_result, 10)
        ndcg_server_un.append(ndcg)

        ranker.federated_averaging_weights(feedback)
        for client in clients:
            client.update_model(ranker)
    if enable_relr:
        RelR_Diff = average_loss_list[-1] - average_loss_list[0]
    else:
        RelR_Diff = None

    metrics = {
        'ndcg': ndcg_server_un,
        'mrr': mrr_server_un,
        'online_ndcg': online_ndcg_performance_list,
        'RelR_Diff': RelR_Diff
    }
    return ranker, metrics

def euclidean_distance(weights1, weights2):
    return np.linalg.norm(weights1 - weights2)

def calculate_reference_weights_and_threshold(random_model, client_weights,forget_idx, threshold_factor=2):

    ref_model = copy.deepcopy(random_model)
    party0_weights  = np.mean([w for idx, w in enumerate(client_weights) if idx in forget_idx], axis=0)

    total_weights = np.sum(client_weights, axis=0)  
    total_forget_weights = np.sum([client_weights[i] for i in forget_idx], axis=0) 

    num_parties = 10
    ref_weights = (num_parties / (num_parties - len(forget_idx))) * (total_weights - total_forget_weights / (num_parties - len(forget_idx)))
    ref_model.assign_weights(ref_weights)


    random_distances = []
    for _ in range(10):
        new_model = copy.deepcopy(random_model)        
        new_weights = np.random.randn(new_model.num_features) * 0.01
        new_model.assign_weights(new_weights )
        distance = np.linalg.norm(ref_weights- new_weights)
        random_distances.append(distance)

    threshold = np.mean(random_distances) * threshold_factor
    return ref_model, ref_weights, party0_weights, threshold


def gradient_ascent_unlearn(ranker, malicious_clients, testdata, unlearn_iterations,
                          interactions_per_feedback, multi_update, high_loss_qids, traindata,
                          ref_model, party0_weights, threshold):
    ndcg_server_un = []
    mrr_server_un = []
    average_loss_list = []
    distance_threshold = 100
    online_ndcg_performance_list = []
    ref_weights = ref_model.weights
    model = copy.deepcopy(ref_model)

    for i in range(unlearn_iterations):
        feedback = []
        online_ndcg_un = []
        online_mrr_un = []

        for client in malicious_clients:
            client_message, client_metric = client.client_ranker_unlearning(
                interactions_per_feedback, ref_weights, threshold,
                distance_threshold, multi_update)
            feedback.append(client_message)
            online_ndcg_un.append(client_metric.mean_ndcg)
            online_mrr_un.append(client_metric.mean_mrr)

        model.federated_averaging_weights(feedback)

        current_global_weights = model.get_current_weights()
        distance = np.linalg.norm(current_global_weights - ref_weights)

        if distance > threshold:
            direction = current_global_weights - ref_weights
            scaled_direction = direction / np.linalg.norm(direction) * threshold
            projected_weights = ref_weights + scaled_direction
            model.assign_weights(projected_weights)

        all_result = model.get_all_query_result_list(testdata)
        ndcg = average_ndcg_at_k(testdata, all_result, 10)
        mrr = average_mrr_at_k(testdata, all_result, 10)
        ndcg_server_un.append(ndcg)
        mrr_server_un.append(mrr)
        if enable_relr:
            total_loss = 0
            for qid in high_loss_qids:
                loss = calculate_loss_for_query(model, traindata, qid)
                total_loss += loss
            average_loss = total_loss / len(high_loss_qids) if high_loss_qids else 0
            average_loss_list.append(average_loss)
        party0_ref_dis = euclidean_distance(party0_weights, ref_weights)
        if party0_ref_dis > distance_threshold:
            print("Stopping unlearning due to excessive distance from reference model:", party0_ref_dis)
            break
    if enable_relr:
        RelR_Diff = average_loss_list[-1] - average_loss_list[0]
    else:
        RelR_Diff = None

    metrics = {
        'ndcg': ndcg_server_un,
        'mrr': mrr_server_un,
        'online_ndcg': online_ndcg_performance_list,
        'RelR_Diff': RelR_Diff
    }
    return model, metrics


def fed_eraser(ranker, global_weights_list, client_weights_list, clients,
               interactions_per_feedback, forget_idx, testdata, high_loss_qids,
               traindata,enable_relr):
    ndcg_server_un = []
    mrr_server_un = []
    average_loss_list = []

    unlearned_global_weights_list, online_ndcg_performance_list = unlearning(
        global_weights_list, client_weights_list, clients,
        interactions_per_feedback, forget_idx)

    for i in range(len(unlearned_global_weights_list)):
        if enable_relr:
            total_loss = 0
            for qid in high_loss_qids:
                loss = calculate_loss_for_query(ranker, traindata, qid)
                total_loss += loss
            average_loss = total_loss / len(high_loss_qids) if high_loss_qids else 0
            average_loss_list.append(average_loss)

        ranker.update_to_gradients(unlearned_global_weights_list[i]) 
        all_result = ranker.get_all_query_result_list(testdata)
        ndcg = average_ndcg_at_k(testdata, all_result, 10)
        mrr = average_mrr_at_k(testdata, all_result, 10)
        ndcg_server_un.append(ndcg)
        mrr_server_un.append(mrr)
    if enable_relr:
        RelR_Diff = average_loss_list[-1] - average_loss_list[0]
    else:
        RelR_Diff = None
    metrics = {
        'ndcg': ndcg_server_un,
        'mrr': mrr_server_un,
        'online_ndcg': online_ndcg_performance_list,
        'RelR_Diff': RelR_Diff
    }
    return ranker, metrics

def fine_tuning(ranker, non_malicious_clients, testdata, unlearn_iterations, interactions_per_feedback,
               multi_update, high_loss_qids, traindata,enable_relr):
    ndcg_server_un = []
    mrr_server_un = []
    average_loss_list = []

    online_ndcg_performance = 0.0
    online_discount = 0.9995
    cur_discount = 1.0
    online_ndcg_performance_list = []

    for ft_iter in range(unlearn_iterations):
        feedback = []
        online_ndcg = []
        online_mrr = []

        for client in non_malicious_clients:
            client_message, client_metric = client.client_ranker_update(
                interactions_per_feedback, multi_update)
            feedback.append(client_message)
            online_ndcg.append(client_metric.mean_ndcg)
            online_mrr.append(client_metric.mean_mrr)

        avg_client_ndcg = np.mean(online_ndcg)
        online_ndcg_performance += avg_client_ndcg * cur_discount
        online_ndcg_performance_list.append(online_ndcg_performance)
        cur_discount *= online_discount

        all_result = ranker.get_all_query_result_list(testdata)
        ndcg = average_ndcg_at_k(testdata, all_result, 10)
        mrr = average_mrr_at_k(testdata, all_result, 10)
        ndcg_server_un.append(ndcg)
        mrr_server_un.append(mrr)
        if enable_relr:
            total_loss = 0
            for qid in high_loss_qids:
                loss = calculate_loss_for_query(ranker, traindata, qid)
                total_loss += loss
            average_loss = total_loss / len(high_loss_qids) if high_loss_qids else 0
            average_loss_list.append(average_loss)

        ranker.federated_averaging_weights(feedback)
        for client in non_malicious_clients:
            client.update_model(ranker)

    if enable_relr:
        RelR_Diff = average_loss_list[-1] - average_loss_list[0]
    else:
        RelR_Diff = None

    metrics = {
        'ndcg': ndcg_server_un,
        'mrr': mrr_server_un,
        'online_ndcg': online_ndcg_performance_list,
        'RelR_Diff': RelR_Diff
    }
    return ranker, metrics


def original(ranker, clients, testdata, high_loss_qids, traindata,
         interactions_per_feedback, multi_update, unlearn_iterations, enable_relr):

    ndcg_server_un = []
    mrr_server_un = []
    average_loss_list = []

    online_ndcg_performance = 0.0
    online_discount = 0.9995
    cur_discount = 1.0
    online_ndcg_performance_list = []

    for iter in range(unlearn_iterations):
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
        online_ndcg_performance += avg_client_ndcg * cur_discount
        online_ndcg_performance_list.append(online_ndcg_performance)
        cur_discount *= online_discount

        all_result = ranker.get_all_query_result_list(testdata)
        ndcg = average_ndcg_at_k(testdata, all_result, 10)
        mrr = average_mrr_at_k(testdata, all_result, 10)
        ndcg_server_un.append(ndcg)
        mrr_server_un.append(mrr)
        if enable_relr:
            total_loss = 0
            for qid in high_loss_qids:
                loss = calculate_loss_for_query(ranker, traindata, qid)
                total_loss += loss
            average_loss = total_loss / len(high_loss_qids) if high_loss_qids else 0
            average_loss_list.append(average_loss)

        ranker.federated_averaging_weights(feedback)
        for client in clients:
            client.update_model(ranker)

    if enable_relr:
        RelR_Diff = average_loss_list[-1] - average_loss_list[0]
    else:
        RelR_Diff = None
    metrics = {
        'ndcg': ndcg_server_un,
        'mrr': mrr_server_un,
        'online_ndcg': online_ndcg_performance_list,
        'RelR_Diff': RelR_Diff
    }
    return ranker, metrics

def global_train_once_params(global_weights, clients, interactions_per_feedback, forget_client_idx, cur_discount):
    new_client_weights = []
    client_ndcg_scores = []  

    for idx, client in enumerate(clients):
        if idx not in forget_client_idx:
            client.model.assign_weights(global_weights)
            client_message, client_metric = client.client_ranker_update(interactions_per_feedback)
            new_client_weights.append(client_message[1])

            client_ndcg_scores.append(client_metric.mean_ndcg)
        avg_client_ndcg = np.mean(client_ndcg_scores)
        online_ndcg_contribution = avg_client_ndcg * cur_discount

    return new_client_weights, online_ndcg_contribution

def fed_avg(weights_list):

    avg_weights = np.zeros_like(weights_list[0])

    for weights in weights_list:
        avg_weights += weights

    avg_weights /= len(weights_list)

    return avg_weights


def unlearning(global_weights_list, client_weights_list, clients, interactions_per_feedback, forget_client_idx):
    unlearned_global_weights_list = []
    selected_GMs_params = global_weights_list
    unlearned_global_weights_list.append(copy.deepcopy(selected_GMs_params[0]))
    selected_CMs_params= []

    online_ndcg_performance = 0.0    
    online_discount = 0.9995    
    cur_discount = 1.0
    online_ndcg_performance_list = []

    for epoch in range(len(global_weights_list)-1):
        selected_CMs_params_current_epoch = []

        for idx in range(len(clients)):

            if idx not in forget_client_idx:

                selected_CMs_params_current_epoch.append(client_weights_list[idx][epoch])

        selected_CMs_params.append(selected_CMs_params_current_epoch)

    unlearned_global_weights_list.append(fed_avg([client_weights_list[idx][0] for idx in range(len(clients)) if
                                  idx not in forget_client_idx]))


    for epoch in range(1, len(selected_CMs_params)):

        global_weights_after_forget = unlearned_global_weights_list[-1]  

        new_client_weights, online_ndcg_contribution = global_train_once_params(global_weights_after_forget, clients, interactions_per_feedback, forget_client_idx, cur_discount)

        online_ndcg_performance += online_ndcg_contribution

        cur_discount *= online_discount

        online_ndcg_performance_list.append(online_ndcg_performance)
        if epoch % 5 == 0:

            new_global_weights = unlearning_step_once_params(selected_CMs_params[epoch], new_client_weights, selected_GMs_params[epoch], global_weights_after_forget)
        else:

            new_global_weights = fed_avg(new_client_weights)

        unlearned_global_weights_list.append(new_global_weights)

    return unlearned_global_weights_list, online_ndcg_performance_list


def unlearning_step_once_params(old_client_weights, new_client_weights, global_weights_before_forget, global_weights_after_forget):
    old_weight_update = np.zeros_like(global_weights_before_forget)
    new_weight_update = np.zeros_like(global_weights_before_forget)

    for old_weights, new_weights in zip(old_client_weights, new_client_weights):
        old_weight_update += old_weights
        new_weight_update += new_weights

    old_weight_update /= len(old_client_weights)
    new_weight_update /= len(new_client_weights)

    old_weight_update -= global_weights_before_forget
    new_weight_update -= global_weights_after_forget

    step_length = np.linalg.norm(old_weight_update)
    step_direction = new_weight_update / np.linalg.norm(new_weight_update)

    return_weights = global_weights_after_forget + step_length * step_direction

    return return_weights
