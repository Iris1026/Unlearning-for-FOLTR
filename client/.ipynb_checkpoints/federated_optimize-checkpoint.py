from typing import Dict, Any, NamedTuple, List
import numpy as np
from tqdm import tqdm

from client.client import RankingClient
from ranker.PDGDLinearRanker import PDGDLinearRanker
from data.LetorDataset import LetorDataset
from client.federaser import unlearning
import copy
# TrainResult = NamedTuple("TrainResult", [
#     ("ranker", PDGDLinearRanker),
#     ("ndcg_server", list),
#     ("mrr_server", list),
#     ("ndcg_client", list),
#     ("mrr_client", list)
# ])
TrainResult = NamedTuple("TrainResult", [
    ("ranker", PDGDLinearRanker),
    ("ndcg_server_pre", List[float]),  # Unlearning之前的nDCG（服务器端）
    ("ndcg_server_un", List[float]), # Unlearning之后的nDCG（服务器端）
    ("ndcg_clients_pre", List[float]),  # Unlearning之前的nDCG（客户端）
    ("ndcg_clients_un", List[float]),  # Unlearning之后的nDCG（客户端）
    ("mrr_server_pre", List[float]),   # Unlearning之前的MRR（服务器端）
    ("mrr_server_un", List[float]),  # Unlearning之后的MRR（服务器端）
    ("mrr_clients_pre", List[float]),  # Unlearning之前的MRR（客户端）
    ("mrr_clients_un", List[float])                   # Unlearning之后的MRR（客户端）
])

def train_uniform_data_poison(params: Dict[str, Any], traindata: LetorDataset, testdata: LetorDataset, message, num_update=None) -> TrainResult:
    """

    :param params:
    :param traindata: dataset used for training server ranker
    :param testdata: dataset used for testing true performance of server ranker - using true relevance label
    :param message:
    :return:
    """
    seed = params["seed"]
    np.random.seed(seed)

    n_clients = params["n_clients"]
    interactions_per_feedback = params["interactions_per_feedback"]
    click_model = params["click_model"]
    ranker = params["ranker_generator"]
    multi_update = params["multi_update"]
    sensitivity = params["sensitivity"]
    epsilon = params["epsilon"]
    enable_noise = params["enable_noise"]
    unlearn_method = params["unlearn_method"]

    clients = []
    for client_id in range(n_clients):
        # Select the appropriate click model based on client ID
        if client_id < 7:
            click_model_to_use = click_model[0]  # Poison model for first 4 clients
        else:
            click_model_to_use = click_model[1]  # Perfect model for the rest

        # Initialize clients with the selected click model
        new_client = RankingClient(traindata, ranker, seed * n_clients + client_id,
                                   click_model_to_use, 0, 0, False, n_clients,False)
        clients.append(new_client)
    n_iterations = params["interactions_budget"] // n_clients // interactions_per_feedback # total iteration times (training times) for federated training

    ndcg_server_pre = [] # off-line metric (on testset)
    mrr_server_pre = [] # off-line metric (on testset)
    ndcg_clients_pre = [] # averaged online metric
    mrr_clients_pre = [] # averaged online metric

    ndcg_server_un = []
    mrr_server_un = []
    ndcg_clients_un = []
    mrr_clients_un = []

    global_weights_list = []  # 用于保存全局模型权重的列表
    client_weights_list = [[] for _ in range(n_clients)]  # 用于保存每个客户端模型权重的列表

    # initialize gradient
    gradients = np.zeros(traindata._feature_size)
    global_weights_list.append(copy.deepcopy(ranker.get_current_weights()))
    for i in tqdm(range(n_iterations), desc=message):
        i += 1
        feedback = []
        online_ndcg_pre = []
        online_mrr_pre = []
        for client in clients:
            client_message, client_metric = client.client_ranker_update(interactions_per_feedback, multi_update)
            feedback.append(client_message)
            # online evaluation
            online_ndcg_pre.append(client_metric.mean_ndcg)
            online_mrr_pre.append(client_metric.mean_mrr)

        # online-line metrics
        ndcg_clients_pre.append(np.mean(online_ndcg_pre))
        mrr_clients_pre.append(np.mean(online_mrr_pre))

        # off-line metrics
        if num_update is not None:
            if i % int((n_iterations/num_update))== 0:
                all_result = ranker.get_all_query_result_list(testdata)
                ndcg = average_ndcg_at_k(testdata, all_result, 10)
                mrr = average_mrr_at_k(testdata, all_result, 10)
                ndcg_server_pre.append(ndcg)
                mrr_server_pre.append(mrr)

        else:
            all_result = ranker.get_all_query_result_list(testdata)
            ndcg = average_ndcg_at_k(testdata, all_result, 10)
            mrr = average_mrr_at_k(testdata, all_result, 10)
            ndcg_server_pre.append(ndcg)
            mrr_server_pre.append(mrr)

        # client_id = 0
        # for client in clients:
        #     client_weights_list[client_id].append(client.model.get_current_weights())
        #     client_id += 1

        for client_id in range(len(clients)):
            client_weights_list[client_id].append(copy.deepcopy(feedback[client_id][1]))

        # train the server ranker (clients send feedback to the server)
        ranker.federated_averaging_weights(feedback)
        # 保存全局模型权重
        global_weights_list.append(copy.deepcopy(ranker.get_current_weights()))

        # the server send the newly trained model to every client
        for client in clients:
            client.update_model(ranker)

    if unlearn_method == 'fineTuning':
        print("fine tuning start!")
        fine_tune_iterations = 400
        ft_num_update = None
        # 筛选出非恶意客户端
        # non_malicious_clients = [client for client in clients if not client.is_malicious]
        non_malicious_clients = [client for client in clients[7:]]
        # 使用非恶意客户端的数据进行微调
        for ft_iter in range(fine_tune_iterations):
            feedback = []
            online_ndcg = []
            online_mrr = []
            for client in non_malicious_clients:
                client_message, client_metric= client.client_ranker_update(params["interactions_per_feedback"])
                feedback.append(client_message)
                # online evaluation
                online_ndcg.append(client_metric.mean_ndcg)
                online_mrr.append(client_metric.mean_mrr)
        
            # online-line metrics
            ndcg_clients_un.append(np.mean(online_ndcg))
            mrr_clients_un.append(np.mean(online_mrr))
        
            if ft_num_update is not None:
                if ft_iter % int((fine_tune_iterations/ft_num_update))== 0:
                    all_result = ranker.get_all_query_result_list(testdata)
                    ndcg = average_ndcg_at_k(testdata, all_result, 10)
                    mrr = average_mrr_at_k(testdata, all_result, 10)
                    ndcg_server_un.append(ndcg)
                    mrr_server_un.append(mrr)
            else:
                all_result = ranker.get_all_query_result_list(testdata)
                ndcg = average_ndcg_at_k(testdata, all_result, 10)
                mrr = average_mrr_at_k(testdata, all_result, 10)
                ndcg_server_un.append(ndcg)
                mrr_server_un.append(mrr)
        
            # 使用来自非恶意客户端的反馈更新全局模型
            ranker.federated_averaging_weights(feedback)
            for client in non_malicious_clients:
                client.update_model(ranker)

    elif unlearn_method == 'fedEraser':
        forget_client_idx = [0,1,2,3,4,5,6]
        unlearned_global_weights_list = unlearning(global_weights_list, client_weights_list, clients, interactions_per_feedback, forget_client_idx)
        for i in range(len(unlearned_global_weights_list)):
            ranker.update_to_gradients(unlearned_global_weights_list[i])
            # off-line metrics
            if num_update is not None:
                if i % int((n_iterations/num_update))== 0:
                    all_result = ranker.get_all_query_result_list(testdata)
                    ndcg = average_ndcg_at_k(testdata, all_result, 10)
                    mrr = average_mrr_at_k(testdata, all_result, 10)
                    ndcg_server_un.append(ndcg)
                    mrr_server_un.append(mrr)

            else:
                all_result = ranker.get_all_query_result_list(testdata)
                ndcg = average_ndcg_at_k(testdata, all_result, 10)
                mrr = average_mrr_at_k(testdata, all_result, 10)
                ndcg_server_un.append(ndcg)
                mrr_server_un.append(mrr)

    elif unlearn_method == 'pga':
        print("Gradient Ascent Unlearning Start!")
        forget_client_indices = [0,1,2,3,4,5,6]
        unlearn_iterations = 1000
        for i in range(unlearn_iterations):
            i += 1
            feedback = []
            online_ndcg_un = []
            online_mrr_un = []
            for idx in range(len(clients)):
                if idx in forget_client_indices:
                    client_message, client_metric = clients[idx].client_ranker_unlearning(interactions_per_feedback,
                                                                                        multi_update)
                else:
                    client_message, client_metric = clients[idx].client_ranker_update(interactions_per_feedback,
                                                                                    multi_update)

                feedback.append(client_message)
                # online evaluation
                online_ndcg_un.append(client_metric.mean_ndcg)
                online_mrr_un.append(client_metric.mean_mrr)
            # online-line metrics
            ndcg_clients_un.append(np.mean(online_ndcg_un))
            mrr_clients_un.append(np.mean(online_mrr_un))

            # off-line metrics
            if num_update is not None:
                if i % int((n_iterations / num_update)) == 0:
                    all_result = ranker.get_all_query_result_list(testdata)
                    ndcg = average_ndcg_at_k(testdata, all_result, 10)
                    mrr = average_mrr_at_k(testdata, all_result, 10)
                    ndcg_server_un.append(ndcg)
                    mrr_server_un.append(mrr)

            else:
                all_result = ranker.get_all_query_result_list(testdata)
                ndcg = average_ndcg_at_k(testdata, all_result, 10)
                mrr = average_mrr_at_k(testdata, all_result, 10)
                ndcg_server_un.append(ndcg)
                mrr_server_un.append(mrr)

            ranker.federated_averaging_weights(feedback)
            # the server send the newly trained model to every client
            for client in clients:
                client.update_model(ranker)
    else:
        print("Retrain start!")
        non_malicious_clients = [client for client in clients[7:]]
        for i in tqdm(range(n_iterations), desc=message):
            i += 1
            feedback = []
            online_ndcg_un = []
            online_mrr_un = []
            for client in non_malicious_clients:
                client_message, client_metric = client.client_ranker_update(interactions_per_feedback, multi_update)
                feedback.append(client_message)
                # online evaluation
                online_ndcg_un.append(client_metric.mean_ndcg)
                online_mrr_un.append(client_metric.mean_mrr)

            # online-line metrics
            ndcg_clients_un.append(np.mean(online_ndcg_pre))
            mrr_clients_un.append(np.mean(online_mrr_pre))

            # off-line metrics
            if num_update is not None:
                if i % int((n_iterations/num_update))== 0:
                    all_result = ranker.get_all_query_result_list(testdata)
                    ndcg = average_ndcg_at_k(testdata, all_result, 10)
                    mrr = average_mrr_at_k(testdata, all_result, 10)
                    ndcg_server_un.append(ndcg)
                    mrr_server_un.append(mrr)

            else:
                all_result = ranker.get_all_query_result_list(testdata)
                ndcg = average_ndcg_at_k(testdata, all_result, 10)
                mrr = average_mrr_at_k(testdata, all_result, 10)
                ndcg_server_un.append(ndcg)
                mrr_server_un.append(mrr)

            # train the server ranker (clients send feedback to the server)
            ranker.federated_averaging_weights(feedback)

            # the server send the newly trained model to every client
            for client in clients:
                client.update_model(ranker)

    return TrainResult(ranker = ranker, ndcg_server_pre = ndcg_server_pre, ndcg_server_un=ndcg_server_un,
                       ndcg_clients_pre=ndcg_clients_pre, ndcg_clients_un= ndcg_clients_un,
                       mrr_server_pre = mrr_server_pre, mrr_server_un = mrr_server_un,
                       mrr_clients_pre = mrr_clients_pre, mrr_clients_un = mrr_clients_un)


def train_uniform_model_poison(params: Dict[str, Any], traindata: LetorDataset, testdata: LetorDataset, message, num_update=None) -> TrainResult:
    """

    :param params:
    :param traindata: dataset used for training server ranker
    :param testdata: dataset used for testing true performance of server ranker - using true relevance label
    :param message:
    :return:
    """
    seed = params["seed"]
    np.random.seed(seed)

    n_clients = params["n_clients"]
    interactions_per_feedback = params["interactions_per_feedback"]
    click_model = params["click_model"]
    ranker = params["ranker_generator"]
    multi_update = params["multi_update"]
    sensitivity = params["sensitivity"]
    epsilon = params["epsilon"]
    enable_noise = params["enable_noise"]
    unlearn_method = params['unlearn_method']

    clients = [RankingClient(traindata, ranker, seed * n_clients + client_id, click_model, sensitivity, epsilon, enable_noise, n_clients, is_malicious=(client_id < 1)) for client_id in range(n_clients)]
    n_iterations = params["interactions_budget"] // n_clients // interactions_per_feedback # total iteration times (training times) for federated training

    ndcg_server_pre = [] # off-line metric (on testset)
    mrr_server_pre = [] # off-line metric (on testset)                                                                                                                                       
    ndcg_clients_pre = [] # averaged online metric
    mrr_clients_pre = [] # averaged online metric

    ndcg_server_un = []
    mrr_server_un = []
    ndcg_clients_un = []
    mrr_clients_un = []

    global_weights_list = []  # 用于保存全局模型权重的列表
    client_weights_list = [[] for _ in range(n_clients)]  # 用于保存每个客户端模型权重的列表

    # initialize gradient
    gradients = np.zeros(traindata._feature_size)
    global_weights_list.append(copy.deepcopy(ranker.get_current_weights()))
    for i in tqdm(range(n_iterations), desc=message):
        i += 1
        feedback = []
        online_ndcg_pre = []
        online_mrr_pre = []
        for client in clients:
            client_message, client_metric = client.client_ranker_update(interactions_per_feedback, multi_update)
            feedback.append(client_message)
            # online evaluation
            online_ndcg_pre.append(client_metric.mean_ndcg)
            online_mrr_pre.append(client_metric.mean_mrr)

        # online-line metrics
        ndcg_clients_pre.append(np.mean(online_ndcg_pre))
        mrr_clients_pre.append(np.mean(online_mrr_pre))

        # off-line metrics
        if num_update is not None:
            if i % int((n_iterations/num_update))== 0:
                all_result = ranker.get_all_query_result_list(testdata)
                ndcg = average_ndcg_at_k(testdata, all_result, 10)
                mrr = average_mrr_at_k(testdata, all_result, 10)
                ndcg_server_pre.append(ndcg)
                mrr_server_pre.append(mrr)

        else:
            all_result = ranker.get_all_query_result_list(testdata)
            ndcg = average_ndcg_at_k(testdata, all_result, 10)
            mrr = average_mrr_at_k(testdata, all_result, 10)
            ndcg_server_pre.append(ndcg)
            mrr_server_pre.append(mrr)

        for client_id in range(len(clients)):
            client_weights_list[client_id].append(copy.deepcopy(feedback[client_id][1]))

        # train the server ranker (clients send feedback to the server)
        ranker.federated_averaging_weights(feedback)
        # 保存全局模型权重
        global_weights_list.append(copy.deepcopy(ranker.get_current_weights()))

        # the server send the newly trained model to every client
        for client in clients:
            client.update_model(ranker)

    if unlearn_method == 'fineTuning':
        print("fine tuning start!")
        fine_tune_iterations = 400
        ft_num_update = None
        # 筛选出非恶意客户端
        # non_malicious_clients = [client for client in clients if not client.is_malicious]
        non_malicious_clients = [client for client in clients[5:]]
        # 使用非恶意客户端的数据进行微调
        for ft_iter in range(fine_tune_iterations):
            feedback = []
            online_ndcg = []
            online_mrr = []
            for client in non_malicious_clients:
                client_message, client_metric= client.client_ranker_update(params["interactions_per_feedback"])
                feedback.append(client_message)
                # online evaluation
                online_ndcg.append(client_metric.mean_ndcg)
                online_mrr.append(client_metric.mean_mrr)
        
            # online-line metrics
            ndcg_clients_un.append(np.mean(online_ndcg))
            mrr_clients_un.append(np.mean(online_mrr))
        
            if ft_num_update is not None:
                if ft_iter % int((fine_tune_iterations/ft_num_update))== 0:
                    all_result = ranker.get_all_query_result_list(testdata)
                    ndcg = average_ndcg_at_k(testdata, all_result, 10)
                    mrr = average_mrr_at_k(testdata, all_result, 10)
                    ndcg_server_un.append(ndcg)
                    mrr_server_un.append(mrr)
            else:
                all_result = ranker.get_all_query_result_list(testdata)
                ndcg = average_ndcg_at_k(testdata, all_result, 10)
                mrr = average_mrr_at_k(testdata, all_result, 10)
                ndcg_server_un.append(ndcg)
                mrr_server_un.append(mrr)
        
            # 使用来自非恶意客户端的反馈更新全局模型
            ranker.federated_averaging_weights(feedback)
            for client in non_malicious_clients:
                client.update_model(ranker)
    
    elif unlearn_method == 'fedEraser':
        forget_client_idx = [0,1,2]
        unlearned_global_weights_list = unlearning(global_weights_list, client_weights_list, clients, interactions_per_feedback, forget_client_idx)
        for i in range(len(unlearned_global_weights_list)):
            ranker.update_to_gradients(unlearned_global_weights_list[i])
            # off-line metrics
            if num_update is not None:
                if i % int((n_iterations/num_update))== 0:
                    all_result = ranker.get_all_query_result_list(testdata)
                    ndcg = average_ndcg_at_k(testdata, all_result, 10)
                    mrr = average_mrr_at_k(testdata, all_result, 10)
                    ndcg_server_un.append(ndcg)
                    mrr_server_un.append(mrr)

            else:
                all_result = ranker.get_all_query_result_list(testdata)
                ndcg = average_ndcg_at_k(testdata, all_result, 10)
                mrr = average_mrr_at_k(testdata, all_result, 10)
                ndcg_server_un.append(ndcg)
                mrr_server_un.append(mrr)

    elif unlearn_method == 'pga':
        print("Gradient Ascent Unlearning Start!")
        forget_client_indices = [0, 1, 2]
        unlearn_iterations = 1000
        for i in range(unlearn_iterations):
            i += 1
            feedback = []
            online_ndcg_un = []
            online_mrr_un = []
            for idx in range(len(clients)):
                if idx in forget_client_indices:
                    client_message, client_metric = clients[idx].client_ranker_unlearning(interactions_per_feedback,
                                                                                        multi_update)
                else:
                    client_message, client_metric = clients[idx].client_ranker_update(interactions_per_feedback,
                                                                                    multi_update)

                feedback.append(client_message)
                # online evaluation
                online_ndcg_un.append(client_metric.mean_ndcg)
                online_mrr_un.append(client_metric.mean_mrr)
            # online-line metrics
            ndcg_clients_un.append(np.mean(online_ndcg_un))
            mrr_clients_un.append(np.mean(online_mrr_un))

            # off-line metrics
            if num_update is not None:
                if i % int((n_iterations / num_update)) == 0:
                    all_result = ranker.get_all_query_result_list(testdata)
                    ndcg = average_ndcg_at_k(testdata, all_result, 10)
                    mrr = average_mrr_at_k(testdata, all_result, 10)
                    ndcg_server_un.append(ndcg)
                    mrr_server_un.append(mrr)

            else:
                all_result = ranker.get_all_query_result_list(testdata)
                ndcg = average_ndcg_at_k(testdata, all_result, 10)
                mrr = average_mrr_at_k(testdata, all_result, 10)
                ndcg_server_un.append(ndcg)
                mrr_server_un.append(mrr)

            ranker.federated_averaging_weights(feedback)
            # the server send the newly trained model to every client
            for client in clients:
                client.update_model(ranker)

    else:
        print("Retrain start!")
        non_malicious_clients = [client for client in clients[1:]]
        for i in tqdm(range(n_iterations), desc=message):
            i += 1
            feedback = []
            online_ndcg_un = []
            online_mrr_un = []
            for client in non_malicious_clients:
                client_message, client_metric = client.client_ranker_update(interactions_per_feedback, multi_update)
                feedback.append(client_message)
                # online evaluation
                online_ndcg_un.append(client_metric.mean_ndcg)
                online_mrr_un.append(client_metric.mean_mrr)

            # online-line metrics
            ndcg_clients_un.append(np.mean(online_ndcg_pre))
            mrr_clients_un.append(np.mean(online_mrr_pre))

            # off-line metrics
            if num_update is not None:
                if i % int((n_iterations/num_update))== 0:
                    all_result = ranker.get_all_query_result_list(testdata)
                    ndcg = average_ndcg_at_k(testdata, all_result, 10)
                    mrr = average_mrr_at_k(testdata, all_result, 10)
                    ndcg_server_un.append(ndcg)
                    mrr_server_un.append(mrr)

            else:
                all_result = ranker.get_all_query_result_list(testdata)
                ndcg = average_ndcg_at_k(testdata, all_result, 10)
                mrr = average_mrr_at_k(testdata, all_result, 10)
                ndcg_server_un.append(ndcg)
                mrr_server_un.append(mrr)

            # train the server ranker (clients send feedback to the server)
            ranker.federated_averaging_weights(feedback)

            # the server send the newly trained model to every client
            for client in clients:
                client.update_model(ranker)

    return TrainResult(ranker = ranker, ndcg_server_pre = ndcg_server_pre, ndcg_server_un=ndcg_server_un,
                       ndcg_clients_pre=ndcg_clients_pre, ndcg_clients_un= ndcg_clients_un,
                       mrr_server_pre = mrr_server_pre, mrr_server_un = mrr_server_un,
                       mrr_clients_pre = mrr_clients_pre, mrr_clients_un = mrr_clients_un)

# generate metrics: ndcg@k & mrr@10
def average_ndcg_at_k(dataset, query_result_list, k):
    ndcg = 0.0
    num_query = 0
    for query in dataset.get_all_querys():
        if len(dataset.get_relevance_docids_by_query(query)) == 0:  # for this query, ranking list is None
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
        if len(dataset.get_relevance_docids_by_query(query)) == 0: # for this query, ranking list is None
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
