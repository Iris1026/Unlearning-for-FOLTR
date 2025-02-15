import sys
import os
import pickle
import argparse
sys.path.append('../')
from data.LetorDataset import LetorDataset
from clickModel.click_simulate import CcmClickModel
from ranker.PDGDLinearRanker import PDGDLinearRanker
from client.federated_optimize import apply_unlearning

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Unlearning Parameters')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='istella-s',
                        choices=['MQ2007', 'MSLR10K', 'istella-s', 'Yahoo'],
                        help='Dataset name')
    
    # Training scenario from which to unlearn
    parser.add_argument('--scenario', type=str, default='clean',
                        choices=['clean', 'data_poison', 'model_poison'],
                        help='Original training scenario type')
     
    # Unlearning parameters
    parser.add_argument('--unlearn_method', type=str, default='fineTuning',
                        choices=['retrain', 'fedEraser', 'fineTuning', 'pga', 'FedRemove','original'],
                        help='Method for unlearning')
    parser.add_argument('--n_malicious', type=int, default=3,
                        help='Number of clients to unlearn')
    
    # Original training parameters (needed for proper unlearning)
    parser.add_argument('--n_clients', type=int, default=10,
                        help='Number of clients from original training')
    parser.add_argument('--interactions_per_feedback', type=int, default=5,
                        help='Batch size from original training')
    parser.add_argument('--interactions_budget', type=int, default=50000,
                        help='Total interactions budget from original training')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--update', type=bool, default=True,
                        help='Enable client multi update')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--enable_relr', type=bool, default=False,
                        help='Enable Relevancy Reset (RelR) evaluation')

    # Path parameters
    parser.add_argument('--dataset_root_dir', type=str, default='../datasets',
                        help='Root directory for datasets')
    parser.add_argument('--save_dir', type=str, default='../save1',
                        help='Directory with saved training results')
    
    return parser.parse_args()

def get_dataset_params(dataset):
    params = {
        "MQ2007": {"n_folds": 1, "n_features": 46, "data_norm": False},
        "MSLR10K": {"n_folds": 1, "n_features": 136, "data_norm": True},
        "istella-s": {"n_folds": 1, "n_features": 220, "data_norm": True},
        "Yahoo": {"n_folds": 1, "n_features": 700, "data_norm": True}
    }
    return params.get(dataset)

def get_click_model(dataset, model_name):
    if dataset == "MQ2007":
        models = {
            "Perfect": CcmClickModel(click_relevance={0: 0.0, 1: 0.5, 2: 1.0},
                                   stop_relevance={0: 0.0, 1: 0.0, 2: 0.0},
                                   name="Perfect", depth=10),
            "Navigational": CcmClickModel(click_relevance={0: 0.05, 1: 0.5, 2: 0.95},
                                        stop_relevance={0: 0.2, 1: 0.5, 2: 0.9},
                                        name="Navigational", depth=10),
            "Informational": CcmClickModel(click_relevance={0: 0.4, 1: 0.7, 2: 0.9},
                                         stop_relevance={0: 0.1, 1: 0.3, 2: 0.5},
                                         name="Informational", depth=10)
        }
    else:  # MSLR10K, istella-s, Yahoo
        models = {
            "Perfect": CcmClickModel(click_relevance={0: 0.0, 1: 0.2, 2: 0.4, 3: 0.8, 4: 1.0},
                                   stop_relevance={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
                                   name="Perfect", depth=10),
            "Navigational": CcmClickModel(click_relevance={0: 0.05, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.95},
                                        stop_relevance={0: 0.2, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.9},
                                        name="Navigational", depth=10),
            "Informational": CcmClickModel(click_relevance={0: 0.4, 1: 0.6, 2: 0.7, 3: 0.8, 4: 0.9},
                                         stop_relevance={0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5},
                                         name="Informational", depth=10)
        }
    return models.get(model_name)

def get_training_path(args, dataset_params, model_name, fold_id):
    scenario_dir = {
        'clean': 'clean',
        'data_poison': 'data',
        'model_poison': 'model'
    }[args.scenario]
    
    n_iterations = args.interactions_budget // args.n_clients // args.interactions_per_feedback
    if args.enable_relr == True:
        return f"{args.save_dir}/{args.dataset}/{fold_id + 1}/{scenario_dir}/{model_name}_training_state_{n_iterations}_RelR.pkl"
    else:
        return f"{args.save_dir}/{args.dataset}/{fold_id + 1}/{scenario_dir}/{model_name}_training_state_{n_iterations}.pkl"

def get_unlearning_path(args, dataset_params, model_name, fold_id):
    scenario_dir = {
        'clean': 'clean',
        'data_poison': 'data',
        'model_poison': 'model'
    }[args.scenario]
    
    n_iterations = args.interactions_budget // args.n_clients // args.interactions_per_feedback
    base_path = f"{args.save_dir}/{args.dataset}/{fold_id + 1}/{scenario_dir}/{model_name}"
    if args.enable_relr == True:
        return f"{base_path}_unlearning_{args.unlearn_method}_{n_iterations}_RelR.pkl"
    else:
        return f"{base_path}_unlearning_{args.unlearn_method}_{n_iterations}.pkl"

def run_unlearning(args, dataset_params, model_name, fold_id):
    cache_root = "../datasets/cache"
    os.makedirs(cache_root, exist_ok=True)  
    training_path = get_training_path(args, dataset_params, model_name, fold_id)
    try:
        with open(training_path, 'rb') as f:
            training_result = pickle.load(f)
    except FileNotFoundError:
        print(f"Training result not found at {training_path}")
        return

    trainset = LetorDataset(
        f"{args.dataset_root_dir}/{args.dataset}/Fold{fold_id + 1}/train.txt",
        dataset_params["n_features"], 
        query_level_norm=dataset_params["data_norm"],
        cache_root="../datasets/cache"
    )
    testset = LetorDataset(
        f"{args.dataset_root_dir}/{args.dataset}/Fold{fold_id + 1}/test.txt",
        dataset_params["n_features"], 
        query_level_norm=dataset_params["data_norm"],
        cache_root="../datasets/cache"
    )

    ranker = PDGDLinearRanker(dataset_params["n_features"], args.learning_rate)

    unlearn_params = {
        "scenario": args.scenario,
        "n_clients": args.n_clients,
        "n_malicious": args.n_malicious,
        "interactions_budget": args.interactions_budget,
        "seed": args.seed,
        "interactions_per_feedback": args.interactions_per_feedback,
        "multi_update": args.update,
        "n_features": dataset_params["n_features"],
        "dataset": args.dataset,
        "fold_id": fold_id,
        "ranker_generator": ranker,
        "click_model": get_click_model(args.dataset, model_name)
    }

    if args.scenario == "data_poison":
        unlearn_params["poison_model"] = get_click_model(args.dataset, "Poison")

    unlearning_result = apply_unlearning(
        args.unlearn_method,
        training_result,
        unlearn_params,
        trainset,
        testset,
        enable_relr=args.enable_relr
    )

    unlearning_path = get_unlearning_path(args, dataset_params, model_name, fold_id)
    os.makedirs(os.path.dirname(unlearning_path), exist_ok=True)
    with open(unlearning_path, 'wb') as f:
        pickle.dump(unlearning_result, f)

    print(f"Saved unlearning result to {unlearning_path}")

if __name__ == "__main__":
    args = parse_args()
    dataset_params = get_dataset_params(args.dataset)
    model_names = ["Perfect", "Navigational", "Informational"]
    
    for model_name in model_names:
        for fold_id in range(dataset_params["n_folds"]):
            print(f"\nUnlearning {args.scenario} scenario - {model_name} - Fold {fold_id + 1}")
            print(f"Method: {args.unlearn_method}")
            run_unlearning(args, dataset_params, model_name, fold_id)


