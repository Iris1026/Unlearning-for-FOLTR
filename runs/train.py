import sys
import os
import pickle
import argparse
sys.path.append('../')
from data.LetorDataset import LetorDataset
from clickModel.click_simulate import CcmClickModel
from ranker.PDGDLinearRanker import PDGDLinearRanker
from client.federated_optimize import train_federated

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Training Parameters')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='MQ2007',
                        choices=['MQ2007', 'MSLR10K', 'istella-s', 'Yahoo'],
                        help='Dataset name')
    
    # Training parameters
    parser.add_argument('--n_clients', type=int, default=10,
                        help='Number of clients')
    parser.add_argument('--interactions_per_feedback', type=int, default=5,
                        help='Batch size')
    parser.add_argument('--interactions_budget', type=int, default=50000,
                        help='Total interactions budget')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--update', type=bool, default=True,
                        help='Enable client multi update')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--enable_relr', type=bool, default=False,
                        help='Enable Relevancy Reset (RelR) evaluation')
    
    # Training scenario
    parser.add_argument('--scenario', type=str, default='clean',
                        choices=['clean', 'data_poison', 'model_poison'],
                        help='Training scenario type')
    
    # Poisoning parameters
    parser.add_argument('--n_malicious', type=int, default=3,
                        help='Number of malicious clients')
    
    # Path parameters
    parser.add_argument('--dataset_root_dir', type=str, default='../datasets',
                        help='Root directory for datasets')
    parser.add_argument('--save_dir', type=str, default='../save1',
                        help='Directory to save training results')
    
    return parser.parse_args()

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
                                         name="Informational", depth=10),
            "Poison": CcmClickModel(click_relevance={0: 1.0, 1: 0.5, 2: 0.0},
                                  stop_relevance={0: 0.0, 1: 0.0, 2: 0.0},
                                  name="Poison", depth=10)
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
                                         name="Informational", depth=10),
            "Poison": CcmClickModel(click_relevance={0: 1.0, 1: 0.8, 2: 0.6, 3:0.2, 4:0.0},
                                  stop_relevance={0: 0.0, 1: 0.0, 2: 0.0, 3:0.0, 4:0.0},
                                  name="Poison", depth=10)
        }
    return models.get(model_name)

def get_dataset_params(dataset):
    params = {
        "MQ2007": {"n_folds": 1, "n_features": 46, "data_norm": False},
        "MSLR10K": {"n_folds": 5, "n_features": 136, "data_norm": True},
        "istella-s": {"n_folds": 1, "n_features": 220, "data_norm": True},
        "Yahoo": {"n_folds": 1, "n_features": 700, "data_norm": True}
    }
    return params.get(dataset)

def get_save_path(args, dataset_params, model_name, fold_id):
    scenario_dir = {
        'clean': 'clean',
        'data_poison': 'data',
        'model_poison': 'model'
    }[args.scenario]
    
    save_dir = f"{args.save_dir}/{args.dataset}/{fold_id + 1}/{scenario_dir}"
    os.makedirs(save_dir, exist_ok=True)
    
    n_iterations = args.interactions_budget // args.n_clients // args.interactions_per_feedback
    if args.enable_relr == True:
        return f"{save_dir}/{model_name}_training_state_{n_iterations}_RelR.pkl"
    else:
        return f"{save_dir}/{model_name}_training_state_{n_iterations}.pkl"

def run_training(args, dataset_params, click_model, model_name, fold_id):
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

    train_params = {
        "n_clients": args.n_clients,
        "interactions_budget": args.interactions_budget,
        "seed": args.seed,
        "interactions_per_feedback": args.interactions_per_feedback,
        "multi_update": args.update,
        "n_features": dataset_params["n_features"],
        "dataset": args.dataset,
        "scenario": args.scenario,
        "n_malicious": args.n_malicious,
        "fold_id": fold_id,
        "ranker_generator": ranker,
        "click_model": click_model
    }

    training_result = train_federated(
        args.scenario,
        train_params,
        trainset,
        testset,
        f"Training {args.scenario} - {model_name} - Fold {fold_id + 1}",
        args.enable_relr
    )

    save_path = get_save_path(args, dataset_params, model_name, fold_id)
    with open(save_path, 'wb') as f:
        pickle.dump(training_result, f)

    print(f"Saved training result to {save_path}")

if __name__ == "__main__":
    args = parse_args()
    dataset_params = get_dataset_params(args.dataset)
    
    if args.scenario == "data_poison":
        base_models = ["Perfect", "Navigational", "Informational"]
        poison_model = get_click_model(args.dataset, "Poison")
        click_models = [(model_name, (poison_model, get_click_model(args.dataset, model_name))) 
                       for model_name in base_models]
    else:
        click_models = [(name, get_click_model(args.dataset, name)) 
                       for name in ["Perfect", "Navigational", "Informational"]]

    for model_name, click_model in click_models:
        for fold_id in range(dataset_params["n_folds"]):
            print(f"\nTraining {args.scenario} scenario - {model_name} - Fold {fold_id + 1}")
            run_training(args, dataset_params, click_model, model_name, fold_id)
