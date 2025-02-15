import os
import sys
import pickle
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..")) 
sys.path.append(project_root)
from client.federated_optimize import *

def parse_args():
    parser = argparse.ArgumentParser(description='Print final online NDCG results')
    parser.add_argument('--base_dir', type=str, default='../save',
                       help='Base directory containing results')
    parser.add_argument('--dataset', type=str, default='MQ2007',
                       choices=['MQ2007', 'MSLR10K', 'istella-s', 'Yahoo'],
                       help='Dataset to evaluate')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    return parser.parse_args()

def print_final_online_ndcg(base_dir: str, dataset: str, save_dir: str):

    scenarios = ['clean', 'data_poison', 'model_poison']
    models = ['Perfect', 'Navigational', 'Informational']
    methods = ['Base'] + ['original','retrain', 'fedEraser', 'fineTuning']

    scenario_dirs = {
        'clean': 'clean',
        'data_poison': 'data',
        'model_poison': 'model'
    }

    results = {}
    for scenario in scenarios:
        scenario_dir = scenario_dirs[scenario]
        results[scenario] = {}
        for model in models:
            results[scenario][model] = {}
            base_path = f"{base_dir}/{dataset}/1/{scenario_dir}/{model}_training_state_1000.pkl"
            try:
                with open(base_path, 'rb') as f:
                    base_result = pickle.load(f)
                    if not base_result.online_ndcg_performance_list:
                        print(f"Warning: Empty performance list in {base_path}")
                        results[scenario][model]['Base'] = None
                    else:
                        final_base_ndcg = base_result.online_ndcg_performance_list[-1]
                        results[scenario][model]['Base'] = final_base_ndcg
            except FileNotFoundError:
                print(f"Warning: Base file not found - {base_path}")
                results[scenario][model]['Base'] = None

            for method in methods[1:]: 
                result_path = f"{base_dir}/{dataset}/1/{scenario_dir}/{model}_unlearning_{method}_1000.pkl"
                try:
                    with open(result_path, 'rb') as f:
                        result = pickle.load(f)
                        if not result.online_ndcg_performance_list:
                            print(f"Warning: Empty performance list in {result_path}")
                            results[scenario][model][method] = None
                        else:
                            final_ndcg = result.online_ndcg_performance_list[-1]
                            results[scenario][model][method] = final_ndcg + results[scenario][model]['Base']
                except FileNotFoundError:
                    print(f"Warning: File not found - {result_path}")
                    results[scenario][model][method] = None

    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{dataset}_final_online_ndcg.txt"

    with open(save_path, 'w') as f:
        header = f"Final Online NDCG Results for {dataset}\n"
        f.write(header)
        f.write("=" * len(header) + "\n\n")
        print(header)
        print("=" * len(header))

        for scenario in scenarios:
            scenario_header = f"\n{scenario.upper()}\n"
            f.write(scenario_header)
            f.write("-" * len(scenario_header) + "\n")
            print(scenario_header)
            print("-" * len(scenario_header))

            table_header = f"{'Model':<12} | " + " | ".join(f"{method:>10}" for method in methods[1:])
            f.write(table_header + "\n")
            f.write("-" * len(table_header) + "\n")
            print(table_header)
            print("-" * len(table_header))

            for model in models:
                row = f"{model:<12} | "
                for method in methods[1:]:
                    value = results[scenario][model][method]
                    if value is not None:
                        row += f"{value:>10.2f} | "
                    else:
                        row += f"{'N/A':>10} | "
                f.write(row.rstrip(" |") + "\n")
                print(row.rstrip(" |"))

            f.write("-" * len(table_header) + "\n")
            print("-" * len(table_header))

    print(f"\nResults saved to {save_path}")
    return results

if name == "__main__":
    args = parse_args()
    print_final_online_ndcg(args.base_dir, args.dataset, args.save_dir)