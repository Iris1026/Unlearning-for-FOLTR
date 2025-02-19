import os
import sys
import seaborn as sns
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
from client.federated_optimize import *

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate and plot final NDCG results')
    parser.add_argument('--base_dir', type=str, default='../save',
                        help='Base directory containing results')
    parser.add_argument('--dataset', type=str, default='MQ2007',
                        choices=['MQ2007', 'MSLR10K', 'istella-s', 'Yahoo'],
                        help='Dataset to evaluate')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    return parser.parse_args()

def load_results(base_dir: str, dataset: str, scenario: str, model_name: str, fold: int) -> Dict:
    methods = ['base', 'original', 'retrain', 'fedEraser', 'fineTuning', 'pga', 'FedRemove']
    results = {}

    scenario_map = {
        'Clean': 'clean',
        'Data Poison': 'data',
        'Model Poison': 'model'
    }
    scenario_dir = scenario_map[scenario]

    base_training_path = f"{base_dir}/{dataset}/{fold}/{scenario_dir}/{model_name}_training_state_1000.pkl"
    try:
        with open(base_training_path, 'rb') as f:
            base_result = pickle.load(f)
            results['base'] = base_result.ndcg_server[:1000]  # 0-1000 epoch, if you change the epoch, you'll need to modify this section.
    except FileNotFoundError:
        print(f"Warning: Base training file not found - {base_training_path}")
        return {}

    for method in methods[1:]:  
        unlearn_path = f"{base_dir}/{dataset}/{fold}/{scenario_dir}/{model_name}_unlearning_{method}_1000.pkl"
        try:
            with open(unlearn_path, 'rb') as f:
                unlearn_result = pickle.load(f)
                results[method] = unlearn_result.ndcg_server[:1000] # 0-1000 epoch, if you change the epoch, you'll need to modify this section.
        except FileNotFoundError:
            print(f"Warning: Unlearning file not found - {unlearn_path}")
            results[method] = None  

    return results
    
def evaluate_all_scenarios(base_dir: str, dataset: str, save_dir: str):
    scenarios = ['Clean', 'Data Poison', 'Model Poison']
    models = ['Perfect', 'Navigational', 'Informational']
    
    sns.set(style="ticks")
    
    methods_order = ['original', 'retrain', 'fedEraser', 'fineTuning', 'pga', 'FedRemove']
    
    colors = {
        'base': 'b',
        'original': 'b',
        'retrain': 'g',
        'fedEraser': 'r',
        'fineTuning': 'c',
        'pga': 'm',
        'FedRemove': 'y'
    }
    
    linestyles = {
        'base': '-',
        'original': '-',
        'retrain': (0, (3, 1, 1, 1)),
        'fedEraser': (0, (5, 10)),
        'fineTuning': (0, (1, 5)),
        'pga': (0, (3, 3)),
        'FedRemove': (0, (5, 1))
    }
    
    for i, scenario in enumerate(scenarios):
        fig, axes = plt.subplots(1, len(models), figsize=(35, 6))
        legend_handles = {method: None for method in methods_order}
        
        for j, model in enumerate(models):
            results = load_results(base_dir, dataset, scenario, model, fold=1)
            ax = axes[j]
            ax.tick_params(axis='both', which='major', labelsize=20)
            
            if 'base' in results and results['base'] is not None:
                epochs = np.arange(0, 1000)
                ax.plot(epochs, results['base'], color=colors['base'], 
                       linestyle=linestyles['base'], linewidth=1.5)
            
            for method in methods_order:
                if method not in results or results[method] is None:
                    continue
                
                epochs = np.arange(1000, 2000)
                line, = ax.plot(epochs, results[method], 
                              color=colors[method], 
                              linestyle=linestyles[method], 
                              linewidth=1.5)
                
                if legend_handles[method] is None:
                    legend_handles[method] = line
            
            ax.set_title(f'{model}', fontsize=30, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=25, fontweight='bold')
            ax.set_ylabel(f'Offline nDCG@10', fontsize=30, fontweight='bold')
            ax.grid(True)
            
            ax.axvline(x=1000, color='gray', linestyle='--', alpha=0.5)
            
            min_y, max_y = ax.get_ylim()
            ax.text(500, min_y * 1.05, 'Training', 
                   horizontalalignment='center', fontsize=20, fontweight='bold')
            ax.text(1500, min_y * 1.05, 'Unlearning', 
                   horizontalalignment='center', fontsize=20, fontweight='bold')
        
        valid_handles = []
        valid_labels = []
        for method in methods_order:
            if legend_handles[method] is not None:
                valid_handles.append(legend_handles[method])
                valid_labels.append(method)
        
        fig.legend(handles=valid_handles,
                  labels=valid_labels,
                  loc='upper center', 
                  bbox_to_anchor=(0.5, 1.15), 
                  fontsize=30, 
                  ncol=6, 
                  frameon=False)
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{dataset}_{scenario.replace(' ', '_')}_offline.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved evaluation plot to {save_path}")


if __name__  == "__main__":
    args = parse_args()
    print(f"Evaluating {args.dataset}...")
    evaluate_all_scenarios(args.base_dir, args.dataset, args.save_dir)
    print(f"Evaluation complete! Results saved in {args.save_dir}")
