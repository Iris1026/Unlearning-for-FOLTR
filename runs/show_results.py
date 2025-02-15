import pickle
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))  
sys.path.append(project_root)

def print_training_results(file_path, print_items=None):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    if hasattr(data, '_asdict'):
        data = data._asdict()

    if print_items is None:
        print("可用的数据项：")
        for key in data.keys():
            print(f"- {key}")
        return
    
    for item in print_items:
        if item in data:
            print(f"\n{item}:")
            print(data[item])
        else:
            print(f"\n警告: 未找到 {item}")

if __name__ == "__main__":
    file_path = "/mnt/data/taoyiling/Unlearning-for-FOLTR_20250206/save/MQ2007/1/clean/Perfect_training_state_1000_RelR.pkl"
    print_items = ['RelR_Diff'] 
    print_training_results(file_path,print_items)