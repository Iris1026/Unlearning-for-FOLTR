# Unlearning for Federated Online Learning to Rank
## Download datasets
In the paper, we use four popular LTR datasets: MQ2007, MSLR-WEB10K, Yahoo! and Istella-S.
- MQ2007 can be downloaded from the Microsoft Research [website](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/). 
- MSLR-WEB10K can be downloaded from the Microsoft Research [website](https://www.microsoft.com/en-us/research/project/mslr/).  
- Yahoo! can be downloaded from [Yahoo Webscope program](https://webscope.sandbox.yahoo.com/catalog.php?datatype=c).
- Istella-S can be downloaded from [Instella-S website](https://istella.ai/datasets/letor-dataset/)

After downloading data files, they have to be unpacked within the `./datasets` folder.


## Environment Setups
- Create a new virtual environment:
``` bash
conda create --name environment_name python=3.8.13
```

- Install all the libraries for running the code:
``` bash
pip install -r requirements.txt
```

## Running Federated Learning Process
The experiment is divided into two parts: first, the Federated Learning phase needs to be conducted, followed by the implementation of Federated Unlearning.

### Basic Usage
Run the script with the default parameters:
```bash
cd runs
python train.py
```
This command trains the model using the `MQ2007` dataset with default settings, saving results to the `../save` directory.


### Complete Parameter Example

To customize the training, you can pass arguments to the script. Below is an example of running the script with all available parameters:
```bash
cd runs
python train.py \
  --dataset MQ2007 \
  --n_clients 10 \
  --interactions_per_feedback 5 \
  --interactions_budget 50000 \
  --learning_rate 0.1 \
  --update True \
  --num_update 50 \
  --seed 1 \
  --scenario CLEAN \
  --n_malicious 3 \
  --dataset_root_dir ../datasets \
  --save_dir ../save
```

### Click Models

Click models simulate user behavior based on relevance scores. The codebase supports the following click models:

- `Perfect`
- `Navigational`
- `Informational`
- `Poison` (only used in `data_poison` scenarios)

Click model configurations vary based on the dataset. Refer to the `get_click_model` function in the code for details.

### Training Output

After training, results will be saved in a `.pkl` file in the specified `save_dir`.

Example file path:
```
../save/MQ2007/1/clean/Perfect_training_state_2000.pkl
```
The structure includes the dataset name, fold ID, scenario type, and click model.


## Running Federated Unlearning Process

### Basic Usage
Run the unlearning script with the required method:
```bash
cd runs
python unlearn.py --unlearn_method retrain
```

### Complete Unlearning Example

To customize the unlearning process, you can pass arguments to the script. Below is an example of running the script with all available parameters:
```bash
cd runs
python unlearn.py \
  --dataset MQ2007 \
  --unlearn_method retrain \
  --unlearn_num 3 \
  --n_clients 10 \
  --interactions_per_feedback 5 \
  --interactions_budget 50000 \
  --learning_rate 0.1 \
  --update True \
  --seed 1 \
  --scenario clean \
  --dataset_root_dir ../datasets \
  --save_dir ../save
```


### Unlearning Outputs

After unlearning, results will also be saved in a `.pkl` file in the `save_dir`.

Example file path:
```
../save/MQ2007/1/clean/Perfect_unlearning_FedRemove_2000.pkl
```

The structure includes the dataset name, fold ID, scenario type, click model, and unlearning method.


## Evaluation
We provide four evaluation metrics: offline NDCG@10, online NDCG@10, Distance Gap, and RelR Difference.

Please note that before evaluation, you need to ensure that the results for each unlearning strategy has been generated.

To visualize offline performance, you can run the script：
```bash
cd evaluation
python offline_ndcg.py --dataset MQ2007 
```

To visualize online performance, you can run the script：
```bash
cd evaluation
python online_ndcg.py --dataset MQ2007
```
To evaluate the RelR Difference and Distance Gap metrics, you need to set the following parameters when running `train.py` and `unlearn.py`:
`--enable_relr True --scenario clean`. Then, you can obtain the results by loading the corresponding pickle file.

## Supplementary results

### Figures for Offline Performance (Mean nDCG@10)
- **MQ2007**
 
  No poisoning
  ![image](https://github.com/Iris1026/Unlearning-for-FOLTR/blob/main/Evaluation/results/MQ2007_Clean_offline.png)

  Data poisoning
  ![image](https://github.com/Iris1026/Unlearning-for-FOLTR/blob/main/Evaluation/results/MQ2007_Data_Poison_offline.png)

  Model poisoning
  ![image](https://github.com/Iris1026/Unlearning-for-FOLTR/blob/main/Evaluation/results/MQ2007_Model_Poison_offline.png)

  
- **MSLR-WEB10K**

  No poisoning
  ![image](https://github.com/Iris1026/Unlearning-for-FOLTR/blob/main/Evaluation/results/MSLR10K_Clean_offline.png)

  Data poisoning
  ![image](https://github.com/Iris1026/Unlearning-for-FOLTR/blob/main/Evaluation/results/MSLR10K_Data_Poison_offline.png)

  Model poisoning
  ![image](https://github.com/Iris1026/Unlearning-for-FOLTR/blob/main/Evaluation/results/MSLR10K_Model_Poison_offline.png)

  
- **Yahoo!**

  No poisoning
  ![image](https://github.com/Iris1026/Unlearning-for-FOLTR/blob/main/Evaluation/results/Yahoo_Clean_offline.png)

  Data poisoning
  ![image](https://github.com/Iris1026/Unlearning-for-FOLTR/blob/main/Evaluation/results/Yahoo_Data_Poison_offline.png)

  Model poisoning
  ![image](https://github.com/Iris1026/Unlearning-for-FOLTR/blob/main/Evaluation/results/Yahoo_Model_Poison_offline.png)

  
- **Istella-S**

  No poisoning
  ![image](https://github.com/Iris1026/Unlearning-for-FOLTR/blob/main/Evaluation/results/istella-s_Clean_offline.png)

  Data poisoning
  ![image](https://github.com/Iris1026/Unlearning-for-FOLTR/blob/main/Evaluation/results/istella-s_Data_Poison_offline.png)

  Model poisoning
  ![image](https://github.com/Iris1026/Unlearning-for-FOLTR/blob/main/Evaluation/results/istella-s_Model_Poison_offline.png)







