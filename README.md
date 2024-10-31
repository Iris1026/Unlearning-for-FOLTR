# Unlearning for Federated Online Learning to Rank
## Download datasets
In the paper, we use four popular LTR datasets: MQ2007, MSLR-WEB10K, Yahoo! and Istella-S.
- MQ2007 can be downloaded from the Microsoft Research [website](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/). 
- MSLR-WEB10K can be downloaded from the Microsoft Research [website](https://www.microsoft.com/en-us/research/project/mslr/).  
- Yahoo! can be downloaded from [Yahoo Webscope program](https://webscope.sandbox.yahoo.com/catalog.php?datatype=c).
- Istella-S can be downloaded from [Instella-S website](https://istella.ai/datasets/letor-dataset/)

After downloading data files, they have to be unpacked within the `./datasets` folder.
## Environment Setups
`pip install -r requirements.txt`

## Run
To reproduce our experiments, please run files `./runs/model_poi.py` and `./runs/data_poi.py`

## Figures for RQ3
- **MQ2007**
  ![image](https://github.com/Iris1026/Unlearning-for-FOLTR/blob/main/figures/MQ2007_per.png)
  ![image](https://github.com/Iris1026/Unlearning-for-FOLTR/blob/main/figures/MQ2007_nav.png)
  ![image](https://github.com/Iris1026/Unlearning-for-FOLTR/blob/main/figures/MQ2007_inf.png)
- **MSLR-WEB10K**
- **Yahoo!**
- **Istella-S**

