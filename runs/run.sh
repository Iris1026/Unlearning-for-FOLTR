python train.py --dataset istella-s --scenario clean 
python train.py --dataset istella-s --scenario data_poison 
python train.py --dataset istella-s --scenario model_poison 

python unlearn.py --dataset istella-s --scenario clean --unlearn_method original
python unlearn.py --dataset istella-s --scenario clean --unlearn_method retrain
python unlearn.py --dataset istella-s --scenario clean --unlearn_method fedEraser
python unlearn.py --dataset istella-s --scenario clean --unlearn_method fineTuning 
python unlearn.py --dataset istella-s --scenario clean --unlearn_method FedRemove
python unlearn.py --dataset istella-s --scenario clean --unlearn_method pga

python unlearn.py --dataset istella-s --scenario data_poison --unlearn_method original
python unlearn.py --dataset istella-s --scenario data_poison --unlearn_method retrain
python unlearn.py --dataset istella-s --scenario data_poison --unlearn_method fedEraser
python unlearn.py --dataset istella-s --scenario data_poison --unlearn_method fineTuning
python unlearn.py --dataset istella-s --scenario data_poison --unlearn_method FedRemove
python unlearn.py --dataset istella-s --scenario data_poison --unlearn_method pga
