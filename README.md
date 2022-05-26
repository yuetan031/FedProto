# FedProto: Federated Prototype Learning across Heterogeneous Clients

Implementation of the paper accepted by AAAI 2022 : [FedProto: Federated Prototype Learning across Heterogeneous Clients](https://arxiv.org/abs/2105.00243).

## Requirments
This code requires the following:
* Python 3.6 or greater
* PyTorch 1.6 or greater
* Torchvision
* Numpy 1.18.5

## Data Preparation
* Download train and test datasets manually from the given links, or they will use the defalt links in torchvision.
* Experiments are run on MNIST, FEMNIST and CIFAR10.
http://yann.lecun.com/exdb/mnist/
https://s3.amazonaws.com/nist-srd/SD19/by_class.zip
http://www.cs.toronto.edu/∼kriz/cifar.html

## Running the experiments
The baseline experiment trains the model in the conventional way.

* To train the FedProto on MNIST with n=3, k=100 under statistical heterogeneous setting:
```
python federated_main.py --mode task_heter --dataset mnist --num_classes 10 --num_users 20 --ways 3 --shots 100 --stdev 2 --rounds 100 --train_shots_max 110 --ld 1
```
* To train the FedProto on FEMNIST with n=4, k=100 under both statistical and model heterogeneous setting:
```
python federated_main.py --mode model_heter --dataset femnist --num_classes 62 --num_users 20 --ways 4 --shots 100 --stdev 2 --rounds 120 --train_shots_max 110 --ld 1
```
* To train the FedProto on CIFAR10 with n=5, k=100 under statistical heterogeneous setting:
```
python federated_main.py --mode task_heter --dataset cifar10 --num_classes 10 --num_users 20 --ways 5 --shots 100 --stdev 2 --rounds 110 --train_shots_max 110 --ld 0.1 --local_bs 32
```



You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: 'mnist'. Options: 'mnist', 'femnist', 'cifar10'
* ```--num_classes:```  Default: 10. Options: 10, 62, 10
* ```--mode:```     Default: 'task_heter'. Options: 'task_heter', 'model_heter'
* ```--seed:```     Random Seed. Default set to 1234.
* ```--lr:```       Learning rate set to 0.01 by default.
* ```--momentum:```       Learning rate set to 0.5 by default.
* ```--local_bs:```  Local batch size set to 4 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.


#### Federated Parameters
* ```--mode:```     Default: 'task_heter'. Options: 'task_heter', 'model_heter'
* ```--num_users:```Number of users. Default is 20.
* ```--ways:```      Average number of local classes. Default is 3.
* ```--shots:```      Average number of samples for each local class. Default is 100.
* ```--test_shots:```      Average number of test samples for each local class. Default is 15.
* ```--ld:```      Weight of proto loss. Default is 1.
* ```--stdev:```     Standard deviation. Default is 1.
* ```--train_ep:``` Number of local training epochs in each user. Default is 1.


## Citation
If you find this project helpful, please consider to cite the following paper:
```
@inproceedings{tan2021fedproto,
  title={FedProto: Federated Prototype Learning across Heterogeneous Clients},
  author={Tan, Yue and Long, Guodong and Liu, Lu and Zhou, Tianyi and Lu, Qinghua and Jiang, Jing and Zhang, Chengqi},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2022}
}
```
