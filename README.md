## PyTorch implementation of Recurrent Event Network (RE-Net)

Paper: TPmod: A Tendency-Guided Prediction Model for Temporal Knowledge Graph Completion



This repository contains the implementation of the TPmod architectures described in the paper.

## Installation
Install PyTorch (>= 1.1.0)  following the instuctions on the [PyTorch](https://pytorch.org/) .
Our code is written in Python3.

## Train and Test
Before running, you should preprocess datasets.

```bash
python3 data/DATA_NAME/get_history_new.py
```

Then, we are ready to train and test.
We first train the model.

```bash
python3 train.py -dataset DATA_NAME -cuda 0 -dim 256 -lr 1e-4 -epochs 50 -b 1024 -dropout 0.5
```

We are ready to test!
```bash
python3 test.py -dataset DATA_NAME -cuda 0 -dim 256 -lr 1e-4 -epochs 50 -b 1024 -dropout 0.5
```

## Datasets
There are four datasets: two with TGvals: GDELT-5 and ICEWS-250. and two with IGvals: GDELT-5I and ICEWS-250I.
Each data folder has 'stat.txt', 'train.txt', 'valid.txt', 'test.txt', 'rel2val.txt' and 'get_history_new.py'.

- 'get_history_new.py': This is for getting history.
-  'rel2val.txt' : This is the file mapping relations to TGvals (IGvals)
- 'stat.txt': First value is the number of entities, and second value is the number of relations.
- 'train.txt', 'valid.txt', 'test.txt': First column is subject entities, second column is relations, and third column is object entities. The fourth column is time.

## Baselines
We use the following public codes for baselines and hyperparameters. 

| Baselines   | Code                                                         | parameters                  |
| ----------- | ------------------------------------------------------------ | --------------------------- |
| TransE      | [Link](https://github.com/jimmywangheng/knowledge_representation_pytorch) | { lr=0.0001, dim=512,b=512} |
| TTransE     | [link](https://github.com/INK-USC/RE-Net)                    | { lr=0.001, dim=512,b=512}  |
| DE-SimplE   | [link](https://github.com/BorealisAI/DE-SimplE)              | { lr=0.001, dim=128,b=512}  |
| TA-DistMult | [link](https://github.com/INK-USC/RE-Net)                    | { lr=0.001, dim=512,b=1024} |
| RE-Net      | [link](https://github.com/INK-USC/RE-Net)                    | { lr=0.001, dim=256,b=1024} |


We implemented RESCAL, DistMult refer to [RotatE](: https://github.com/DeepGraphLearning/ KnowledgeGraphEmbedding.). The user can run the baselines by the following command.

```bash
cd ./baselines
bash run.sh train MODEL_NAME DATA_NAME 0 0 512 1024 512 200.0 0.0005 10000 8 0
```

The user can find implementations in the 'baselines' folder.
