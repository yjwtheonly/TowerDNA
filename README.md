# TowerDNA

Official implementation for **TowerDNA: Fast and Accurate Graph Retrieval with Dividing, Contrasting and Alignment**

## Introduction

Graph retrieval (GR), a ranking procedure that aims to sort the graphs in a database by their relevance to a query graph in decreasing order, has wide applications across diverse domains, such as visual object detection and drug discovery.
Existing Graph Retrieval (GR) approaches usually compare graph pairs at a detailed level and generate quadratic similarity scores.
In realistic scenarios, conducting quadratic fine-grained comparisons is costly.
However, coarse-grained comparisons would result in performance loss.
Moreover, label scarcity in real-world data brings extra challenges.
To tackle these issues, we investigate a more realistic GR problem, namely, efficient graph retrieval (EGR).
Our key intuition is that, since there are numerous underutilized unlabeled pairs in realistic scenarios, by leveraging the additional information they provide, we can achieve speed-up while simplifying the model without sacrificing performance.
Following our intuition, we propose an efficient model called Dual-**Tower** Model with **D**ividing, Co**n**trasting and **A**lignment (TowerDNA).
TowerDNA utilizes a GNN-based dual-tower model as a backbone to quickly compare graph pairs in a coarse-grained manner.
In addition, to effectively utilize unlabeled pairs, TowerDNA first identifies confident pairs from unlabeled pairs to expand labeled datasets.
It then learns from remaining unconfident pairs via graph contrastive learning with geometric correspondence.
To integrate all semantics with reduced biases, TowerDNA generates prototypes using labeled pairs, which are aligned within both confident and unconfident pairs.
Extensive experiments on diverse realistic datasets demonstrate that TowerDNA achieves comparable performance to fine-grained methods while providing a 10 $\times$ speed-up.

## Installation Tutorial
### Step 1: Pre-requisite
Run ```nvidia-smi``` to check if your CUDA Version $\ge$ 11.3. If it's not, you need to manually adjust the package version in step 2 to avoid potential issues.

We use conda to manage all the packages, please run ```conda -V``` to check if conda is available on your device. If not, please follow the relevant tutorial to install conda (Anaconda). Here is the official tutorial: https://docs.anaconda.com/free/anaconda/install/index.html.

### Step 2: Create TowerDNA environment and install dependencies
Run the following command in the terminal:
```
conda create --name TowerDNA python=3.8.12
conda activate TowerDNA
conda install cudatoolkit==11.3.1 cudnn==8.2.1
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

This will take a few minutes to install the necessary packages. Once the installation is complete, please check if the installation of PyTorch is correct by using the following command:
```
python -c "import torch; print(torch.cuda.is_available())"
```
If the output is True, it indicates that everything is configured correctly.

## How to use our code

You can run our model with a single command:
```
python train_one_way.py --dataset <DATANAME>
```
where <DATANAME> is the name of the dataset to be tested, for example, MUTAG.
