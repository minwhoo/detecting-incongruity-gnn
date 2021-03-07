# Learning to Detect Incongruence in News Headline and Body Text via a Graph Neural Network
Official PyTorch implementation of Graph Hierarchical Dual Encoder model from the following paper:

**Learning to Detect Incongruence in News Headline and Body Text via a Graph Neural Network**, IEEE Access, 2021, [paper](https://ieeexplore.ieee.org/document/9363185)

## Requirements
 - Python 3.7
 - PyTorch 1.2.0
 - PyTorch Geometric 1.3.2
 - GloVe word embeddings

## Installation
```
pip install numpy==1.17.2
pip install scikit-learn==0.21.3
pip install torch==1.2.0
pip install --verbose --no-cache-dir torch-scatter==1.3.2
pip install --verbose --no-cache-dir torch-sparse==0.4.3
pip install --verbose --no-cache-dir torch-cluster==1.4.5
pip install torch-geometric==1.3.2
```

## Download Dataset

We are providing the dataset download link to researchers for non-commercial research purposes only. Please request through the [google form link](https://forms.gle/dMiWJAnBjdXvhPMy8).

## Preprocess
```
python preprocessing.py --dataset-path paragraph_swap_news_{random,similar} \
                        --glove-path $GLOVE_DATA_DIR/glove.840B.300d.txt
```
## Train
```
python train.py --processed-data-path paragraph_swap_news_{random,similar}/processed \
                --lr 0.001 \
                --batch-size 120 \
                --min-iterations 50000 \
                --train \
                --eval \
                --para-level-supervision True \
                --edge-supervision True \
                --save-checkpoint
```
