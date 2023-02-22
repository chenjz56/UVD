# UVD

## code
This is the source code for Paper: User View Dynamic Graph-Driven Sequential Recommendation.  We have implemented our methods in PyThorch.

## Usage

Train and evaluate the model:

~~~~
python build_graph.py --dataset baby-1000 --similar_rate 0.3
python main.py --dataset baby-1000
~~~~

## Requirements

- torch = 1.12.0+cu102
- torch-cluster = 1.6.0
- torch-geometric = 2.1.0.post1
- torch-scatter = 2.0.9
- torch-sparse = 0.6.14
- python = 3.8.13
- numpy = 1.23.3