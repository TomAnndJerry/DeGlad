# DeGlad

"Towards Robust Graph-Level Anomaly Detection via Counterfactual Augmentation" published at KBS 2025.

# Operating Environment

Our DeGlad model is implemented using PyTorch Geometric (PyG) 2.6.1 and PyTorch 2.1.1. All experiments are conducted on a GeForce RTX 3090 GPU with 24GB of memory.  

* numpy == 1.24.3  
* pytorch == 2.1.1 

# Dataset

TUDataset (https://chrsmrrs.github.io/datasets/docs/datasets/)

# Usage

Run anomaly detection on COX2 datasets:

```
bash script/COX2.sh
```
