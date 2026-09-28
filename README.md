# fashion-mnist-optuna

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adityakumar626/fashion-mnist-optuna/blob/main/fashion_mnist_optuna.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Optuna](https://img.shields.io/badge/Optuna-5.0.0-007ACC?logo=optuna&logoColor=white)](https://optuna.org/)

An end-to-end PyTorch pipeline for Fashion-MNIST classification using a dynamically constructed Multilayer Perceptron (MLP) with Bayesian hyperparameter tuning via Optuna.

---

## ⚡ Highlights

- **Dynamic Network Depth & Width**: Automatically evaluates architectures from 1 to 5 hidden layers with 8 to 128 neurons per layer.
- **Layer Regularization**: Each hidden block stacks `Linear ➔ BatchNorm1d ➔ ReLU ➔ Dropout`.
- **Bayesian Optimization**: Tunes architecture, optimizers (`Adam`, `SGD`, `RMSprop`), learning rate, weight decay, epochs, and batch size.
- **Top Accuracy**: Achieved **89.76%** validation accuracy on test splits.

---

## 🏗️ Model Architecture

```text
Input Vector (784)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Linear(in_features, neurons_per_layer)                │
│  BatchNorm1d(neurons_per_layer)                        │  x (num_hidden_layers)
│  ReLU()                                                │
│  Dropout(p=dropout_rate)                               │
└────────────────────────────────────────────────────────┘
       │
       ▼
Linear(neurons_per_layer, 10)
       │
       ▼
CrossEntropyLoss Logits
```
---

## 🔍 Hyperparameter Search Space

| Hyperparameter | Type | Search Range |
| :--- | :--- | :--- |
| `num_hidden_layers` | Integer | `1` to `5` |
| `neurons_per_layer` | Integer (`step=8`) | `8` to `128` |
| `epochs` | Integer (`step=10`) | `10` to `50` |
| `learning_rate` | Float (`log=True`) | `1e-5` to `1e-1` |
| `dropout_rate` | Float (`step=0.1`) | `0.1` to `0.5` |
| `batch_size` | Categorical | `16`, `32`, `64`, `128` |
| `optimizer` | Categorical | `Adam`, `SGD`, `RMSprop` |
| `weight_decay` | Float (`log=True`) | `1e-5` to `1e-3` |

---

## 📊 Optimization Results

### 🏆 Best Trial (Trial 6) — **89.76% Accuracy**

```yaml
Validation Accuracy: 89.76% (0.89758)
num_hidden_layers: 4
neurons_per_layer: 128
epochs: 40
batch_size: 64
optimizer: Adam
learning_rate: 2.5618e-05
dropout_rate: 0.1
weight_decay: 2.3415e-04
```


### Full 10-Trial Benchmark

| Trial | Accuracy | Layers | Neurons | Epochs | Batch Size | Optimizer | Learning Rate | Dropout | Weight Decay |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 86.24% | 2 | 32 | 30 | 128 | Adam | 4.82e-2 | 0.3 | 1.49e-4 |
| **1** | 84.02% | 2 | 24 | 50 | 16 | SGD | 7.63e-3 | 0.3 | 4.64e-4 |
| **2** | 88.83% | 3 | 120 | 30 | 64 | SGD | 5.41e-4 | 0.4 | 2.73e-5 |
| **3** | 88.78% | 1 | 120 | 20 | 32 | Adam | 7.55e-3 | 0.3 | 2.36e-5 |
| **4** | 88.15% | 1 | 56 | 40 | 64 | SGD | 6.43e-3 | 0.3 | 1.53e-5 |
| **5** | 86.92% | 3 | 80 | 40 | 32 | RMSprop | 5.43e-5 | 0.1 | 8.20e-4 |
| **6** 🏆 | **89.76%** | **4** | **128** | **40** | **64** | **Adam** | **2.56e-5** | **0.1** | **2.34e-4** |
| **7** | 89.41% | 2 | 112 | 20 | 64 | Adam | 2.71e-5 | 0.3 | 3.49e-5 |
| **8** | 88.04% | 3 | 48 | 10 | 64 | RMSprop | 2.88e-3 | 0.3 | 3.33e-5 |
| **9** | 88.48% | 1 | 104 | 20 | 64 | Adam | 1.42e-3 | 0.5 | 1.17e-4 |

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/Adityakumar626/fashion-mnist-optuna.git
cd fashion-mnist-optuna
pip install torch torchvision pandas scikit-learn matplotlib optuna
```


### 2. Dataset Setup (Kaggle)
Download `fashion-mnist_train.csv` directly from [Kaggle Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist):
```bash
# Using Kaggle CLI (optional)
kaggle datasets download -d zalando-research/fashionmnist
unzip fashionmnist.zip
```

```text
fashion-mnist-optuna/
├── fashion-mnist_train.csv
├── train.py
└── README.md
```

### 3. Run Optimization

```bash
python train.py
```

---

## 💻 Implementation (`train.py`)

<details>
<summary>Click to view complete reproducible code</summary>

```python
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna

# Reproducibility & Device
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and scale data
df = pd.read_csv("fashion-mnist_train.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, 1:] / 255.0, df.iloc[:, 0], test_size=0.2, random_state=42
)

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_ds = CustomDataset(X_train, y_train)
test_ds = CustomDataset(X_test, y_test)

# Dynamic MLP Architecture
class myNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers, neurons_per_layer, dropout_rate):
        super().__init__()
        layers = []
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, neurons_per_layer))
            layers.append(nn.BatchNorm1d(neurons_per_layer))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = neurons_per_layer

        layers.append(nn.Linear(neurons_per_layer, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Optuna Objective Function
def objective(trial):
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 5)
    neurons_per_layer = trial.suggest_int("neurons_per_layer", 8, 128, step=8)
    epochs = trial.suggest_int("epochs", 10, 50, step=10)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = myNN(784, 10, num_hidden_layers, neurons_per_layer, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training
    model.train()
    for epoch in range(epochs):
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    return correct / total

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("\n================== Optimization Complete ==================")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Accuracy: {study.best_value * 100:.2f}%")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  • {k}: {v}")
```
</details>
