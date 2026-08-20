# Neural Network Implementation from Scratch

**Course:** Generative AI Lab — T.Y. Tech, Dept. of CSE (AIML)
**Assignment:** Practice Lab Assignment 1 — Individual Task

## Overview
A simple feedforward neural network built entirely from scratch using NumPy — no deep learning frameworks (TensorFlow, PyTorch, Keras) were used for the model itself. Covers the forward pass, backpropagation, and training with gradient descent.

## Dataset
**EMNIST — Letters split** (Extended MNIST): 28x28 grayscale handwritten letter images, 26 classes (A-Z).

Loaded via `torchvision.datasets.EMNIST`, which downloads from NIST's official servers (more reliable than the `emnist` PyPI package, which relies on Google Drive and can fail with a "BadZipFile" error).

```python
!pip install torchvision --quiet

import numpy as np
import torchvision

train_data = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, download=True)
test_data = torchvision.datasets.EMNIST(root='./data', split='letters', train=False, download=True)

X_train_img = train_data.data.numpy()
y_train_raw = train_data.targets.numpy()
X_test_img = test_data.data.numpy()
y_test_raw = test_data.targets.numpy()

# torchvision's EMNIST images are rotated/transposed compared to normal - this fixes it
X_train_img = np.transpose(X_train_img, (0, 2, 1))
X_test_img = np.transpose(X_test_img, (0, 2, 1))
```

Training uses a 5,000-image subsample (and 1,000 for testing) for fast training on plain NumPy/CPU.

## Architecture
| Layer  | Size | Activation |
|--------|------|------------|
| Input  | 784  | —          |
| Hidden | 64   | ReLU       |
| Output | 26   | Softmax    |

- Loss: Cross-Entropy Loss
- Optimizer: Gradient Descent (learning rate 0.5, 300 epochs)
- Weight init: small random values (x0.01)

## How to Run
1. Install dependencies: `pip install numpy matplotlib torchvision`
2. Open `FirstName_LastName_GenerativeAILabAssignment.ipynb` in Jupyter or Colab
3. Run all cells top to bottom
   - First run downloads EMNIST automatically (~536MB, one-time, needs internet — may take a few minutes)
   - If the download fails, try clearing `./data` and re-running, or switch to the Kaggle CSV fallback (see notebook)

## Files
- `Himanshu_Gharde_GenerativeAILabAssignment.ipynb` — main notebook (all code + explanations)
- `README.md` — this file

## Author
202401110007 Himanshu Gharde
