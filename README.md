# MNISTImageClassifier

Basic Image Classifier built in PyTorch using MNIST Dataset.
Can recognize handwritten digits from 0 to 9.

## Network architecture:

```
(In Channel, Out Channel, Convolving Kernel Size)
Input Layer: (1, 32, (3, 3)) { Convolutional 2D layer }
Hidden Layer 1: (32, 64, (3, 3)) { Convolutional 2D layer }
Hidden Layer 2: (64, 64, (3, 3)) { Convolutional 2D layer }
Output Layer: Flattened, 10 neurons { Linear Layer }

Activation Function: ReLU()
```

## Optimizer & Loss Function
```
Optimizer: Adam (LR: 1e-3)
Loss Function: CrossEntropyLoss
```
