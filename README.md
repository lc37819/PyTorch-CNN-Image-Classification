## PyTorch CNN CIFAR 10 Image Classification

Utilizing a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset using PyTorch. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The classes include 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', and 'truck'.

![CIFAR-10](https://pytorch.org/tutorials/_static/img/cifar10.png)

## Overview

This tutorial covers the following steps:

1. Loading and normalizing the CIFAR-10 training and test datasets using `torchvision`.
2. Defining a Convolutional Neural Network (CNN).
3. Defining a loss function and an optimizer.
4. Training the network on the training data.
5. Testing the network on the test data.

## Getting Started

### Prerequisites

To run this tutorial, you need to have the following packages installed:

- `torch`
- `torchvision`
- `matplotlib`
- `numpy`

You can install these packages using `pip`:

```bash
pip install torch torchvision matplotlib numpy
```

### Running the Tutorial

1. **Load and Normalize CIFAR-10**

2. **Define a Convolutional Neural Network**

3. **Define a Loss Function and Optimizer**

4. **Train the Network**

5. **Test the Network**

## Goal

Enhance network accuracy through modifying network hyperparameters, experimenting with different convolutional layer configurations, and incorporating dropout.

## Improvements
1. Increase Model Complexity
   
   Add more convolutional layers and use Batch Normalization to improve the model's learning ability.

2. Use Data Augmentation

   Data augmentation techniques like random horizontal flip, random crop, and random rotation can help improve the model's robustness.

3. Use a Different Optimizer

   Using Adam optimizer instead of SGD can lead to faster convergence.

## Saving the Model

To save the trained model, use the following code in your Jupyter notebook:

```python
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
```

For more details on saving and loading models in PyTorch, see [PyTorch's documentation](https://pytorch.org/docs/stable/notes/serialization.html).

## Conclusion

This repository demonstrated how to train a Convolutional Neural Network on the CIFAR-10 dataset using PyTorch. By following the steps outlined, you can load and preprocess the dataset, define a neural network, train it, and evaluate its performance.

For further exploration, you can experiment with different network architectures, hyperparameters, and training strategies to improve the model's performance.

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TorchVision Documentation](https://pytorch.org/vision/stable/index.html)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project is based on the [CIFAR-10 tutorial](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/4e865243430a47a00d551ca0579a6f6c/cifar10_tutorial.ipynb) from the official PyTorch tutorials.
- Thanks to the PyTorch team for providing extensive documentation and tutorials.
