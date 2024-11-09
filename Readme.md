# CGAN Implementation with Keras

This repository provides an implementation of Conditional Generative Adversarial Networks (CGANs) using Keras, trained on the MNIST and CIFAR-10 datasets. CGANs allow for conditional generation of images based on class labels, enabling the model to generate images of a specified class.

This implementation:

- Trains a CGAN model on the MNIST and CIFAR-10 datasets.
- Supports flexible conditioning on class labels.
- Provides visualization of generated samples during training.

## Project Structure

- `model.py` - Core script containing the implementation of the Generator, Discriminator and CGAN architectures.
- `train.py` - Script to train the CGAN model on MNIST and CIFAR-10 datasets.
- `datasets.py` - Contains code for loading and processing the datasets.
- `utils.py` - Contains some utility functions.
- `main.py` - The starting point of the project.
- `state/` - Directory to save the trained CGAN model weights and progression outputs.

## MNIST Digit Generation Progression

<img src="state/mnist/animated_progress.gif" alt="Digit Generation Progress" width="500" height="500">

## CIFAR-10 Image Generation Progression


<img src="state/cifar10/animated_progress.gif" alt="Object Generation Progress" width="500" height="500">
