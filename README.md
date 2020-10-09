# Deep-learning
This Repository has demonstrated Concepts and Projects in the field of deep Learning.

Following is the project list

1. [Bike-Sharing](bike-sharing-company-problem)
2. [Dog-Breed-Classifier](dog-breed-classification-problem)

## Bike Sharing Company Problem

### Problem Statement

A bike-sharing company wants to predict how many cycles it needs. If  the company buys too few cycles it looses money from potential riders. And if it has too many, then it is wasting money on bikes that are just sitting around. So, the task is to predict from the historical [data](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset), how many bikes will the company need in the near future. A good way to do is with a Neural Network.

### Solution

`Deep-learning/Project-bikesharing` contains a Jupyter Notebook file `Neural_Network.ipynb` which demonstrates step by step procedure to train build and test the Neural Network.`Deep-learning/Neural-Net_class.py`defines `class NeuralNetwork`  which contains functions for training the Neural Networks.  

This Project focuses on the following concepts.

1. Forward Pass Algorithm
2. Gradient Descent
3. Backpropagation

## Dog Breed Classification Problem

### Problem Statement

Given a user-supplied image of a dog, your algorithm should identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

### Solution

The goal of the project is how to build a pipeline to process real-world, user-supplied images and classify them using the state-of-art CNN models (**PyTorch** **Framework**). 

`Deep-learning/Dog-Breed-Classifier` contains a jupyter Notebook file `dog-breed-classifier`which demonstrates step by step implementation of the project.

Steps:

1. Detect Humans-  **OpenCV**'s implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html)  used to detect human faces in images.  
2. Detect Dogs- we use a [pre-trained model](http://pytorch.org/docs/master/torchvision/models.html) to detect dogs in images.

3. Create a CNN to Classify Dog Breeds (from Scratch)
   1. Specification of Data Loaders for the dog Dataset- **Data Augmentation** is incorporated in this procedure to increase the accuracy of the model.
   2. CNN Model Architecture
   3. Specification of Loss function and Optimizer
   4. Train and Validate the model
   5. test the model
4. Create a CNN to Classify Dog Breeds (Using **Transfer Learning**)
   1. Specification of Data Loaders for the dog Dataset
   2. CNN Model Architecture
   3. Specification of Loss function and Optimizer
   4. Train and Validate the model
   5. test the model
5. Writing an algorithm to accept the user supplied Images to classify the images as Dog, Human or Neither
6. Testing the algorithm.