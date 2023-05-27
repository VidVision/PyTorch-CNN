# PyTorch-CNN
Implementing CNN using Pytorch

In this work, I have built a convolutional Neural Network architecture using PyTorch. The goal is to apply a CNN Model on the CIFAR10 image data set and test the accuracy of the model on the basis of image classification. CIFAR10 is a collection of images used to train Machine Learning and Computer Vision algorithms. It contains 60K images having dimension of 32x32 with ten different classes such as airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. We train our Neural Net Model specifically Convolutional Neural Net (CNN) on this data set.

## Python and dependencies

Python 3 is used in this project.

- **environment.yaml**
Contains a list of needed libraries. 

$ conda env create -f environment.yaml

## Dataset ##
- **./data**
Download the CIFAR dataset with  provided script under ./data by:

$ cd data

$ sh get_data . sh

$ cd . . /

Microsoft Windows 10 Only

C: \ c o de  folder > cd data

C: \ c o de  folder \ data> get_data . bat

C: \ c o de  folder \ data> cd . .


## Model Implementation ##

- **main.py**

The main function in *main.py* contains the major logic of the code and can be run by

$ python main.py --config configs/<name_of_config_file>.yaml

- **./models**

This folder contains different models to implement:

1. *./models/twolayer.py*   (Two-Layer Network)
 
 The model is built with two fully connected layers and a sigmoid activation function in between the two layers. 
 
2.*./models/cnn.py*  Vanilla Convolutional Neural Network. (CNN : Conv Layer--> ReLU--> MAX pooling --> FC layer)
 A model with a convolution layer, a ReLU activation, a max-pooling layer, followed by a fully connected layer for classification.  
 
3. *./models/my_model.py*

 (Conv Layer--> ReLU--> MAX pooling) -->(Conv Layer--> ReLU--> MAX pooling) --> (Conv Layer--> ReLU--> MAX pooling)
 (Conv Layer--> ReLU--> MAX pooling) --> (FC layer --> ReLU)--> dropout --> (FC layer --> ReLU) --> (FC layer --> ReLU)
  --> FC layer

## Imbalanced Dataset ##

- **./models/resnet.py.**

- **./losses/focal_loss.py**

In practice, datasets are often not balanced. In this section, I explored the limitation of standard training strategy on this type of dataset using unbalanced version of CIFAR-10.
Class-Balanced Focal Loss was implemented as one solution to the imabalnce problem.
These papers have been used for implementation of focal loss: https://arxiv.org/pdf/1901.05555.pdf, https://arxiv.org/abs/1708.02002

