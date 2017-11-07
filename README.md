# cifar10-pretrained-models
This repository contains 2 different pretrained models on cifar-10 dataset with 92% and 93.41% accuracy.

I couldn't find any pretrained model for cifar-10 dataset. So I trained two different models and put them in this repository.

These models are trained with Keras. I got the code for training these models from https://github.com/BIGBALLON/cifar-10-cnn and made a few changes to them to train them using multiple GPUs by using this script: https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py. 


train_model: This directory contains the code for training the vgg model and wide resnet on cifar10.

trained_models: This directory contains the vgg and wide resnet pretrained models. These models are trained using four Tesla K80 GPUs.
The vgg model is trained with data augmentation for 60 epochs (almost 2 hours) and achieves an accuracy of 93.41%
The wide resnet model is trained without data augmentation for for 200 epochs (almost 7 hours) and achieves an accuracy of 92%

test_codes: This directory contains sample codes to load these models and calculate the accuracy in Keras and TensorFlow.


