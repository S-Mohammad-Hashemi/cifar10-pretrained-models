#################################################################
#                                                               #
#  This model has an accuracy of 93.41 % on CIFAR-10 test set   #
#                                                               #
#################################################################
import keras
import numpy as np
from keras.datasets import cifar10
num_classes= 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# x_test shape is: (10000,32,32,3)
print ('x_test shape: ',x_test.shape)
print ('Pixel range for this model is between -1.0 to 1.0')
x_train=x_train/255.0
x_train=x_train*2.0-1.0

x_test=x_test/255.0
x_test=x_test*2.0-1.0
print ('Loading pretrained model ...')
model = keras.models.load_model('../trained_models/vgg19_cifar10.model')

print ('Predicting the labels for test set. Please wait ...')
predictions = model.predict(x_test)

print 'Model accuracy on test set is:'
print (np.mean(predictions.argmax(axis=1) == y_test.argmax(axis=1)))