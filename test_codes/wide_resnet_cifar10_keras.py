#################################################################
#                                                               #
#  This model has an accuracy of 92 % on CIFAR-10 test set      #
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
print ('Data Normalization ...')
mean = [125.307, 122.95, 113.865]
std  = [62.9932, 62.0887, 66.7048]
for i in range(3):
    x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
    x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
print ('Loading pretrained model ...')
model = keras.models.load_model('../trained_models/wideresnet_cifar10.h5')

print ('Predicting the labels for test set. Please wait ...')
predictions = model.predict(x_test)

print 'Model accuracy on test set is:'
print (np.mean(predictions.argmax(axis=1) == y_test.argmax(axis=1)))