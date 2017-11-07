#################################################################
#                                                               #
#  This model has an accuracy of 93.41 % on CIFAR-10 test set   #
#                                                               #
#################################################################
import keras
import numpy as np
from keras.datasets import cifar10
from keras import backend as K
import tensorflow as tf

tf.reset_default_graph()
sess = tf.Session()
K.set_session(sess)

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

x = tf.placeholder(tf.float32,[None,32,32,3])
y = tf.placeholder(tf.float32,[None,10])

print ('Loading pretrained model ...')
model = keras.models.load_model('../trained_models/vgg19_cifar10.model')


y_pred_t = model(x) #y_pred_t is the output tensor
correct =0
print ('Predicting the labels for test set. Please wait ...')
batch_size = 64
for i in range(0,len(x_test),batch_size):
    x_batch = x_test[i:i+batch_size]
    y_batch = y_test[i:i+batch_size]
    predictions = sess.run(y_pred_t,{x:x_batch, K.learning_phase():0})
    correct+=np.sum(predictions.argmax(axis=1) == y_batch.argmax(axis=1))
accuracy=correct/(len(x_test)+0.0)



print 'Model accuracy on test set is:'
print (accuracy)