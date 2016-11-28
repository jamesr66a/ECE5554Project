'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import numpy as np
import scipy.misc

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.models.image.cifar10 import cifar10

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

# Parameters
learning_rate = 0.001
training_iters = 1000000
batch_size = 1024
display_step = 10
lstm_depth = 5

# Network Parameters
n_input = 32 # MNIST data input (img shape: 28*28)
n_steps = 32 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    states = x
    for i in xrange(lstm_depth):
      # Define a lstm cell with tensorflow
      lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

      # Get lstm cell output
      outputs, states_t = rnn.rnn(
        lstm_cell, states, dtype=tf.float32, scope='rnn{}'.format(i)
      )

      if i != 0:
        states = states_t + states
      else:
        states = states_t

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    d = unpickle('cifar-10-batches-py/data_batch_1')
    data = d['data']
    labels = np.array(d['labels'])
    data = np.linalg.norm(data.reshape((-1, 3, 1024)), axis=1)
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        idxs = np.random.choice(10000, batch_size)
        batch_x = data[idxs, :]
        ls = labels[idxs]
        batch_y = np.zeros((batch_size, n_classes))
        for idx in xrange(batch_size):
          batch_y[idx, ls[idx]] = 1.
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_d = unpickle('cifar-10-batches-py/data_batch_1')
    test_data = d['data']
    test_ls = np.array(d['labels'])
    test_data = np.linalg.norm(test_data.reshape((-1, 3, 1024)), axis=1)\
      .reshape((-1, n_steps, n_input))
    nex, _, _ = test_data.shape
    test_labels = np.zeros((nex, n_classes))
    for idx in xrange(nex):
      test_labels[idx, test_ls[idx]] = 1.
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_labels}))

    #imtest_template = np.random.randn(32, 32)
    imtest_template = test_data[0, :, :]
    imtest = np.zeros((10, 32, 32))
    for idx in xrange(10):
      imtest[idx, :, :] = imtest_template
    labeltest = np.identity(10)

    while True:
      var_grad = tf.gradients(cost, [x])[0]
      imgrad = sess.run(var_grad, feed_dict={x: imtest, y: labeltest})
      loss = sess.run(cost, feed_dict={x: imtest, y: labeltest})

      adjusted = imtest - 100*imgrad
      print(
        'loss', loss, 'delta', np.linalg.norm(adjusted[0,:,:] - imtest[0,:,:])
      )
      imtest = adjusted

      if loss < 1:
        break

    for num in range(10):
      scipy.misc.imsave(
        'out/adjusted{}.jpg'.format(num),
        np.squeeze(imtest[num,:,:]).repeat(2, axis=0).repeat(2, axis=1)
      )
