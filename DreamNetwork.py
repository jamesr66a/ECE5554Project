from __future__ import print_function

import numpy as np
import scipy.misc

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

class DreamNetwork:
  def __init__(learning_rate=0.001, lstm_depth=5, n_input=32, n_steps=32,\
               n_hidden=128, n_classes=10, forget_bias=1.0):
    self.learning_rate = learning_rate
    self.lstm_depth = lstm_depth
    self.n_input = n_input
    self.n_steps = n_steps
    self.n_hidden = n_hidden
    self.n_classes = n_classes

    self.arch = {}

  def RNN(self, x, weights, biases):
    x = transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)

    states = x
    for i in range(self.lstm_depth):
      lstm_cell = rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
      outputs, states_t = rnn.rnn(
        lstm_cell, states, dtype=tf.float32, scope='rnn{}'.format(i)
      ) 

      # Residual connection
      if i != 0:
        states = states_t + states
      else:
        states = states_t

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

  def init_network(x):
    with tf.variable_scope('CoreArch') as self.vs:
      weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
      }
      biases = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
      }

      pred = RNN(x, weights, biases)

    self.vs.reuse
    return pred


  def train(x, y, batch_size=128, training_iters=100000, display_step=10):
    with tf.Session() as sess:
      xs = tf.placeholder('float', [None, self.n_steps, self.n_input])
      ys = tf.placeholder('float', [None, n_classes])

      pred = init_network(x)

      cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, ys))
      optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)\
        .minimize(cost)

      correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float_32))

      init = tf.initialize_all_variables()
      sess.run(init)

      step = 1
      while step*batch_size < training_iters:
        data_size, _, _ = np.size(x)
        idxs = np.choice(data_size, batch_size)
        batch_x = x[idxs, :]
        ls = labels[idxs]
        batch_y = np.zeros((batch_size, n_classes))
        for idx in xrange(batch_size):
          batch_y[idx, ls[idx]] = 1.
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_Y})
        if step % display_step == 0:
          acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
          loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
          print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                "{:.6f}".format(loss) + ", Training Accuracy= " + \
                "{:.5f}".format(acc))
        step += 1
    print("Optimization complete!")
   
  def dream(x, y, batch_size=128, training_iters=100000, display_step=10):
    with tf.Session() as sess:
      xs = tf.Variable(x)
      ys = tf.placeholder('float', [None, n_classes])

      pred = init_network(xs)

      cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, ys)
      optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)\
        .minimize(cost, [xs])

      correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float_32))

      init = tf.initialize_all_variables()
      sess.run(init)

      step = 1
      while step*batch_size < training_iters:
        batch_y = np.expand_dims(y, 0)
        sess.run(optimizer, feed_dict={y: batch_y})
        if step % display_step == 0:
          acc = sess.run(accuracy, feed_dict={y: batch_y})
          loss = sess.run(cost, feed_dict={y: batch_y})
          print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                "{:.6f}".format(loss) + ", Training Accuracy= " + \
                "{:.5f}".format(acc))
        step += 1
    print("Optimization complete!")
    return sess.run(xs) 
