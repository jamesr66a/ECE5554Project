from __future__ import print_function

import numpy as np
import scipy.misc

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

class DreamNetwork:
  def __init__(self, learning_rate=0.001, lstm_depth=5, n_input=32, n_steps=32,\
               n_hidden=128, n_classes=10, forget_bias=1.0):
    self.learning_rate = learning_rate
    self.lstm_depth = lstm_depth
    self.n_input = n_input
    self.n_steps = n_steps
    self.n_hidden = n_hidden
    self.n_classes = n_classes

    self.arch = {}

  def RNN(self, x, weights, biases, reuse):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, self.n_input])
    x = tf.split(0, self.n_steps, x)

    states = x
    for i in range(self.lstm_depth):
      with tf.variable_scope('rnn{}'.format(i), reuse=reuse):
        lstm_cell = rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        outputs, states_t = rnn.rnn(
          lstm_cell, states, dtype=tf.float32
        ) 

      # Residual connection
      if i != 0:
        states = states_t + states
      else:
        states = states_t

    return tf.add(tf.matmul(outputs[-1], weights['out']), biases['out'])

  def init_network(self, x, reuse=False):
    with tf.variable_scope('params', reuse=reuse) as scope:
      weights = {
        'out': tf.get_variable('weights', [self.n_hidden, self.n_classes],
          initializer=tf.random_normal_initializer(0., 1.)
        )
      }
      biases = {
        'out': tf.get_variable('biases', [1, self.n_classes],
          initializer=tf.random_normal_initializer(0., 1.)
        )
      }

    pred = self.RNN(x, weights, biases, reuse)

    return pred


  def train(self, x, y, batch_size=100, training_iters=100000, display_step=10):
    with tf.Session() as sess:
      xs = tf.placeholder('float32', [None, self.n_steps, self.n_input])
      ys = tf.placeholder('float32', [None, self.n_classes])

      pred = self.init_network(xs)

      cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, ys))
      optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)\
        .minimize(cost)

      correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(ys, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

      init = tf.initialize_all_variables()
      sess.run(init)

      step = 1
      while step*batch_size < training_iters:
        data_size, _ = x.shape
        idxs = np.random.choice(data_size, batch_size)
        batch_x = x[idxs, :]
        ls = y[idxs]
        batch_y = np.zeros((batch_size, self.n_classes))
        for idx in xrange(batch_size):
          batch_y[idx, ls[idx]] = 1.
        batch_x = batch_x.reshape((batch_size, self.n_steps, self.n_input))
        sess.run(optimizer, feed_dict={xs: batch_x, ys: batch_y})
        if step % display_step == 0:
          acc = sess.run(accuracy, feed_dict={xs: batch_x, ys: batch_y})
          loss = sess.run(cost, feed_dict={xs: batch_x, ys: batch_y})
          print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                "{:.6f}".format(loss) + ", Training Accuracy= " + \
                "{:.5f}".format(acc))
        step += 1
      print("Optimization complete!")
      x_z = np.zeros((1, 32, 32))
      print(sess.run(pred, feed_dict={xs: x_z}))
   
  def dream(self, x, y, training_iters=10000, display_step=10,learning_rate=.1):
    with tf.Session() as sess:
      x = np.reshape(x, (-1, self.n_steps, self.n_input))
      print(x.shape)
      xs = tf.Variable(x, dtype=np.float32)
      ys = tf.placeholder('float32', [None, self.n_classes])

      pred = self.init_network(xs, reuse=True)

      cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, ys))
      optimizer = tf.train.FtrlOptimizer(
        learning_rate=learning_rate, l2_regularization_strength=1.0
      ).minimize(cost, var_list=[xs])

      correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(ys, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

      init = tf.initialize_all_variables()
      sess.run(init)

      step = 1
      while step < training_iters:
        nex, _, _ = x.shape
        batch_y = np.zeros((nex, 10))
        for idx in xrange(nex):
          batch_y[idx, y[idx]] = 1.
        sess.run(optimizer, feed_dict={ys: batch_y})
        if step % display_step == 0:
          acc = sess.run(accuracy, feed_dict={ys: batch_y})
          loss = sess.run(cost, feed_dict={ys: batch_y})
          print("Iter " + str(step) + ", Minibatch Loss= " + \
                "{:.6f}".format(loss) + ", Training Accuracy= " + \
                "{:.5f}".format(acc))
        step += 1
      print("Optimization complete!")
      return sess.run(xs) 

  def test(self, x, y):
    with tf.Session() as sess:
      x = np.reshape(x, (-1, self.n_steps, self.n_input))
      xs = tf.placeholder('float32', [None, self.n_steps, self.n_input])
      ys = tf.placeholder('float32', [None, self.n_classes])
      nex, _, _ = x.shape
      labels = np.zeros((nex, 10))
      for idx in xrange(nex):
        labels[idx, y[idx]] = 1.

      pred = self.init_network(xs, reuse=True)

      correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(ys, 1))
      accuracy = tf.reduce_mean(tf.abs(tf.cast(correct_pred, tf.float32)))

      init = tf.initialize_all_variables()
      sess.run(init)

      print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={xs: x, ys: labels}))
 
      x_z = np.zeros((1, 32, 32))
      print(sess.run(pred, feed_dict={xs: x_z}))

