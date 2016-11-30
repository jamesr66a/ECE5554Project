from __future__ import print_function

import numpy as np
import scipy.misc

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

from pixelrnntensorflow.ops import *

import os

class DreamNetwork:
  def __init__(self, learning_rate=0.001, lstm_depth=5, n_input=32, n_steps=32,\
               n_hidden=128, n_classes=10, forget_bias=1.0, out_recurrent_length=2,
               out_hidden_dims=32):
    self.learning_rate = learning_rate
    self.lstm_depth = lstm_depth
    self.n_input = n_input
    self.n_steps = n_steps
    self.n_hidden = n_hidden
    self.n_classes = n_classes
    self.out_recurrent_length = out_recurrent_length
    self.out_hidden_dims = out_hidden_dims

    self.arch = {}

  def PixelRNN(self, x, weights, biases, reuse):
    with tf.variable_scope('pixelrnn', reuse=reuse):
      conv_inputs = conv2d(x, self.n_hidden * 2, [7,7], "A", scope='pixelrnn')

      class meme(object):
        pass
  
      conf = meme()
      setattr(conf, 'use_residual', True)
      setattr(conf, 'hidden_dims', self.n_hidden)
      setattr(conf, 'use_dynamic_rnn', False)

      l_hid = conv_inputs
      recs = {}
      # main recurrent layers
      for idx in xrange(self.lstm_depth):
        scope = 'LSTM{}'.format(idx)
        recs[scope] = l_hid = diagonal_bilstm(l_hid, conf, scope=scope)

      # output recurrent layers
      for idx in xrange(self.out_recurrent_length):
        scope = 'CONV_OUT{}'.format(idx)
        l_hid = tf.nn.relu(conv2d(l_hid, self.out_hidden_dims, [1,1], 'B', scope=scope))

      conv2d_out_logits = conv2d(l_hid, 1, [1,1], 'B', scope='conv2d_out_logits')

      output = tf.nn.sigmoid(conv2d_out_logits)

      return output 

  def RNN(self, x, weights, biases, reuse):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, self.n_input])
    x = tf.split(0, self.n_steps, x)

    with tf.variable_scope('rnn', reuse=reuse):
      lstm = rnn_cell.BasicLSTMCell(self.n_hidden//2, forget_bias=1.0)
      stacked_lstm = rnn_cell.MultiRNNCell([lstm] * self.lstm_depth)
      stacked_lstm_bw = rnn_cell.MultiRNNCell([lstm] * self.lstm_depth)
      outputs, _, _ = rnn.bidirectional_rnn(stacked_lstm, stacked_lstm_bw, x, dtype=tf.float32)

    return tf.add(tf.matmul(outputs[-1], weights['out']), biases['out'])

  def init_network(self, x, reuse=False):
    with tf.variable_scope('params', reuse=reuse) as scope:
      weights = {
        'out': tf.get_variable('weights', [1024, self.n_classes],
          initializer=tf.random_normal_initializer(0., 1.)
        )
      }
      biases = {
        'out': tf.get_variable('biases', [1, self.n_classes],
          initializer=tf.random_normal_initializer(0., 1.)
        )
      }

      #pred = self.RNN(x, weights, biases, reuse)
      pred = self.PixelRNN(x, weights, biases, reuse)

      return pred, weights, biases


  def train(self, x, y, sess, testx, testy, batch_size=100, training_iters=10000000, display_step=10, test_step=100, savepath=None): 
    xs = tf.placeholder('float32', [None, self.n_steps, self.n_input, 1])
    ys = tf.placeholder('float32', [None, self.n_classes])

    pred, weights, biases = self.init_network(xs)
    print(pred.get_shape())

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, xs))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
    grads_and_vars = optimizer.compute_gradients(cost)
    new_grads_and_vars = \
        [(tf.clip_by_value(gv[0], -1, 1), gv[1]) for gv in grads_and_vars]
    optim = optimizer.apply_gradients(new_grads_and_vars) 

    correct_pred = tf.equal(pred, xs)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    if savepath is not None and os.path.exists(savepath):
      saver = tf.train.Saver()
      saver.restore(sess, savepath)
      print("Model successfully restored from {}".format(savepath))
      return

    init = tf.initialize_all_variables()
    sess.run(init)

    testx = np.reshape(testx, (-1, self.n_steps, self.n_input, 1))
    nex, _, _, _ = testx.shape
    testlabels = np.zeros((nex, 10))
    for idx in xrange(nex):
      testlabels[idx, testy[idx]] = 1.  

    step = 1
    try:
      while step*batch_size < training_iters:
        data_size, _, _, _ = x.shape
        idxs = np.random.choice(data_size, batch_size)
        batch_x = x[idxs, :]
        ls = y[idxs]
        batch_y = np.zeros((batch_size, self.n_classes))
        for idx in xrange(batch_size):
          batch_y[idx, ls[idx]] = 1.
        batch_x = batch_x.reshape((batch_size, self.n_steps, self.n_input, 1))
        sess.run(optim, feed_dict={xs: batch_x, ys: batch_y})
        if step % display_step == 0:
          acc = sess.run(accuracy, feed_dict={xs: batch_x, ys: batch_y})
          #acc = 0.0
          loss = sess.run(cost, feed_dict={xs: batch_x, ys: batch_y})
          print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                "{:.6f}".format(loss) + ", Training Accuracy= " + \
                "{:.5f}".format(acc))
        #if step % test_step == 0:
          print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={xs: testx, ys: testlabels}))
        step += 1
    except KeyboardInterrupt:
      pass
    print("Optimization complete!")

    saver = tf.train.Saver()
    saver.save(sess, savepath)
    print("Model saved to {}".format(savepath))
 
  def dream(self, x, y, sess, training_iters=500, display_step=10,learning_rate=.1):
    x = np.reshape(x, (-1, self.n_steps, self.n_input, 1))
    xs = tf.Variable(x, dtype=np.float32)
    ys = tf.placeholder('float32', [None, self.n_classes])

    pred, weights, biases = self.init_network(xs, reuse=True)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, x))

    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(learning_rate, global_step, 1000, 0.5)
    optimizer = tf.train.AdamOptimizer(
      learning_rate=lr
    ).minimize(cost, var_list=[xs], global_step=global_step)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    to_init = [xs]
    for var in tf.all_variables():
      if not sess.run(tf.is_variable_initialized(var)):
        to_init.append(var)

    sm = tf.nn.softmax(pred)

    sess.run(tf.initialize_variables(to_init))

    step = 1
    while True:
      nex, _, _, _ = x.shape
      batch_y = np.zeros((nex, 10))
      for idx in xrange(nex):
        batch_y[idx, y[idx]] = 1.
      sess.run(optimizer, feed_dict={ys: batch_y})
      p = sess.run(sm, feed_dict={ys: batch_y})
      if step % display_step == 0:
        acc = sess.run(accuracy, feed_dict={ys: batch_y})
        loss = sess.run(cost, feed_dict={ys: batch_y})
        print("Iter " + str(step) + ", Minibatch Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))
        print(p)
      step += 1
      b = True
      for i in xrange(nex):
        if p[i, y[i]] < 0.95:
          b = False
      if b:
        break
    print("Optimization complete!")

    return sess.run(xs) 

  def test(self, x, y, sess):
    x = np.reshape(x, (-1, self.n_steps, self.n_input, 1))
    xs = tf.placeholder('float32', [None, self.n_steps, self.n_input, 1])
    ys = tf.placeholder('float32', [None, self.n_classes])
    nex, _, _, _ = x.shape
    labels = np.zeros((nex, 10))
    for idx in xrange(nex):
      labels[idx, y[idx]] = 1.

    pred, weights, biases = self.init_network(xs, reuse=True)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.abs(tf.cast(correct_pred, tf.float32)))

    print("Testing Accuracy:", \
      sess.run(accuracy, feed_dict={xs: x, ys: labels}))
