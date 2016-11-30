from __future__ import print_function

import numpy as np
import scipy.misc

import tensorflow as tf

from tensorflow.python.ops import rnn, rnn_cell

from pixelrnntensorflow.ops import *

import os

height = width = 32
channel = 1

class DreamNetwork2:
  def __init__(self, hidden_dims=16, recurrent_length=3, out_recurrent_length=2, out_hidden_dims=32, learning_rate=1e-3):
    self.hidden_dims = hidden_dims
    self.recurrent_length = recurrent_length
    self.out_recurrent_length = out_recurrent_length
    self.out_hidden_dims = out_hidden_dims
    self.learning_rate = learning_rate

  def PixelRNN(self, x, reuse):
    print('Initialize Network')
    with tf.variable_scope('pixelrnn', reuse=reuse):
      conv_inputs = conv2d(x, self.hidden_dims *  2, [7,7], 'A', scope='conv_inputs')

      class meme(object):
        pass

      conf = meme()
      setattr(conf, 'use_residual', True)
      setattr(conf, 'hidden_dims', self.hidden_dims)
      setattr(conf, 'use_dynamic_rnn', False)

      l_hid = conv_inputs
      for idx in xrange(self.recurrent_length):
        l_hid = diagonal_bilstm(l_hid, conf, scope='LSTM{}'.format(idx))

      for idx in xrange(self.out_recurrent_length):
        l_hid = tf.nn.relu(conv2d(l_hid, self.out_hidden_dims, [1,1], 'B', scope='CONV_OUT{}'.format(idx)))

      conv2d_out_logits = conv2d(l_hid, 1, [1,1], 'B', scope='conv2d_out_logits')
      output = tf.nn.sigmoid(conv2d_out_logits)

      loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        conv2d_out_logits, x, name='loss'))

      optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
      grads_and_vars = optimizer.compute_gradients(loss)
      new_grads_and_vars = [(tf.clip_by_value(gv[0], -1, 1), gv[1]) for gv in grads_and_vars]
      optim = optimizer.apply_gradients(new_grads_and_vars)

      return output, optim, loss

  def train(self, x, sess, batch_size=128, niterations=500000):
    print('train()')
    print(x.shape)
    inputs = tf.placeholder(tf.float32, [None, height, width, channel])

    output, optim, loss = self.PixelRNN(inputs, reuse=False)

    print('initialize variables')
    sess.run(tf.initialize_all_variables())

    nex, _, _, _ = x.shape
    for idx in xrange(niterations):
      x_batch = x[np.random.choice(nex, batch_size), :, :, :]
      _, cost = sess.run([
        optim, loss,
      ], feed_dict={ inputs: x_batch })
      print('Iteration: ', idx, 'Loss: ', cost)

  def dream(self):
    pass

  def test(self, x, sess):
    inputs = tf.placeholder(tf.float32, [None, height, width, channel])

    output, optim, loss = self.PixelRNN(inputs, reuse=True)
    return sess.run(loss, feed_dict={ inputs: x })
