from DreamNetwork import DreamNetwork
import numpy as np
import scipy.misc
from tensorflow.models.image.cifar10 import cifar10
from util import *
import tensorflow as tf

def load_cifar_data(path):
  d = unpickle(path)
  data = d['data']
  labels = np.array(d['labels'])
  data = np.linalg.norm(data.reshape((-1, 3, 1024)), axis=1)

  return data, labels 

def main():
  with tf.Session() as sess:
    dn = DreamNetwork(n_hidden=16, lstm_depth=7, learning_rate=1e-3)

    n_test = 100

    savepath = "./model.ckpt"
    data, labels = load_cifar_data('cifar-10-batches-py/data_batch_1')
    data = np.reshape(data, [-1, 32, 32, 1])
    test_data, test_labels = load_cifar_data('cifar-10-batches-py/test_batch')
    test_data = np.reshape(test_data, [-1, 32, 32, 1])
    dn.train(data, labels, sess, test_data[0:n_test, :], test_labels[0:n_test], training_iters=500000, savepath=savepath)

    dn.test(test_data[0:n_test, :], test_labels[0:n_test], sess)

    scipy.misc.imsave(
      'orig.jpg', np.squeeze(np.reshape(test_data[0, :], (32, 32)))\
        .repeat(2, axis=0).repeat(2, axis=1)
    )

    dream_img = dn.dream(
      np.tile(np.zeros((32, 32)), 10), xrange(10),
      sess, learning_rate=1.0, training_iters=100, display_step=1
    ) 

    for idx in xrange(10):
      scipy.misc.imsave(
        'dream{}.bmp'.format(idx), np.squeeze(dream_img)[idx,:,:].repeat(8, axis=0).repeat(8, axis=1)
      )


if __name__ == '__main__':
  main()
