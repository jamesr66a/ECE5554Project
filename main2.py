from DreamNetwork2 import DreamNetwork2
import numpy as np
import scipy.misc
from tensorflow.models.image.cifar10 import cifar10
from util import *
import tensorflow as tf

def load_cifar_data(path):
  d = unpickle(path)
  data = d['data']
  labels = np.array(d['labels'])
  data = data.reshape((-1, 3, 1024))
  data = np.sqrt(np.power(data[:, 0, :], 2) + np.power(data[:, 1, :], 2) + np.power(data[:, 2, :], 2))
  data = np.divide(data, 255.)

  return np.reshape(data, [-1, 32, 32, 1]), labels

def main():
  with tf.Session() as sess:
    dn = DreamNetwork2()

    savepath = "./model.ckpt"
    data, labels = load_cifar_data('cifar-10-batches-py/data_batch_1')
    test_data, test_labels = load_cifar_data('cifar-10-batches-py/test_batch')

    dn.train(data, sess) 

if __name__=='__main__':
  main()
