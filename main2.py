from DreamNetwork2 import DreamNetwork2
import numpy as np
import scipy.misc
from tensorflow.models.image.cifar10 import cifar10
from util import *
import tensorflow as tf
from skimage.color import rgb2gray

def load_cifar_data(path):
  d = unpickle(path)
  data = d['data']
  labels = np.array(d['labels'])
  data = np.divide(data, 255.)
  data = data.reshape((-1, 3, 1024))
  data = np.transpose(data, [0, 2, 1])
  data = np.reshape(data, [-1, 32, 32, 3])
  data = rgb2gray(data)

  return np.reshape(data, [-1, 32, 32, 1]), labels

def main():
  with tf.Session() as sess:
    dn = DreamNetwork2()

    savepath = "./model.ckpt"
    data, labels = load_cifar_data('cifar-10-batches-py/data_batch_1')
    test_data, test_labels = load_cifar_data('cifar-10-batches-py/test_batch')

    dn.train(data, sess)

    dream = dn.dream(np.expand_dims(test_data[0, :, :, :], 0), sess)

    scipy.misc.imsave(
      'dream.bmp', np.squeeze(dream).repeat(8, axis=0).repeat(8, axis=1)
    )

if __name__=='__main__':
  main()
