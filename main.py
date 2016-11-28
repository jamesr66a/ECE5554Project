from DreamNetwork import DreamNetwork
import numpy as np
import scipy.misc
from tensorflow.models.image.cifar10 import cifar10
from util import *

def load_cifar_data(path):
  d = unpickle(path)
  data = d['data']
  labels = np.array(d['labels'])
  data = np.linalg.norm(data.reshape((-1, 3, 1024)), axis=1)

  return data, labels 

def main():
  dn = DreamNetwork()

  data, labels = load_cifar_data('cifar-10-batches-py/data_batch_1')
  dn.train(data, labels, training_iters=1000000)

  #test_data, test_labels = load_cifar_data('cifar-10-batches-py/test_batch')
  test_data = np.zeros((1, 1024))
  test_labels = [0]

  dream_img = dn.dream(
    np.expand_dims(test_data[0, :], 0), np.expand_dims(test_labels[0], 0)
  ) 

  scipy.misc.imsave(
    'dream.jpg', np.squeeze(dream_img).repeat(2, axis=0).repeat(2, axis=1)
  )


if __name__ == '__main__':
  main()
