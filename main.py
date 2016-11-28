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

def main(_):
  dn = DreamNetwork()

  data, labels = load_cifar_data('cifar-10-batches-py/data_batch_1')
  dn.train(data, labels)

  train_data, train_labels = load_cifar_data('cifar-10-batches-py/test_batch')
  dream_img = dn.dream(train_data[0, :, :], train_labels[0]) 

  scipy.misc.imsave(
    'dream.jpg', np.squeeze(dream_img).repeat(2, axis=0).repeat(2, axis=1)
  )


if __name__ == '__main__':
  main()
