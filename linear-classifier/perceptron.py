import sys
import numpy as np
import numpy.linalg as ln
import scipy.io
from save_plot import *

def batch(w, X, d):
  learning_rate = .01
  th = 1e-10
  ths = th * np.ones(d+1)
  while True:
    grad = learning_rate * np.asarray(sum([x for x in X.T if np.dot(w, x.T) < 0]))
    if type(grad) == int: break # grad == 0
    elif (abs(grad) < ths).all(): break
    w = w + grad
  return w

def online(w, X, d):
  w = np.random.randn(1+d)
  k = 0
  while True:
    k += 1
    misclassified_list = [x for x in X.T if np.dot(w, x.T) < 0]
    l = len(misclassified_list)
    if l == 0: break
    i = k % l
    w = w + misclassified_list[i]
  return w

def perceptron(src_filename, dst_filename, method='batch'):
  # read data
  data_mat = scipy.io.loadmat(src_filename)
  X = data_mat['x']
  d, n = X.shape
  X = np.vstack((np.ones(n), X)) # augumented
  label = data_mat['l'][0]

  # normalization
  X_ = np.zeros((d+1, n))
  for i in range(n):
    if label[i] == -1: X_[:,i] = -X[:,i]
    else: X_[:,i] = X[:,i]

  # learning
  w = np.random.randn(1+d) # initial w
  if method == 'batch': w = batch(w, X_, d)
  elif method == 'online': w = online(w, X_, d)
  else: sys.exit('Error: you can only choose `batch` or `online` as a method')

  save_plot(w, X, label, dst_filename)

def main():
  perceptron('linear-data.mat', 'linear_batch.pdf', 'batch')
  perceptron('linear-data.mat', 'linear_online.pdf', 'online')
  perceptron('slinear-data.mat', 'slinear_batch.pdf', 'batch')
  perceptron('slinear-data.mat', 'slinear_online.pdf', 'online')
  #perceptron('nonlinear-data.mat', 'nonlinear_batch.pdf', 'batch')

if __name__ == '__main__':
  main()
