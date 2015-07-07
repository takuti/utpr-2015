import numpy as np
import numpy.linalg as ln
import scipy.io
from save_plot import *

def mse(src_filename, dst_filename):
  # read data
  data_mat = scipy.io.loadmat(src_filename)
  X = data_mat['x']
  d, n = X.shape
  X = np.vstack((np.ones(n), X)).T # augumented
  b = data_mat['l'][0].T

  # learning
  w = np.dot(np.dot(ln.inv(np.dot(X.T, X)), X.T), b)

  save_plot(w, X.T, b.T, dst_filename)

def main():
  mse('linear-data.mat', 'linear_mse.pdf')
  mse('nonlinear-data.mat', 'nonlinear_mse.pdf')
  mse('slinear-data.mat', 'slinear_mse.pdf')

if __name__ == '__main__':
  main()
