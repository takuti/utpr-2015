import numpy as np
import numpy.linalg as ln
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages

def save_plot(W01, W12, X, b, filename):
  fig = plt.figure()
  xx, yy = np.meshgrid(np.arange(-1.5, 1.6, .1), np.arange(-1.5, 1.6, .1))
  data = []
  for x, y in zip(xx.flat, yy.flat):
    data.append([1, x, y])
  data = np.asarray(data)

  # predict output for every possible points
  g10 = sigmoid(np.dot(data, W01.T)) # n x m
  g21 = sigmoid(np.dot(g10, W12.T)) # n x 1

  # plot dicision boundary
  Z = g21.T[0].reshape(xx.shape)
  plt.contourf(xx, yy, Z, 1, cmap=cm.coolwarm, alpha=.3)

  # plot original samples
  for i in range(X.shape[0]):
    if b[i] == 1: plt.plot(X[i][1], X[i][2], 'mo', mec='m', ms=4, alpha=.5)
    else: plt.plot(X[i][1], X[i][2], 'co', alpha=1., ms=4, mec='c')

  pp = PdfPages(filename)
  pp.savefig(fig)
  pp.close()
  plt.clf()

def sigmoid(u):
  return 1. / (1. + np.e ** -u)

def NeuralNetwork(src_filename, dst_filename, m):
  """Multilayer Neural Network
  input units:  3 (1, x1, x2)
  output units: 1 (0 or 1)

  :param src_filename: data source filename
  :param dst_filename: output pdf name
  :param m: number of intermediate units
  """

  learning_rate = .1
  th = 1e-1

  # read data
  data_mat = scipy.io.loadmat(src_filename)
  X = data_mat['x']
  d, n = X.shape
  X = np.vstack((np.ones(n), X)).T # augumented; n x d+1

  # read label, and convert -1, 1 labels to 0, 1 labels
  label = data_mat['l'][0]
  for i in range(n):
    if label[i] == -1: label[i] = 0

  # weight matrix from input layer (d+1=3) to intermediate layer (m)
  W01 = np.random.randn(m, d+1)

  # weight matrix from intermediate layer (m) to intermediate layer (1)
  W12 = np.random.randn(1, m)

  epoch = 0
  b = np.asarray([label]).T

  # learning
  while True:
    # compute output for n input data
    g10 = sigmoid(np.dot(X, W01.T)) # n x m
    g21 = sigmoid(np.dot(g10, W12.T)) # n x 1

    total_err = sum(abs(b - g21))
    epoch += 1
    if epoch % 1000 == 0: print 'error:', total_err

    # check convergence based on the threshold value
    if (abs(b - g21) < th).all(): break

    # epsilon from output layer to intermediate layer
    e21 = (g21 - b) * g21 * (1. - g21) # n x 1

    # epsilon from intermediate layer to input layer
    e10 = np.dot(e21, W12) * g10 * (1. - g10) # n x m

    # adjust weights
    W12 -= learning_rate * np.dot(e21.T, g10) # n x m
    W01 -= learning_rate * np.dot(e10.T, X) # m x d+1

  save_plot(W01, W12, X, b, dst_filename)

def main():
  NeuralNetwork('linear-data.mat', 'linear_nn.pdf', 3)
  NeuralNetwork('nonlinear-data.mat', 'nonlinear_nn.pdf', 5)
  NeuralNetwork('slinear-data.mat', 'slinear_nn.pdf', 5)

if __name__ == '__main__':
  main()
