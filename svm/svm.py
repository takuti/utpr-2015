import numpy as np
from numpy.linalg import norm
import cvxopt
import cvxopt.solvers

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages

N = 100
C = 0.5
sigma = 0.3

def plot_save(l, X, alpha, b, filename):
  fig = plt.figure()

  xx, yy = np.meshgrid(np.arange(-1.5, 1.6, .1), np.arange(-1.5, 1.6, .1))
  zz = []
  for x, y in zip(xx.flat, yy.flat):
    zz.append(f(np.array([x, y]), alpha, l, X, b))
  Z = np.array(zz).reshape(xx.shape)

  plt.contourf(xx, yy, Z, 1, cmap=cm.coolwarm, alpha=.3)

  # plot original samples
  for i in range(X.shape[0]):
    if l[i] == 1: plt.plot(X[i][0], X[i][1], 'mo', mec='m', ms=4, alpha=.5)
    else: plt.plot(X[i][0], X[i][1], 'co', alpha=1., ms=4, mec='c')

  plt.xlim(-1, 1)
  plt.ylim(-1, 1)

  pp = PdfPages(filename)
  pp.savefig(fig)
  pp.close()
  plt.clf()

def kernel(x, y):
  return np.exp(-norm(x-y)**2 / (2 * (sigma ** 2)))

def f(x, alpha, l, X, b):
  total = 0
  for n in range(N):
    total += alpha[n] * l[n] * kernel(x, X[n])
  return total + b

def generate_data(d, n):
  # return d x n matrix
  # each element will be in [0, 1]
  # label is -1 or 1
  x = np.array([np.random.random(n) for i in range(d)])
  x = 2 * x - np.ones((d, n))
  l = 2 * ((2 * x[0, :] + x[1, :]) > .5) - 1
  flip = abs((2 * x[0, :] + x[1, :]) - 0.5) < 0.2
  for i in range(n):
    if flip[i]: l[i] = -l[i]
  return l, x

if __name__ == '__main__':

  # generate data
  l, X = generate_data(2, N)
  X = X.T

  ### Solve Quadratic Programming

  # create Q
  k = np.zeros((N, N))
  for i in range(N):
    for j in range(N):
      k[i, j] = l[i] * l[j] * kernel(X[i], X[j])
  Q = cvxopt.matrix(k)

  # create p
  p = cvxopt.matrix(-np.ones(N))

  # notations are different from handout
  G = cvxopt.matrix(np.vstack((np.diag([-1]*N), np.identity(N))))
  h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N) * C)))
  qp = cvxopt.solvers.qp(Q, p, G, h)
  alpha = np.array(qp['x']).reshape(N) # flatten by reshape

  # get support vectors
  isv1 = []
  isv2 = []
  for i in range(len(alpha)):
    if 0 < alpha[i]: isv1.append(i)
    if 0 < alpha[i] < C: isv2.append(i)

  # compute b
  total = 0
  for i in isv1:
    tmp = 0
    for j in isv2:
      tmp += alpha[j] * l[j] * kernel(X[i], X[j])
    total += l[i] - tmp
  b = total / len(isv2)

  plot_save(l, X, alpha, b, 'svm.pdf')
