# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main():
  # generate random samples
  n = 1000 # 1000 samples
  d = 2 # 2D

  x1 = np.random.randn(n, d)
  s = np.array([[1, 0], [0, 2]]) # [stretch] x-axis: *1, y-axis: *2
  r = np.array([[np.cos(np.pi/3), np.sin(np.pi/3)], \
                [-1*np.sin(np.pi/3), np.cos(np.pi/3)]]) # [rotate] -PI/3
  t = np.array([[-2, 2]]) # [move] x-2, y+2
  x1 = np.dot(np.dot(x1, s), r) + np.dot(np.ones((n, 1)), t)

  x2 = np.random.randn(n, d)
  s = np.array([[2, 0], [0, 1]]) # [stretch] x-axis: *2, y-axis: *1
  r = np.array([[np.cos(np.pi/3), np.sin(np.pi/3)], \
                [-1*np.sin(np.pi/3), np.cos(np.pi/3)]]) # [rotate] -PI/3
  t = np.array([[2.5, -1]]) # [move] x+2.5, y-1
  x2 = np.dot(np.dot(x2, s), r) + np.dot(np.ones((n, 1)), t)

  X, Y = np.meshgrid(np.arange(-7.5, 7.5, .5), np.arange(-7.5, 7.5, .5))
  flatten_meshgrid = np.zeros((np.prod(X.shape), 2))
  i = 0
  for vx, vy in zip(X.flat, Y.flat):
    flatten_meshgrid[i,0], flatten_meshgrid[i,1] = vx, vy
    i += 1

  # plot each normal distribution and generated points
  fig = plt.figure()
  covs = []
  for x, c in [(x1, 'c'), (x2, 'm')]:
    # compute mean of each variable; m = [[mean v1, mean v2]]
    m = np.array([np.mean(x, 0)])

    # get covariance matrix and their eigen[values|vectors]
    cov = np.dot(np.transpose(x - np.dot(np.ones((n, 1)), m)), \
        x - np.dot(np.ones((n, 1)), m)) / float(n)
    covs.append(cov)

    # plot based on the covariance matrix of the obserbed samples
    # = plotting emmstimated normal distribution
    tmp = flatten_meshgrid - np.dot(np.ones((np.prod(X.shape), 1)), m) # Z = X - M
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    p = 1 / (2 * np.pi * np.sqrt(det)) \
        * np.exp(-1/2 * np.diag(np.dot(np.dot(tmp, inv), np.transpose(tmp))))

    plt.scatter(x[:,0], x[:,1], color=c, alpha=0.2)
    plt.contour(X, Y, np.reshape(p, np.shape(X)), 10)
  plt.axis([-10, 10, -10, 10])
  pp = PdfPages('original.pdf')
  pp.savefig(fig)
  pp.close()
  plt.clf()

  ### Simultaneous diagonalization
  # find A (eigenvector matrix of cov1^(-1)*cov2)
  ed, A = np.linalg.eig(np.dot(np.linalg.inv(covs[0]), covs[1]))

  # compute diagonalized points
  y1 = np.dot(x1, A)
  y2 = np.dot(x2, A)

  # show each diagonalized normal distribution
  fig = plt.figure()
  for n, (y, c) in enumerate([(y1, 'c'), (y2, 'm')]):
    m = np.array([np.mean(y, 0)])
    tmp = flatten_meshgrid - np.dot(np.ones((np.prod(X.shape), 1)), m)

    diagonalized = np.dot(np.dot(np.transpose(A), covs[n]), A)
    det = np.linalg.det(diagonalized)
    inv = np.linalg.inv(diagonalized)
    p = 1 / (2 * np.pi * np.sqrt(det)) \
        * np.exp(-1/2 * np.diag(np.dot(np.dot(tmp, inv), np.transpose(tmp))))

    plt.scatter(y[:,0], y[:,1], color=c, alpha=0.2)
    plt.contour(X, Y, np.reshape(p, np.shape(X)), 10)
  plt.axis([-10, 10, -10, 10])
  pp = PdfPages('diagonalized.pdf')
  pp.savefig(fig)
  pp.close()
  plt.clf()

if __name__ == '__main__':
  main()
