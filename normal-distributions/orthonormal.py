# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

def main():
  # generate random samples
  n = 1000 # 1000 samples
  d = 2 # 2D
  x = np.random.randn(n, d)

  # adjust position
  s = np.array([[2, 0], [0, 1]]) # [stretch] x-axis: *1, y-axis: *2
  r = np.array([[np.cos(np.pi/3), np.sin(np.pi/3)], [-1*np.sin(np.pi/3), np.cos(np.pi/3)]]) # [rotate] PI/3
  t = np.array([[2.5, -1]]) # [move] x+0,5, y-1
  x = np.dot(np.dot(x, s), r) + np.dot(np.ones((n, 1)), t)

  # compute mean of each variable; m = [[mean x, mean y]]
  m = np.array([np.mean(x, 0)])

  # get covariance matrix and their eigen[values|vectors]
  cov = np.dot(np.transpose(x - np.dot(np.ones((n, 1)), m)), x - np.dot(np.ones((n, 1)), m)) / float(n)
  ed, ev = np.linalg.eig(cov) # ed: eigenvalues, ev: eigenvector matrix
  print ed
  print ev

  # plot based on the covariance matrix of the samples
  # = plotting emmstimated normal distribution
  X, Y = np.meshgrid(np.arange(-5, 5.5, .5), np.arange(-5, 5.5, .5))
  tmp = np.zeros((np.prod(X.shape), 2))
  i = 0
  for vx, vy in zip(X.flat, Y.flat):
    tmp[i,0], tmp[i,1] = vx, vy
    i += 1
  tmp = tmp - np.dot(np.ones((np.prod(X.shape), 1)), m) # Z = X - M
  det = np.linalg.det(cov)
  inv = np.linalg.inv(cov)
  p = 1 / (2 * np.pi * np.sqrt(det)) *np.exp(-1/2*np.diag(np.dot(np.dot(tmp, inv), np.transpose(tmp))))

  plt.scatter(x[:,0], x[:,1], color='b', alpha=0.3)
  plt.contour(X, Y, np.reshape(p, np.shape(X)), 10)
  plt.show()

  print '---'
  print ed
  print ev

  ed, ev = np.linalg.eig(cov) # ed: eigenvalues, ev: eigenvector matrix
  ed = np.diag(ed) # create diagonal matrix which has eigenvalues

  # orthonormal transformation
  x1 = np.dot(x, ev)
  m1 = np.array([np.mean(x1, 0)])
  tmp = np.zeros((np.prod(X.shape), 2))
  i = 0
  for vx, vy in zip(X.flat, Y.flat):
    tmp[i,0], tmp[i,1] = vx, vy
    i += 1
  tmp = tmp - np.dot(np.ones((np.prod(X.shape), 1)), m1) # Z = X - M
  det = np.linalg.det(np.dot(np.dot(np.transpose(ev), cov), ev))
  inv = np.linalg.inv(np.dot(np.dot(np.transpose(ev), cov), ev))
  p = 1 / (2 * np.pi * np.sqrt(det)) *np.exp(-1/2*np.diag(np.dot(np.dot(tmp, inv), np.transpose(tmp))))

  plt.scatter(x1[:,0], x1[:,1], color='b', alpha=0.3)
  plt.contour(X, Y, np.reshape(p, np.shape(X)), 10)
  plt.show()

if __name__ == '__main__':
  main()
