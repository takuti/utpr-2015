# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

def main():
  cov = np.array([[1, 0],[0, 3]])
  r = np.array([[np.cos(np.pi/3), -1*np.sin(np.pi/3)], [np.sin(np.pi/3), np.cos(np.pi/3)]])
  r = np.array([[1,0],[0,1]])
  X, Y = np.meshgrid(np.arange(-5, 5.5, .25), np.arange(-5, 5.5, .25))
  XY = np.zeros((np.prod(np.shape(X)), 2))
  i = 0
  for vx, vy in zip(X.flat, Y.flat):
    XY[i,0], XY[i,1] = vx, vy
    i += 1
  det = np.linalg.det(cov)
  #inv = np.linalg.inv(cov)
  inv = np.linalg.inv(np.dot(np.dot(np.transpose(r), cov), r))
  p = 1 / (2 * np.pi * np.sqrt(det)) *np.exp(-1/2*np.diag(np.dot(np.dot(XY, inv), np.transpose(XY))))
  plt.contour(X, Y, np.reshape(p, np.shape(X)))
  plt.show()

if __name__ == '__main__':
  main()
