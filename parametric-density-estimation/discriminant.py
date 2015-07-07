# coding: utf-8

import numpy as np
import numpy.linalg as ln
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def case(cov1, cov2):
  """ detect the case for a pair of given 2 different covariance matrices
  """
  if np.array_equal(cov1, cov2):
    n = cov1.shape[0]
    if np.array_equal(cov1 / float(cov1[0, 0]), np.identity(n)): return 1
    return 2
  return 3

def discriminated_plot(x1, l1, x2, l2):
  n = x1.shape[0]
  # correct x1
  x = np.array([x1[i] for i in range(n) if l1[i]])
  if x.shape[0] != 0: plt.plot(x[:,0], x[:,1], 'co', alpha=.3, ms=3, mec='c')
  # wrong x1
  x = np.array([x1[i] for i in range(n) if not l1[i]])
  if x.shape[0] != 0: plt.plot(x[:,0], x[:,1], 'mo', mec='m', ms=4, alpha=.5)
  # correct x2
  x = np.array([x2[i] for i in range(n) if not l2[i]])
  if x.shape[0] != 0: plt.plot(x[:,0], x[:,1], 'm^', alpha=.2, ms=4, mec='m')
  # wrong x2
  x = np.array([x2[i] for i in range(n) if l2[i]])
  if x.shape[0] != 0: plt.plot(x[:,0], x[:,1], 'c^', mec='g', ms=5)


def main():
  X, Y = np.meshgrid(np.arange(-7.5, 7.5, .5), np.arange(-7.5, 7.5, .5))
  flatten_meshgrid = np.zeros((np.prod(X.shape), 2))
  i = 0
  for vx, vy in zip(X.flat, Y.flat):
    flatten_meshgrid[i,0], flatten_meshgrid[i,1] = vx, vy
    i += 1

  # set parameters of 2 different normal distributions
  m1 = np.array([[0, 2]])
  m2 = np.array([[3, 0]])
  cov_pairs = []
  cov_pairs.append((np.identity(2), np.identity(2)))
  cov_pairs.append((np.array([[2, 0], [0, 3]]), np.array([[2, 0], [0, 3]])))
  cov_pairs.append((np.array([[2, 0], [0, 3]]), np.array([[6, 0], [0, 4]])))

  # generate random samples
  n = 1000 # 1000 samples
  d = 2 # 2D

  for cov1, cov2 in cov_pairs:
    x1 = np.random.randn(n, d)
    x2 = np.random.randn(n, d)

    # move generated points to the center of each distribution
    # and stretch to fit given diagonal covariance matrix
    x1 = np.dot(x1, np.sqrt(cov1)) + (m1 * np.ones((n, 1)))
    x2 = np.dot(x2, np.sqrt(cov2)) + (m2 * np.ones((n, 1)))

    for prior_idx, prior in enumerate([.5, .9]):
      # show each normal distribution as contour plots
      fig = plt.figure()
      for x, m, cov in [(x1, m1, cov1), (x2, m2, cov2)]:
        tmp = flatten_meshgrid - np.dot(np.ones((np.prod(X.shape), 1)), m) # Z = X - M
        p = 1 / (2 * np.pi * np.sqrt(ln.det(cov))) \
            * np.exp(-1/2 * np.diag(np.dot(np.dot(tmp, ln.inv(cov)), np.transpose(tmp))))
        plt.contour(X, Y, np.reshape(p, np.shape(X)), 10)

      # Compute discriminant functions for each data set
      p1 = prior
      p2 = 1 - p1
      c = case(cov1, cov2)
      if c == 1:
        w = m1 - m2
        x0 = 1/2. * (m1+m2) - 1. / (ln.norm(m1-m2)**2) * np.log(p1/p2) * (m1-m2)
        l1 = np.dot(x1 - x0 * np.ones((n,1)), w.T) > 0
        l2 = np.dot(x2 - x0 * np.ones((n,1)), w.T) > 0
      elif c == 2:
        inv = ln.inv(cov1)
        w = np.dot(m1-m2, inv)
        x0 = 1/2. * (m1+m2) - 1. / np.dot(np.dot(m1-m2, inv), (m1-m2).T) * np.log(p1/p2) * (m1-m2)
        l1 = np.dot(x1 - x0 * np.ones((n,1)), w.T) > 0
        l2 = np.dot(x2 - x0 * np.ones((n,1)), w.T) > 0
      else:
        inv1 = ln.inv(cov1)
        inv2 = ln.inv(cov2)
        W1 = -1/2. * inv1
        W2 = -1/2. * inv2
        w1 = np.dot(inv1, m1.T)
        w2 = np.dot(inv2, m2.T)
        w10 = np.dot(np.dot(m1, W1), m1.T) - 1/2. * np.log(ln.det(inv1)) + np.log(p1)
        w20 = np.dot(np.dot(m2, W2), m2.T) - 1/2. * np.log(ln.det(inv2)) + np.log(p2)
        g1 = np.array([[np.dot(np.dot(x, W1), x.T) for x in x1]]).T + np.dot(x1, w1) + w10 * np.ones((n,1))
        g2 = np.array([[np.dot(np.dot(x, W2), x.T) for x in x2]]).T + np.dot(x2, w2) + w20 * np.ones((n,1))
        l1 = g1 > 0
        l2 = g2 > 0

      discriminated_plot(x1, l1, x2, l2)

      c_patch = mpatches.Patch(color='c', label='$P(\omega_1) = %.1f$' % p1)
      m_patch = mpatches.Patch(color='m', label='$P(\omega_2) = %.1f$' % p2)
      plt.legend(handles=[c_patch, m_patch])
      plt.axis([-4, 6.5, -4, 6.5])
      pp = PdfPages('case%d_%d.pdf' % (c, prior_idx+1))
      pp.savefig(fig)
      pp.close()
      plt.clf()

if __name__ == '__main__':
  main()
