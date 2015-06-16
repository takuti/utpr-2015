# coding: utf-8

import numpy as np
import numpy.linalg as ln
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

def norm(x):
  return np.sum(np.abs(x)**2,axis=-1)**(1./2)

def main():
  data = scipy.io.loadmat('data.mat')
  samples = np.array([data['x1'][0], data['x2'][0]]).T
  n = len(samples)
  hl = [2, 4, 8, 16]
  f = lambda x, xi, hn: 1./hn*1./np.sqrt(2*np.pi)*np.exp(-norm(x-xi)/hn**2/2.)

  x = np.arange(-6, 6.05, 0.05)
  y = np.arange(-6, 6.05, 0.05)
  X, Y = np.meshgrid(x, y)
  flatten_meshgrid = np.zeros((np.prod(X.shape), 2))
  i = 0
  for vx, vy in zip(X.flat, Y.flat):
    flatten_meshgrid[i,0], flatten_meshgrid[i,1] = vx, vy
    i += 1

  fig = plt.figure()
  plt.rcParams['font.size'] = 7
  # computing p(x) for different hn
  for i in range(len(hl)):
    h = hl[i]
    hn = h / np.sqrt(n)
    # compute probabilities at each mesh-point
    p = np.zeros(flatten_meshgrid.shape[0])
    for sample in samples:
      p = p + f(flatten_meshgrid, sample, hn)
    p = p / n
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    ax.set_title(r'$h_n = %.3f$' % (hn))
    ax.set_xlabel(r'$x_1$', fontsize=10)
    ax.set_ylabel(r'$x_2$', fontsize=10)
    ax.set_zlabel(r'$p(\mathbf{x})$', fontsize=10)
    ax.plot_surface(X, Y, (p*.5).reshape(X.shape), \
        rstride=3, cstride=3, cmap = cm.coolwarm, linewidth=0, antialiased=False)
  plt.tight_layout()
  pp = PdfPages('posterior.pdf')
  pp.savefig(fig)
  pp.close()
  plt.clf()

if __name__ == '__main__':
  main()
