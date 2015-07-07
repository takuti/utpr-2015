# coding: utf-8

import numpy as np
import numpy.linalg as ln
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main():
  data = scipy.io.loadmat('data.mat')
  x1 = data['x1'][0]
  x2 = data['x2'][0]
  n = len(x1)
  kl = [1, 7, 14, 28] # k = 14 may be an optimal
  x = np.arange(-6, 6.05, 0.05)

  fig = plt.figure()
  plt.rcParams['font.size'] = 10
  for i in range(len(kl)):
    k = kl[i]
    p1 = np.zeros(len(x))
    p2 = np.zeros(len(x))
    for j in range(len(x)):
      r1 = sorted(abs(x1 - x[j]))
      r2 = sorted(abs(x2 - x[j]))
      p1[j] = float(k) / (n * 2 * r1[k-1])
      p2[j] = float(k) / (n * 2 * r2[k-1])
    plt.subplot(2, 2, i+1)
    plt.plot(x, p1, label=r'$p(\mathbf{x} \mid c_1)$')
    plt.plot(x, p2, label=r'$p(\mathbf{x} \mid c_2)$')
    plt.legend(framealpha=0, fontsize=7)
    plt.title(r'$k = %d$' % k)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$p(\mathbf{x} \mid c_i)$')
  plt.tight_layout()
  pp = PdfPages('knn.pdf')
  pp.savefig(fig)
  pp.close()
  plt.clf()

if __name__ == '__main__':
  main()
