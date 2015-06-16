# coding: utf-8

import numpy as np
import scipy.io
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main():
  nl = [1, 16, 256, 65536]
  x = np.arange(-3, 3.05, 0.05)
  fig = plt.figure()
  for i in range(len(nl)):
    n = nl[i]
    k = int(np.sqrt(n))
    s = np.random.randn(n)
    p = np.zeros(len(x))
    for j in range(len(x)):
      r = sorted(abs(s-x[j]))
      p[j] = float(k) / (n * 2 * r[k-1])
    plt.subplot(2, 2, i+1)
    plt.plot(x, p)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.title('k = %d' % k)
  pp = PdfPages('knng_sample.pdf')
  pp.savefig(fig)
  pp.close()
  plt.clf()

if __name__ == '__main__':
  main()
