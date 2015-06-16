# coding: utf-8

import numpy as np
import scipy.io
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main():
  nl = [1, 10, 100, 100000]
  hl = [1, .5, .1]
  x = np.arange(-3, 3.05, 0.05)
  f = lambda x, xi, hn: 1./hn*1./np.sqrt(2*np.pi)*np.exp(-((x-xi)/hn)**2/2.)
  f = lambda x, xi, hn: 1./hn*(abs(x-xi)/hn<=0.5)
  fig = plt.figure()
  for i in range(len(hl)):
    h = hl[i]
    for j in range(len(nl)):
      n = nl[j]
      hn = h / np.sqrt(n)
      p = np.zeros(len(x))
      for xi in np.random.randn(n):
        p = p + f(x, xi, hn)
      p = p / float(n)
      plt.subplot(len(nl), len(hl), (i+1)+j*len(hl))
      plt.plot(x, p)
      plt.xticks(fontsize=7)
      plt.yticks(fontsize=7)
  pp = PdfPages('parzeng_sample.pdf')
  pp.savefig(fig)
  pp.close()
  plt.clf()

if __name__ == '__main__':
  main()
