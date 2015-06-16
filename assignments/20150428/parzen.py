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
  hl = [2, 4, 8, 16]
  fd = {'gaussian': lambda x, xi, hn: 1./hn*1./np.sqrt(2*np.pi)*np.exp(-((x-xi)/hn)**2/2.),
      'box': lambda x, xi, hn: 1./hn*(abs(x-xi)/hn<=0.5)}
  x = np.arange(-6, 6.05, 0.05)
  p_c1 = p_c2 = 0

  for key, f in fd.items():
    fig = plt.figure()
    plt.rcParams['font.size'] = 10
    # computing p(x|ci) for different hn
    for i in range(len(hl)):
      h = hl[i]
      hn = h / np.sqrt(n)
      p1 = np.zeros(len(x))
      p2 = np.zeros(len(x))
      for sample1, sample2 in zip(x1, x2):
        p1 = p1 + f(x, sample1, hn)
        p2 = p2 + f(x, sample2, hn)
      p1 = p1 / n
      p2 = p2 / n
      if key == 'gaussian' and h == 8: # copy for the next task (posterior probabilities)
        p_c1 = p1.copy()
        p_c2 = p2.copy()
      plt.subplot(2, 2, i+1)
      plt.plot(x, p1, label=r'$p(\mathbf{x} \mid c_1)$')
      plt.plot(x, p2, label=r'$p(\mathbf{x} \mid c_2)$')
      plt.legend(framealpha=0, fontsize=7)
      plt.title(r'$h_n = %.3f$' % hn)
      plt.xlabel(r'$x$')
      plt.ylabel(r'$p(\mathbf{x} \mid c_i)$')
    plt.tight_layout()
    pp = PdfPages('parzen_%s.pdf' % key)
    pp.savefig(fig)
    pp.close()
    plt.clf()

  # compute posterior probabilities
  Px = sum(p_c1) * .5 + sum(p_c2) * .5
  P1 = p_c1 * .5 / Px
  P2 = p_c2 * .5 / Px
  fig = plt.figure()
  plt.plot(x, P1, label=r'$P(c_1 \mid \mathbf{x})$')
  plt.plot(x, P2, label=r'$P(c_2 \mid \mathbf{x})$')
  plt.ylabel(r'$P(c_i \mid \mathbf{x})$')
  plt.legend()
  pp = PdfPages('parzen_posterior.pdf')
  pp.savefig(fig)
  pp.close()
  plt.clf()

if __name__ == '__main__':
  main()
