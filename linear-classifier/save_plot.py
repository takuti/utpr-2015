import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def save_plot(w, X, label, filename):
  fig = plt.figure()
  plt.ylim(-1., 1.)

  # plot discriminant function
  X1 = np.linspace(-1., 1.)
  X2 = -(w[0] / w[2]) - (w[1] / w[2] * X1)
  plt.plot(X1, X2)

  for i in range(X.shape[1]):
    if label[i] == 1: plt.plot(X[1][i], X[2][i], 'mo', mec='m', ms=4, alpha=.5)
    else: plt.plot(X[1][i], X[2][i], 'co', alpha=1., ms=4, mec='c')

  pp = PdfPages(filename)
  pp.savefig(fig)
  pp.close()
  plt.clf()

