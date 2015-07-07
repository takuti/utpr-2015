# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

def main():
  m = 1000 # Number of trials for each size
  sample_sizes = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
  biased = np.zeros(len(sample_sizes))
  unbiased = np.zeros(len(sample_sizes))
  for i in range(len(sample_sizes)):
    n = sample_sizes[i]
    biased_trials = np.zeros(m)
    unbiased_trials = np.zeros(m)
    for j in range(m):
      r = np.random.rand(n)
      r = r - np.mean(r)
      r = r * r
      biased_trials[j] = np.sum(r) / float(n)
      unbiased_trials[j] = np.sum(r) / float(n-1)
    biased[i] = np.mean(biased_trials)
    unbiased[i] = np.mean(unbiased_trials)
  # plotting the results
  exact = 1./12
  plt.plot(sample_sizes, np.array([exact] * len(sample_sizes)))
  plt.plot(sample_sizes, biased, label='biased')
  plt.plot(sample_sizes, unbiased, label='unbiased')
  plt.xscale('log')
  plt.legend(loc='lower right')
  plt.xlabel("sample size")
  plt.ylabel("estimated variance")
  plt.show()

if __name__ == '__main__':
  main()
