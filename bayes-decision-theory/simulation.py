# coding: utf-8

import random

def main():
  # produced rates
  P_M1 = .2
  P_M2 = .3
  P_M3 = .5
  # defective rates
  P_M1x = .01
  P_M2x = .02
  P_M3x = .03

  N = 100000
  cnt_total = cnt_M1x = cnt_M2x = cnt_M3x = 0

  for i in range(N):
    r_produced = random.random()
    r_defective = random.random()
    if r_produced < P_M1: # produced by M1
      if r_defective < P_M1x:
        cnt_total += 1
        cnt_M1x += 1
    elif r_produced < P_M1 + P_M2: # produced by M2
      if r_defective < P_M2x:
        cnt_total += 1
        cnt_M2x += 1
    else: # produced by M3
      if r_defective < P_M3x:
        cnt_total += 1
        cnt_M3x += 1
  print 'P(M1|x) =', float(cnt_M1x) / cnt_total
  print 'P(M2|x) =', float(cnt_M2x) / cnt_total
  print 'P(M3|x) =', float(cnt_M3x) / cnt_total

if __name__ == '__main__':
  main()
