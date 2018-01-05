#!/bin/env python

import pylab as plt
import numpy as np
import sys

if __name__ == '__main__':
    fin = sys.argv[1]
    npz = np.load(open(fin,'r'))
    dir = '/'.join(fin.split('/')[:-1])
    hp = fin.split('/')[-2]
    try:
        epochs = fin.split('/')[-3]+' '
    except IndexError:
        epochs = '? '
    real = npz['real']
    pred = npz['pred']
    plt.figure(figsize=(7/1.5,3/1.5))
    plt.plot(real)
    plt.plot(pred,'r')
    plt.ylim(0,1000)

    plt.title(epochs+hp.replace('_',' '))

    plt.savefig(dir+'/validation.png',bbox_inches='tight',dpi=96)
