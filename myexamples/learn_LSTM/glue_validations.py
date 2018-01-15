import sys
import os
import glob

if __name__ == '__main__':
    indir = sys.argv[1]

    names = glob.glob(indir+'/*_epochs')
    names = sorted(names,key=lambda x:int(x.split('/')[1].split('_')[0]))
    # print names
    files = ' '.join([j+'/validation.png' for j in names])
    CMD = 'convert -loop 0 -delay 50 {infiles:s} {indir:s}/validation.gif'.format(infiles=files,indir=indir)
    # print CMD
    # exit()
    os.system(CMD)
