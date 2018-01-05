import sys
import os
import glob

if __name__ == '__main__':
    try:
        os.makedirs('gifs')
    except OSError:
        pass

    names = [j.split('/')[1] for j in glob.glob('3000_epochs/*')]
    CMD = 'convert -loop 0 -delay 75 *_epochs/{name:s}/validation.png gifs/{name:s}.gif'
    for name in names:

        go = CMD.format(name=name)
        print go
        os.system(go)
        
