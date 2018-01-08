#!/bin/env python2.7
import numpy as np
from torch.autograd import Variable
from torch import nn
import torch
from sklearn.preprocessing import RobustScaler
from tqdm import trange
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)
import os
# load data file
time_series = np.load(open('data/time_series_traffic_paris.npy','r'))
h = np.arange(time_series.shape[0],dtype=np.float32)%24
ch = np.cos(h/24.)
sh = np.sin(h/24.)
time_series = np.c_[h,ch,sh,time_series]
print time_series.shape
# exit()
# scale data
scaler = RobustScaler()
Q = scaler.fit_transform(time_series)

# Hyper Parameters
WEIGHT_DECAY = 1e-5
NESTEROV = True
MOMENTUM = 0.9
BATCH_SIZE = 600
LR = 1/10.
# Parameters
FEATURES = Q.shape[-1]

# functions
# assumes data fits in memory
def batch_gen(data,window_size,batch_size=BATCH_SIZE,shuffle=True):
    #~ win creat
    ws = window_size
    windows = [[data[i+n] for i in range(0, len(data)-1-ws, ws)] for n in range(ws)]
    windows = np.asarray(windows).transpose(1,0,-1)
    # print windows.shape
    #~
    if shuffle:
        index = np.random.permutation(windows.shape[0])
        windows = windows[index,:]
    #~
    windows = Variable(torch.from_numpy(windows.astype(np.float32)))
    for idx in range(0,windows.size(0),batch_size):
        outin = windows[idx:idx+batch_size]
        x = outin[:,:-1]
        y = outin[:,-1]
        yield x,y
## validation
np2y= lambda x: Variable(torch.from_numpy(x.astype(np.float32))).unsqueeze(0)
def validate(Q,model,W):
    real = []
    pred = []
    model.eval()
    W = W - 1
    for i in xrange(W,min(Q.shape[0]-1,200)): # min to limit time spent on this
        t0 = Q[i-W:i]
        t1 = Q[i]
        t1_pred = model(np2y(t0)).squeeze().data.numpy()

        real.append(scaler.inverse_transform(t1.reshape(1,-1)).mean())
        pred.append(scaler.inverse_transform(t1_pred.reshape(1,-1)).mean())

    real = np.asarray(real)
    pred = np.asarray(pred)

    MSEscore = np.sum((real-pred)**2)**.5

    return MSEscore,{'real':real,'pred':pred}

## Training
def train_model(batches_gen,data,model,window_size,epochs=1000,train=True,ev=1):
    if train:
        model.train()
    else:
        model.eval()

    log = '' # log output as a string
    for epoch in trange(epochs,leave=False):
        batches = batches_gen(data,window_size=window_size)
        running_loss = 0.0

        for x,y in batches:
            out = model(x)
            loss = model.loss(out,y)
            # print out.size(),y.size(),x.size()
            model.zero_grad()
            loss.backward()
            model.optimizer.step()

            running_loss += loss.data[0]

        if epoch%ev==0 or epoch == epochs-1:

            epoch_loss = running_loss
            log+=('{0:d} {1:e} {2:e}\n'.format(epoch,epoch_loss,validate(Q,model,window_size)[0]))
    return log

# model(s)
class RNN(nn.Module):
    def __init__(self, features, hidden_dim =256, lr = 0.001,weight_decay = WEIGHT_DECAY):
        """
            this class will have to contain also:
            1. the scaler used for training (eg. RobustScaler)
            2. the parameters used for trainig (eg window)
            3. the input generation preparation (eg. np2var)
        """
        super(RNN, self).__init__()

        # self._inout_dim = features
        self._hidden_dim = hidden_dim
        self.lin_in = nn.Linear(features,hidden_dim)
        self.cell = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.lin_out = nn.Linear(hidden_dim,features)

        # functions
        self.__lr = lr
        self.__weight_decay = weight_decay
        self.loss = nn.MSELoss()
        self.optimizer = self.__init_optimizer()
#         self.activation = torch.nn.Activation('linear')

    def forward(self, c):
        hidden = (Variable(torch.zeros(1,1,self._hidden_dim)),
                  Variable(torch.zeros(1,1,self._hidden_dim)))

        output = self.lin_in(c)
        output,hidden = self.cell(output,hidden)
        output = self.lin_out(output)
        # print output.size()
        # output = output.max(dim=1)[0]
        return output

    @property
    def lr(self):
        return self.__lr

    @lr.setter
    def lr(self,new_lr):
        self.__lr = new_lr
        self.optimizer = self.__init_optimizer()

    @property
    def weight_decay(self):
        return self.__weight_decay

    @weight_decay.setter
    def weight_decay(self,new_wd):
        self.__weight_decay=new_wd
        self.optimizer = self.__init_optimizer()

    def __init_optimizer(self):
        return torch.optim.SGD(self.parameters(),
                               lr = self.__lr,
                               momentum=MOMENTUM,
                               weight_decay=self.__weight_decay,
                               nesterov=NESTEROV)

# main
def main(WINDOW,HIDDEN_DIM,EPOCHS,WEIGHT_DECAY,EV=1000):
    # os stuff
    DIR = '{}_{}_{}/'.format(WINDOW,HIDDEN_DIM,WEIGHT_DECAY)
    print DIR

    already_done_epochs = 0
    try:
        os.makedirs(DIR)
    except OSError:
        # logging.warninging("overwriting previous files in folder "+DIR)
        try:
            already_done_epochs = int(open(DIR+'/epoch').read())
            logging.warning('found {0:d} epochs done'.format(already_done_epochs))
        except IOError:
            pass
    # load from last epoch
    oDIR = DIR + '/{0:d}_epochs/'.format(already_done_epochs)
    try:
        rnn = torch.load(oDIR+'model.pth')
        fitness = float(open(oDIR + 'fitness','r').read())
        logging.warning('Continuing with previous training up to {EPOCHS:d} epochs'.format(EPOCHS=EPOCHS))
    except:
        logging.warning('Starting a new training sequence from scratch')
        rnn = RNN(FEATURES,hidden_dim=HIDDEN_DIM,weight_decay=WEIGHT_DECAY)
        already_done_epochs = 0

    rnn.lr=LR
    for millennium in xrange(already_done_epochs,EPOCHS,EV):
        print millennium,'->',millennium + EV
        # train
        train_history = train_model(batch_gen,Q,rnn,epochs=EV,ev=EV/2,window_size=WINDOW)
        # validate
        fitness, validation = validate(Q,rnn,WINDOW)
        # package output
        oDIR = DIR + '/{0:d}_epochs/'.format(millennium+EV)
        try:
            os.makedirs(oDIR)
        except OSError:
            pass
        # save model
        torch.save(rnn,oDIR+'model.pth')
        # save validation
        np.savez(open(oDIR+'validation.npz','wb'),**validation)
        os.system('python plot_validation.py {0:s}'.format(oDIR+'validation.npz'))
        # save log train_history
        with open(oDIR + 'train_history.txt','w') as fout:
            fout.write(train_history)

        with open(oDIR + 'torch_version','w') as fout:
            fout.write(torch.__version__+'\n')

        with open(DIR+'epoch','w') as fout:
            fout.write(str(millennium+EV))

        with open(oDIR + 'fitness','w') as fout:
            fout.write(str(fitness))

    return fitness

if __name__ == '__main__':
    from fire import Fire
    Fire(main)
