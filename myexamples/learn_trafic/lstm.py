#!/bin/env python2.7
import numpy as np
from torch.autograd import Variable
from torch import nn
import torch
from sklearn.preprocessing import RobustScaler
from tqdm import trange
import warnings
import os
# load data file
warnings.warn('fitting only on closing price')
# time_series = np.load(open('../data/offline_prices.npy','r'))[:,:,-1]
time_series = np.load(open('data/time_series_traffic_paris.npy','r'))

# scale data
scaler = RobustScaler()
Q = scaler.fit_transform(time_series)

# Hyper Parameters
WEIGHT_DECAY = 1e-5
NESTEROV = True
MOMENTUM = 0.9
BATCH_SIZE = 600
# Parameters
FEATURES = Q.shape[-1]

# functions
# assumes data fits in memory
def batch_gen(data,window_size,batch_size=BATCH_SIZE,shuffle=True):
    #~ win creat
    ws = window_size
    windows = [[data[i+n] for i in range(0, len(data)-1-ws, ws)] for n in range(ws)]
    windows = np.asarray(windows).transpose(1,0,-1)
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
def new_train_model(batches_gen,data,model,window_size,epochs=1000,train=True,ev=1):
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
    def __init__(self, inout_dim, hidden_dim =256, lr = 0.001):
        super(RNN, self).__init__()

        self._inout_dim = inout_dim
        self._hidden_dim = hidden_dim
        self.lin_in = nn.Linear(inout_dim,hidden_dim)
        self.cell = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.lin_out = nn.Linear(hidden_dim,inout_dim)

        # functions
        self.__lr = lr
        self.loss = nn.MSELoss()
        self.optimizer = self.__init_optimizer()
#         self.activation = torch.nn.Activation('linear')

    def forward(self, c):
        hidden = (Variable(torch.randn(1,1,self._hidden_dim)),
                  Variable(torch.randn(1,1,self._hidden_dim)))

        output = self.lin_in(c)
        output,hidden = self.cell(output,hidden)
        output = self.lin_out(output)
        output = output.max(dim=1)[0]
#         output = self.activation(output)
        return output

    @property
    def lr(self):
        return self.__lr

    @lr.setter
    def lr(self,new_lr):
        self.__lr = new_lr
        self.optimizer = self.__init_optimizer()

    def __init_optimizer(self):
        return torch.optim.SGD(self.parameters(),
                               lr = self.__lr,
                               momentum=MOMENTUM,
                               weight_decay=WEIGHT_DECAY,
                               nesterov=NESTEROV)

# main
def main(W,HIDDEN_DIM,EPOCHS,EV=1000):
    # os stuff
    DIR = '{}_{}/'.format(W,HIDDEN_DIM)
    print DIR

    already_done_epochs = 0
    try:
        os.makedirs(DIR)
    except OSError:
        # warnings.warn("overwriting previous files in folder "+DIR)
        try:
            already_done_epochs = int(open(DIR+'/epoch').read())
            warnings.warn('found {0:d} epochs done'.format(already_done_epochs))
        except IOError:
            pass
    # load from last epoch
    oDIR = DIR + '/{0:d}_epochs/'.format(already_done_epochs)
    try:
        rnn = torch.load(oDIR+'model.pth')
        warnings.warn('Continuing with previous training up to {EPOCHS:d} epochs'.format(EPOCHS=EPOCHS))
    except:
        warnings.warn('Starting a new training sequence from scratch')
        rnn = RNN(FEATURES,hidden_dim=HIDDEN_DIM)
        already_done_epochs = 0

    rnn.lr=0.1
    for millennium in xrange(already_done_epochs,EPOCHS,EV):
        print millennium,'->',millennium + EV
        # train
        train_history = new_train_model(batch_gen,Q,rnn,epochs=EV,ev=EV/2,window_size=W)
        # validate
        validation = validate(Q,rnn,W)[1]
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
        # save log train_history
        with open(oDIR + 'train_history.txt','w') as fout:
            fout.write(train_history)

        with open(oDIR + 'torch_version','w') as fout:
            fout.write(torch.__version__+'\n')

        with open(DIR+'epoch','w') as fout:
            fout.write(str(millennium+EV))

if __name__ == '__main__':
    from fire import Fire
    Fire(main)
