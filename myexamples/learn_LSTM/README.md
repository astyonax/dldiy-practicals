# Modeling Paris traffic

After my previous two studies on [traveling times in Paris](https://github.com/astyonax/TimingParis) and [traffic in Paris](https://github.com/astyonax/heartbeat-traffic) it is time to model the traffic in Paris. As before, I use the dataset of OpenDataParis reporting the number of cars going through checkpoints fixed on the city, the data is sampled once per hour.

In this case, by modeling I mean finding a function `F(x(t))=x(t+1h)` to propagate forward of 1h any arbitrary state.
For the sake of learning some more tricks with `pytorch`, I'll first make an NN model based on a LSTM cell.

TL;DR: **NOT** a success story yet:
[notebook here](LSTM-2.ipynb) and [full code here](lstm.py).

To see: https://discuss.pytorch.org/t/lstm-time-series-prediction/4832

## The data

The data is a coarse grained version of the raw dataset. 
It is a matrix ~ 300x2000  (time x features), visualized in [traffic in Paris](https://github.com/astyonax/heartbeat-traffic).

## The model
The model, which contains the NN itself, the optimizer, the data scaler (scikit.learn's RobustScaler) and the data loader is in the file `lstm.py`.

To the incoming data, 3 features are appended:
1. the time (hour)
2. cos(h/24)^2
3. sin(h/24)^2


### The NN

```
    Linear(in,hid)->
    Cell(hid,hid)->
    [Cell2(hid,hid)->] *optional layer
    Linear(hid,in)->
    Max(in)
```
The incoming data has `in` dimensions, which are squeezed to `hid` but the linear map,
then the `LSTM` cell operates, and the `hid` counters are decompressed to the original number.
The idea of compressing comes about because PCA showed that 1 mode covers about 90% of the signal, so may be there is no need have a 2000 dimensional LSTM cell.

### Optimization

When learning NN and deep-learning, we are usually confronted with the need to learn the model weights.
But, we need to `learn` or some-how find/optimize the hyperparamters too.

Finding the best hyper-parameters (HP) it's usually a pain. To avoid wasting time,
I turned to the idea that if a set of HP performs better than other on a small number of epochs, it will also perform better on a long optimization.

So a first genetic optimization algorithm (GEA), in `gea.py`, optimizes for the HP leading to the best validation (see later)  in 1000 epochs.
This set of HP are then used to train up to 1e5 epochs (it's really fast, just minutes).

The individuals of the GEA are a dictionary of HP: window size, hidden dimensions, and weight decay.
To evaluate their fitness I simply train the model with the given set of parameters (training is done with `Adagrad(lr=0.1)`.

Learning is measured with `MSELoss` or mean-squared-error between the predicted counts and the real ones,
validation is MSE against the average counts at fixed time.

## Result

### On the ML
Well.. we get that the machine does not learn the peaks, and has a rather noisy signal (see below).
Plus, it essentially ignores the time signal. Probably there is a way to get it included.
May be a sort of convolutional approach would help bcs many counters are strogly correlated (as shown by the PCA).

### On the traffic
Since there is no conservation law for number of cars though all counters, we should not expect the machine to be able to propagate forward
an arbitrary state. This is obvious _a posteriori_ but I didn't though of it initially. 

Even with a simple linear model $y_{t+1}=W_{t,t+1}y_t+b_t$ and enforcing that $|W|=1$, the bias has to be not zero $b_t\neq 0$ then the machine can just `learn` the number of expected cars at time $t+1$. Indeed, then it is more efficient to learn $y_{t+1}=0_{t,t+1}y_t+b_t$!

I see no way to estimate the flux.. may be only by keeping the average constant?
