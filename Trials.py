import numpy as np

def time_window_sample(x, y, T):
    ##to do: return random time window of length T in x and Y 2D arrays
    #np.random.seed(seed)
    i = np.int(np.random.uniform(0,x.shape[0]-T-1))
    return x[range(i, i+T),:], y[range(i,i+T),:]

def batch(batch_size,x,y,T):
    batch_x = list()
    batch_y = list()
    for i in range(batch_size):
        x_rd, y_rd = time_window_sample(x,y,T)
        batch_x.append(x_rd.T)
        batch_y.append(y_rd.T)
    batch_x = np.dstack(batch_x)
    batch_y = np.dstack(batch_y)
    return batch_x.T,batch_y.T

def training_extraction(batch_size, x, y, T):
    batch_x = list()
    batch_y = list()
    for i in range(x.shape[0]):
        x_rd,y_rd = batch(batch_size, x[i,:,:], y[i,:,:],T)
        batch_x.append(x_rd.T)
        batch_y.append(y_rd.T)
    batch_x = np.dstack(batch_x)
    batch_y = np.dstack(batch_y)
    return batch_x.T,batch_y.T

def offset(x_prev):
    x = x_prev
    t = np.random.normal()
    n_features = np.int(x.shape[2]/3)
    for j in range(n_features): 
        x[:,:,3*j] += t 
        x[:,:,3*j+1] += t 
    return x
    