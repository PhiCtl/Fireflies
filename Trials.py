import numpy as np
from Utils import train_te_val_split, preprocess
from sklearn.model_selection import train_test_split

def time_window_sample(x, y, T): #OK
    """Returns random time window of length T in x and Y 2D arrays"""
    #np.random.seed(seed)
    i = np.int(np.random.uniform(0,x.shape[0]-T-1))
    return x[range(i, i+T),:], y[range(i,i+T),:]


def batch(batch_size,x,y,T): #OK
    """Creates #batch_size batches per sample (x,y) with window of length T
    Input: number of batches, X, Y (2D arrays), length of window
    Output: 3D arrays batch_x and batch_y of sizes (batch_size,T,x.shape[1]) and (batch_size,T,y.shape[1]) """
    batch_x = np.empty((batch_size,T,x.shape[1]))
    batch_y = np.empty((batch_size,T,y.shape[1]))
    for i in range(batch_size):
        x_rd, y_rd = time_window_sample(x,y,T)
        batch_x[i,:,:] = x_rd
        batch_y[i,:,:] = y_rd
    return batch_x, batch_y

def data_augmentation(batch_size, x, y, T):
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size = 0.2, random_state = 200)
    x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size = 0.2, random_state = 200)
    batch_x = np.empty((0,T,x_tr.shape[2]))
    batch_y = np.empty((0,T,y_tr.shape[2]))
    for i in range(x_tr.shape[0]):
        x_rd,y_rd = batch(batch_size, x_tr[i,:,:], y_tr[i,:,:],T)
        offset(x_rd)
        batch_x = np.vstack((batch_x,x_rd))
        batch_y = np.vstack((batch_y,y_rd))
    batch_x = preprocess(batch_x)
    x_te = preprocess(x_te)
    
    return [batch_x, batch_y], [x_te, y_te], [x_val, y_val]

def data_augmentation_2(x,y):
    "Ugly function but might do the job"
    train, test, val = train_te_val_split(preprocess(x), y)
    x_tr_1 = train[0]
    o_x = np.empty((0,x_tr_1.shape[1],x_tr_1.shape[2]))
    o_y = np.empty((0,test[1].shape[1],test[1].shape[2]))

    n_features = np.int(x.shape[2]/3)
    for j in range(n_features):
        x_tr_1[:,:,3*j] *= -1
        x_tr_1[:,:,3*j+1] *= -1
    for i in range(train[1].shape[0]):
        a = np.where(train[1][i,:,4] == 1)
        b = np.where(train[1][i,:,5] == 1)
        if len(a) > 0 or len(b) > 0:
            o_x = np.vstack((o_x, np.expand_dims(x_tr_1[i,:,:], axis = 0)))
            o_y = np.vstack((o_y, np.expand_dims(train[1][i,:,:], axis = 0)))
    train, test, val = train_te_val_split(x, y)
    X_tr = np.vstack((train[0], o_x))
    Y_tr = np.vstack((train[1], o_y))
    return [X_tr, Y_tr], test, val

def offset(x):
    t = np.random.normal()
    n_features = np.int(x.shape[2]/3)
    for j in range(n_features): 
        x[:,:,3*j] += t 
        x[:,:,3*j+1] += t 