
import numpy as np
from Utils import train_te_val_split, preprocess
from sklearn.model_selection import train_test_split

def data_augmentation(x, y, T):
    """Split training set into smaller time windows
    Arguments: (x,y) total data set (3D arrays)
    Return: training set (batches of time steps size 2 times less than validation and test set)
            validation set (full time steps length)
            test set (full time steps length) """

    x_tr, x_te, y_tr, y_te = train_test_split(preprocess(x), y, test_size = 0.2, random_state = 200)
    x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size = 0.2, random_state = 200)
    #build indices
    n = np.int(x_tr.shape[1]/T)
    bs = x_tr.shape[0]
    batch_x = np.zeros((bs*T,n,x_tr.shape[2]))
    batch_y = np.zeros((bs*T,n,y_tr.shape[2]))

    #build batches
    for i in range(T):
      batch_x[i*bs:bs*(i+1),:,:] = x_tr[:,i*n:n*(i+1),:]
      batch_y[i*bs:bs*(i+1),:,:] = y_tr[:,i*n:n*(i+1),:]
    """batch_x[bs:bs*2,:,:] = x_tr[:,n:n*2,:]
    batch_y[bs:bs*2,:,:] = y_tr[:,n:n*2,:]
    batch_x[bs*2:,:,:] = x_tr[:,n*2:,:]
    batch_y[bs*2:,:,:] = y_tr[:,n*2:,:]"""
    
    
    return [batch_x, batch_y], [x_te, y_te], [x_val, y_val]


def data_augmentation_2(x,y):
    "Creates a new larger data set by randomly perturbing positions with gaussian noise"
    train, test, val = train_te_val_split(preprocess(x), y)
    x_tr_1 = train[0]
    offset(x_tr_1)
    train, test, val = train_te_val_split(x, y)
    X_tr = np.vstack((train[0], x_tr_1))
    Y_tr = np.vstack((train[1], train[1]))
    return [X_tr, Y_tr], test, val

def offset(x):
    """Adds gaussian noise to features vector
    Argument: (3D) features vector x
    Returns: perturbed positions"""
    np.random.seed(123)
    t = np.random.normal()
    n_features = np.int(x.shape[2]/3)
    for j in range(n_features): 
        x[:,:,3*j] += t 
        x[:,:,3*j+1] += t 