
import numpy as np
from Utils import train_te_val_split, preprocess
from sklearn.model_selection import train_test_split

def data_augmentation(x, y):
    """Split training set into smaller time windows
    Arguments: (x,y) total data set (3D arrays)
    Return: training set (batches of time steps size 2 times less than validation and test set)
            validation set (full time steps length)
            test set (full time steps length) """

    x_tr, x_te, y_tr, y_te = train_test_split(preprocess(x), y, test_size = 0.2, random_state = 200)
    x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size = 0.2, random_state = 200)
    #build indices
    n = np.int(x_tr.shape[1]/2)
    bs = x_tr.shape[0]
    batch_x = np.zeros((bs*2,n,x_tr.shape[2]))
    batch_y = np.zeros((bs*2,n,y_tr.shape[2]))

    #build batches
    batch_x[0:bs,:,:] = x_tr[:,0:n,:]
    batch_y[0:bs,:,:] = y_tr[:,0:n,:]
    batch_x[bs:,:,:] = x_tr[:,n:,:]
    batch_y[bs:,:,:] = y_tr[:,n:,:]
    
    return [batch_x, batch_y], [x_te, y_te], [x_val, y_val]


def data_augmentation_2(x,y):
    "Ugly function but might do the job"
    train, test, val = train_te_val_split(preprocess(x), y)
    x_tr_1 = train[0]
    offset(x_tr_1)
    train, test, val = train_te_val_split(x, y)
    X_tr = np.vstack((train[0], x_tr_1))
    Y_tr = np.vstack((train[1], train[1]))
    return [X_tr, Y_tr], test, val

def offset(x):
    t = np.random.normal()
    n_features = np.int(x.shape[2]/3)
    for j in range(n_features): 
        x[:,:,3*j] += t 
        x[:,:,3*j+1] += t 