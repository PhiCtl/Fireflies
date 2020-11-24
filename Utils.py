import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #,normalize
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K

#get categories proportions -> can be used in report to show imbalance: TO DO
def get_labels_prop(y, verbose = 0):
    """From labels vector, compute proportion of each label
    Argument: label vector, flag to print the label distribution
    Return: Vector of proportions for each label"""
    
    tot_items = np.sum(y) #sum of all entries (1s and 0s) of y vector
    labels = ["arch", "burrow + arch", "drag + arch", "groom","tap + arch","egg", "proboscis", "idle"]
    F = np.array(8)
    for i in range(y.shape[0]):
        F = F + np.sum(y[i,:,:], axis = 0)
        
    F = F / tot_items #label proportion defined as number of occurrences of this label / tot number of occurences
    
    if verbose :
        print("Labels proportions: ")
        for label, prop in zip(labels,F):
            print(label, ": ", prop)
            print('\n')
    return F

def preprocess(x):
    """Standardize (x,y) positions (so every X[:,:,3*j], X[:,:,3*j+1] for j=0 -> 75/3)
    Argument: feature vector
    Return: standardize feature vector"""
    n_features = np.int(x.shape[2]/3)
    for j in range(n_features): 
        scaler = StandardScaler()
        x[:,:,3*j] = scaler.fit_transform(x[:,:,3*j])
        x[:,:,3*j+1] = scaler.fit_transform(x[:,:,3*j+1])
            
    return x

def class_weights(y_tr):
    """Compute the weights for the custom loss
    Argument: training Y vector
    Return: Positive weights, negative weights, proportional weights"""
    positive_weights = {}
    negative_weights = {}
    F = 1/get_labels_prop(y_tr)
    tot = 0
    for el in F:
        tot += el
    F = F/tot
    print("check: ", np.sum(F))
    for i in range(8):
        positive_weights[i] = y_tr.shape[0]/(2*np.count_nonzero(y_tr[:,:,i]==1))
        negative_weights[i] = y_tr.shape[0]/(2*np.count_nonzero(y_tr[:,:,i]==0))
    return positive_weights, negative_weights, F

def binary_CE_weighted(y_true, y_pred):
    """Weighted custom loss, two options for the weights: 
    [positive and negative], or [inversely proportional to label distribution], see weights computation above
    WEIGHTS SHOULD BE DEFINED GLOBALLY (FOR THE MOMENT)
    Argument: ground truth Y, predicted Y
    Return: weighted binary cross entropy"""
    loss = 0
    for i in range(8):
        #loss -= pos[i]*y_true[i]*K.log(y_pred[i]) + neg[i]*(1-y_true[i])*K.log(1-y_pred[i])
        loss -= w[i] * (y_true[:,:,i]*K.log(y_pred[:,:,i]) + (1-y_true[:,:,i])*K.log(1-y_pred[:,:,i]))
        
    return loss

def train_val_test_split(X, Y, ratio_tr_te = 0.2, ratio_tr_val = 0.2):
    """Build train validation test splits from raw data
    Argument: Raw matrices X and Y (3D)
    Return: processed train, validation and test sets"""
    X_processed = preprocess(X)
    x_tr, x_te, y_tr, y_te = train_test_split(X_processed, Y, test_size = ratio_tr_te, random_state = 200)
    x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size = ratio_tr_val, random_state = 240)
    get_labels_prop(y_tr)
    get_labels_prop(y_te)
    #get_labels_prop(y_val)
    return [x_tr, y_tr], [x_val, y_val], [x_te, y_te]
   

