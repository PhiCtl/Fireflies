import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.model_selection import train_test_split, StratifiedKFold
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
    """Standardizes and normalizes (x,y) positions (so every X[:,:,3*j], X[:,:,3*j+1] for j=0 -> 75/3)
    Argument: features vector X (3D)
    Return: standardized normalized features vector"""
    n_features = np.int(x.shape[2]/3)
    for j in range(n_features): 
        scaler = StandardScaler()
        x[:,:,3*j] = scaler.fit_transform(x[:,:,3*j])
        x[:,:,3*j+1] = scaler.fit_transform(x[:,:,3*j+1])
        x[:,:,3*j] = normalize(x[:,:,3*j])
        x[:,:,3*j+1] = normalize(x[:,:,3*j+1])     
    return x

def class_weights(y_tr):
    """Computes the proportional weights
    Argument: training Y vector (3D)
    Return: Proportional weights"""
    #positive_weights = {}
    #negative_weights = {}
    F = 1/get_labels_prop(y_tr)
    tot = 0
    for el in F:
        tot += el
    F = F/tot
    #print("check: ", np.sum(F))
    #for i in range(8):
        #positive_weights[i] = y_tr.shape[0]/(2*np.count_nonzero(y_tr[:,:,i]==1))
        #negative_weights[i] = y_tr.shape[0]/(2*np.count_nonzero(y_tr[:,:,i]==0))
    return F

def binary_CE_weighted(y_true, y_pred):
    """Weighted custom loss, two options for the weights: 
    [positive and negative], or [inversely proportional to label distribution], see weights computation above
    Argument: ground truth Y, predicted Y
    Return: weighted binary cross entropy"""
    pos = [0.013478341699200595, 0.004400339888322408, 0.020631758679567444, 0.009843856076035303, 0.025709219858156027, 0.5, 0.009639675575056508, 0.0011364705144684454]
    neg = [0.0007321679239757224, 0.0008245757699831673, 0.00071863291239617, 0.0007471530890915649, 0.0007137231738531207, 0.0006954102920723226, 0.0007483561969054181, 0.0017854504260454121]
    
    loss = 0
    y_pred = y_pred > 0.5
    y_pred = tf.cast(y_pred, tf.int64)
    print(y_pred)
    for i in range(8):
        loss -= pos[i]*y_true[:,:,i]*K.log(y_pred[:,:,i]) + neg[i]*(1-y_true[:,:,i])*K.log(1-y_pred[:,:,i])
        #loss -= w[i] * (y_true[:,:,i]*K.log(y_pred[:,:,i]) + (1-y_true[:,:,i])*K.log(1-y_pred[:,:,i]))
        
    return loss

def train_te_val_split(X, Y, ratio_tr_te = 0.2):
    """Builds train validation test splits from raw data
    Argument: Raw matrices X and Y (3D)
    Returns: processed train and test sets"""
    X_processed = preprocess(X)
    x_tr, x_te, y_tr, y_te = train_test_split(X_processed, Y, test_size = ratio_tr_te, random_state = 200)
    #x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size = ratio_tr_te, random_state = 200)
    x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size = ratio_tr_te, random_state = 200)
    #get_labels_prop(y_val)
    return [x_tr, y_tr], [x_te, y_te], [x_val, y_val]
   

