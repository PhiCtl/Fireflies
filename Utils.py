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
    F = 1/get_labels_prop(y_tr)
    tot = 0
    for el in F:
        tot += el
    F = F/tot
    return F

def train_te_val_split(X, Y, ratio_tr_te = 0.2):
    """Builds train validation test splits from raw data
    Argument: raw matrices X and Y (3D)
    Returns: processed train and test sets"""
    X = preprocess(X)
  
    x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size = ratio_tr_te, random_state = 200)
    x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size = ratio_tr_te, random_state = 200)
    
    return [x_tr, y_tr], [x_te, y_te], [x_val, y_val]

def time_window_sample(x, y, T):
    """ Expand the features by adding the data T/2 before and T/2 after a specific time step
    Arguments: x set of features 3D
               y set of label 3D
               T number of time steps of the chosen time window, should be a multiple of 2
                  """
    nb_samples=x.shape[0]
    nb_frames=x.shape[1] 
    nb_features=x.shape[2]
    T_=int(np.floor(T/2))
    new_X=np.zeros((nb_samples,nb_frames-T,(T_*2+1)*nb_features))
    new_Y=np.zeros((nb_samples,nb_frames-T,y.shape[2]))
    for k in range(0,nb_samples):
        new_Y[k,:,:]=y[k,T_:nb_frames-T_]
        for i in range(T_,nb_frames-T):
            x_t=x[k,i-T_:i+T_+1,:] ##select columns of x between i-T/2 and i+T/2
            X_T=x_t.reshape(-1)  ##linearize time window
            new_X[k,i,:]=X_T    ##columns of new_X are composed of linearized X_T
        
    return new_X,new_Y

def feature_expansion(x):
    """ Add the velocity in x and y between two consecutive frames to the features, and the distance between two consecutive body parts x positions, which doubles the amount of features
    Arguments:x the features vector to expand
    
                  """
    ##to do:augment features with distance between 2 consecutive points and velocities
    nb_samples=x.shape[0]
    nb_frames=x.shape[1] 
    nb_features=x.shape[2]
    new_X=np.empty([nb_samples,nb_frames,nb_features*2])
        
    for k in range(0,nb_samples):
        for i in range(0, nb_frames-1):
            l=0;
            for j in range(0,nb_features*2-6, 6):
                d_x=abs(x[k,i,l]-x[k,i,l+3]);
                v_x=x[k,i+1,l]-x[k,i,l]/time_step
                v_y=x[k,i+1,l+1]-x[k,i,l+1]/time_step
                
                new_X[k,i,j]=x[k,i,l]
                new_X[k,i,j+1]=x[k,i,l+1]
                new_X[k,i,j+2]=x[k,i,l+2]
                l+=3;
                #print(l)
                new_X[k,i,j+3]=d_x
                new_X[k,i,j+4]=v_x
                new_X[k,i,j+5]=v_y
    new_X=new_X[:,:,0:new_X.shape[2]-6]
    return new_X 

