import h5py
import numpy as np
import pandas as pd
import os
from Utils import preprocess
from tensorflow.keras.backend import expand_dims

#load the pose files from a given folder
def load_file_x(folder, scorer = 'DeepCut_resnet50_FlyMar16shuffle0_500000' ):
    """Loads pose files (.hf5) from folder in data_fly data folder
    Arguments: folder (should be as follow from current directory : data_fly/folder)
    Returns: panda DataFrame of size 720 x 75 (75 features)"""
    
    pose=DC = pd.read_hdf(os.path.join('data_fly',folder,folder+scorer+'.h5'), 'df_with_missing')
    return pose

#load the annotation files
def load_file_y(folder, ann = 'all_ann.csv'):
    """Loads annotation files (.csv)
    Argument: folder (should be as follow from current directory : data_fly/folder)
    Returns: panda Data Frame with 720 rows and 8 columns"""
    annotation = pd.read_csv(os.path.join('data_fly',folder,ann),header=None).T
    annotation = expand(annotation) #reshape annotation, adding a category and merging two labels
    return annotation

#expand to 8 categories
def expand(vect):
    """Expands an annotation vector from 7 to 8 categories and merges two labels
    Argument: annotation panda data frame of size 720 x 7
    Returns: expanded data frame"""
    y = vect
    #create idle category as 8th category
    y.insert(7,7,0)
    y.loc[(np.sum(y,axis = 1) == 0),7] = 1
    #create burrow + arch, arch + drag, tap + arch categories since arch is a mandatory behaviour for burrow, drag and tap
    y.loc[y.loc[:,1] == 1,0] = 0
    y.loc[y.loc[:,2] == 1,0]= 0
    y.loc[y.loc[:,4] == 1,0] = 0
    return y

#create data set
def load_all(fold_list):
    """Creates overall data set
    Argument: list of folders within data_fly/
    Returns: Features vector (np.array) of size 145 (datapoints) x 720 (time frames) x 75 (features)
            Labels vector(np.array) of size 145 x 720 x 8 (categories)"""
    x = list()
    y = list()
    for name in fold_list:
        data_x = load_file_x(name)
        data_y = load_file_y(name)
        x.append(data_x.T)
        y.append(data_y.T)
    x = np.dstack(x)
    y = np.dstack(y)
    return x.T,y.T


def load_training_data():
    """Should be run at same level as data_fly folder"""
    dir_list = os.listdir('data_fly')
    dir_list.remove('.ipynb_checkpoints')
    folders = dir_list
    return load_all(folders)

def load_test_data(folder_name):
  """Loads test data for final prediction
     Arguments: folder_name containing pose tracking file  and annotated file
                should be contained in data_fly folder
     Returns: 3D preprocessed X and 3D Y for predictions"""

  X = load_file_x(folder_name, scorer = 'DLC_resnet50_FlyMar16shuffle0_500000')
  Y = np.empty((0,8));
  for ann in ['ann1.csv','ann2.csv','ann3.csv']:
    y = load_file_y(folder_name, ann)
    print(y.shape)
    y = expand(y)
    Y = np.vstack((Y,y))
  print(Y.shape, X.shape)
    
  X = np.expand_dims(X,axis =0)
  Y = np.expand_dims(Y, axis =0)
  X = preprocess(X)
  return X, Y
