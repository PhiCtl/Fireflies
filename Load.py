import h5py
import numpy as np
import pandas as pd
import os

#load the pose files from a given folder
def load_file_x(folder):
    """Loads pose files (.hf5) from folder in data_fly data folder
    Arguments: folder (should be as follow from current directory : data_fly/folder)
    Returns: panda DataFrame of size 720 x 75 (75 features)"""
    
    scorer='DeepCut_resnet50_FlyMar16shuffle0_500000' #scorer to load poses
    pose=DC = pd.read_hdf(os.path.join('data_fly',folder,folder+scorer+'.h5'), 'df_with_missing')
    return pose

#load the annotation files
def load_file_y(folder):
    """Loads annotation files (.csv)
    Argument: folder (should be as follow from current directory : data_fly/folder)
    Returns: panda Data Frame with 720 rows and 8 columns"""
    annotation = pd.read_csv(os.path.join('data_fly',folder,'all_ann.csv'),header=None).T
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