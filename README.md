# Supervised classification of fly behaviors from pose tracking data

## Table of contents
* [Architecture](#Architecture)
* [Installation](#Installation)
* [Predictions](#Predictions)
* [Notebooks](#Notebooks)

## Architecture
The repository contains the code files and Jupyter Notebooks for cross validation, along with a folder _Results_. 
The user should first of all unzip the model weights folders and place them in the _Results_ folder. 
The following lines can be run at the beginning of a Colab notebook:
```
$ !unzip Results/opt_LSTM_model.zip
$ !unzip Results/opt_TCN_model.zip
```
Note that the optimal model architecture for Random Forest is already in a correct format.

### Files
    .
    ├── Results/                            #Models architecture (the predictions results will be saved here as well)
    |  ├── opt_LSTM_model.zip
    |  ├── opt_TCN_model.zip
    |  ├── opt_random_forest_model.joblib
    ├── Data_augmentation.py                #Data augmentation trials            
    ├── Load.py                             #Functions to load the data           
    ├── Metrics.py                          #Metrics custom functions
    ├── Model_training_LSTM.ipynb           #Cross validation notebook LSTM       
    ├── Model_training_TCN.ipynb            #Cross validation notebook TCN  
    ├── Model_training_random_forest.ipynb  #Cross validation notebook R.F.  
    ├── README.md
    ├── run.py                              #run file
    ├── tcn.py                              #tcn functions
    ├── Train.py                            #training and predict functions
    └── Utils.py                            #utils
    

## Installation
We strongly advise the user to run our program on Google Colab framework (with a GPU) to speed up the runtime. 

### Packages
We use keras to build our neural networks model, along with the binary focal loss. 
The following lines should be run at the beginning of the colab notebook.  
```
$ !pip install focal-loss
$ !pip install plot_keras_history
```
The versions are : 
* focal-loss :  0.0.6
* plot_keras_history: 1.1.26
* tensorflow: 2.3.0
* keras: 2.4.3
* sklearn: 0.0
    
### Data
For confidentiality issues, the training data we use may not be shared, and the user might request access to Prof. Alexander Mathis. The data should be inserted in a folder called _data_fly_.

## Predictions
If one wants to make predictions using one of our model, the run.py file  should be launch at the same level as the other code files. Further instructions will be given to the user through the command line prompt.


## Notebooks

### LSTM model training
This notebook offers several functionalities
* Building and training an LSTM / bidirectional LSTM model on training data set, evaluating it against validation set. Several formats of datasets can be built.
* Cross validating the different hyperparameters
* Making predictions of the hold out test set

#### How to reproduce exactly the results on the test set
The hyper parameters obtained by cross validation for best model predictions on validation set are:
- Model type : 1 bidirectional LSTM layer
- Number of nodes: 600
- Regularization: none
- Dropout: 1 dropout layer, dropout = 0.1
- Dense layer: 1 dense activation layer with sigmoid activation function
- Gamma (Binary Focal loss): 2
- Batch size: 32
- Epochs: 200

