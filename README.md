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
$ !unzip data_fly.zip
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
However, due to Colab stochasticity, we have some issues with the reproducibility of our results. Our report results refer to scores obtained with the optimal models stored under opt_LSTM_model.zip, opt_TCN_model.zip and opt_random_forest_model.joblib. 

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

### From terminal
If one wants to make predictions using one of our model, the run.py file  should be launch at the same level as the other code files. Further instructions will be given to the user through the command line prompt.
The user should first of all enter the name of the folder in _data_fly_ folder that contains the test files. Then the wanted model for predictions can be entered: LSTM, TCN or Random_Forest.
´´´
$ ROI5_t4
´´´
So _data_fly_ should have the following architecture:
    .
    ├── data_fly/                            
    |  ├── CS_170910_avi_ROI1_E02
    |  ├── ... (145 files)
    |  ├── ROI5_t4

### From Notebooks
In all our notebook, if one only wants to make predictions on our test set, the following line should be run once _X_tr_ and _Y_tr_ are built with the correct flag: LSTM, Random_Forest or TCN.
´´´
$ predict(test[0],test[1], flag)
´´´

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
- regularization: 1e-6
- Regularization: none
- Dropout: 1 dropout layer, dropout = 0.1
- Dense layer: 1 dense activation layer with sigmoid activation function
- Gamma (Binary Focal loss): 2
- Batch size: 32
- Epochs: 200
We tried to fix the seeds to ensure reproducible results but it didn't work perfectly due to GPU stochasticity. To reproduce exactly the results in the report, the user should launch ´´´ predict(X,Y,"LSTM") ´´´  with the best model architecture _opt_LSTM_model_. 

## Random Forest model training
This notebook offers several functionalities
* Building and training a random forest model on training data set, evaluating it against train set. 
* Random search and grid search using cross validation in order to find the optimal hyperparameters

### How to reproduce exactly the results on the test set
The hyper parameters obtained by cross validation for best model predictions on train set are:
- bootstrap
- True
- max_depth: 10
- max_features: 'sqrt'
- min_samples_leaf: 4
- min_samples_split: 2
- n_estimators: 10

## TCN model training
This notebook offers several functionalities
* Building and training a TCN model on training data set, evaluating it against validation set. Several formats of datasets can be built.
* Making predictions of the hold out test set
* The results of the cross validation among the different hyperparameters


###  How to reproduce the results on the test set

The hyper parameters obtained by cross validation for best model predictions on validation set are:

- Regularization: 1e-6
- Dense layer: one dense activation layer with Leaky Relu activation function and one dense layer with sigmoid activation function.
- Gamma (Binary Focal loss): 5
- Filter_size=64
- Kernel_size=2
- Batch size: 1
- Epochs: 200
