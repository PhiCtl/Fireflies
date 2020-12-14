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
The optimal model architecture for Random Forest is already in a correct format.

## Installation
We strongly advise the user to run our program on Google Colab framework to speed the runtime. 

### Packages
We use keras to build our neural networks model, along with the binary focal loss. 
The following lines should be run at the beginning of the colab notebook.  
```
$ !pip install focal-loss
$ !pip install plot_keras_history
```
### Data
For confidentiality issues, the training data we use may not be shared, and the user might request access to Prof. Alexander Mathis. The data should be inserted in a folder called _data_fly_.

## Predictions
If one wants to make predictions using one of our model, the run.py file  should be launch at the same level as the other code files. Further instructions will be given to the user through the command line prompt.


## Notebooks

### LSTM model training
