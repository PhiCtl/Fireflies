# Supervised classification of fly behaviors from pose tracking data

## Table of contents
* [Architecture] (#Architecture)
* [Installation](#Installation)
* [Predictions](#Predictions)
* [Notebooks](#Notebooks)

## Architecture
The repository contains the code files and Jupyter Notebooks for cross validation, along with a folder _Results_. 
You should first of all unzip the model weights files and place them in the Results folder. 
The following lines can be run at the beginning of your Colab notebook:
```
$ !unzip Results/opt_LSTM_model.zip
$ !unzip Results/opt_TCN_model.zip
```
The optimal model architecture for Random Forest is already in a correct format.

## Installation
We strongly advise you to run our program on Google Colab framework to speed the runtime. 

### Packages
We use keras to build our neural networks model, along with the binary focal loss. 
The following lines should be run at the beginning of the colab notebook.  
```
$ !pip install focal-loss
$ !pip install plot_keras_history
```
### Data
For confidentiality issues, the training data we use may not be shared, and you might request access to Prof. Alexander Mathis. The data should be inserted in a folder called _data_fly_.

## Predictions
Once the 




## Notebooks

### Run an experiment
### Cross validation
### Data augmentation
