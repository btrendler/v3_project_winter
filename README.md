# Volume 3 Project: Brainwave Time Series Data Analysis

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)

## Introduction

A typical sleep cycle consists of various stages of REM (rapid eye movement) 
and NREM (non-REM) sleep states. These sleep stages can be identified 
by their relationship to brain activity. Lower brain activity indicates
a deeper sleep state, while higher brain activity indicates wide-awakeness.
Our goal with this project is to find a predictive model between the measured 
metrics and the annotated sleep states. 

Additionally, we seek to explore predicting future brain activity and sleep states using 
Reservoir Computing and Kalman Filtering. Succesful models in these areas will help with
sleep studies and may help create hardware which can identify a user’s sleep
state in real time, to help improve our understanding of the sleep states and
their purposes, especially those with sleeping disorders.

## File Structure
animations -- various visualizations of the brainwaves over time for patients 4001 and 4111.

figures -- figures used in the paper.

projections -- the projected data from using tsne and umap

aggregator.py -- used for aggregating the data -- performs fft and separates the data to train, 
test, and validation sets.

assignment .ipynb -- looks at clustering via UMAP and random forest classifier.

clustering.ipynb -- k-means, gmm, and rfc gridsearch results.

clustering_2.ipynb -- several UMAP clusterings

clustering_figures.py -- used to generate plots 

deep_learning.ipynb -- a deep learning network (not used in the paper)

fancy_alternative.ipynb -- an alternative to running UMAP on unseen data by 
including data as unlabeled data into UMAP

nathan-data-exploration.ipynb -- used for understanding the dataset
and generating preliminary plots

nathan-fourier-exploration-modeling.ipynb -- fft exploration and softmax classification

nathan-modeling -- main exploration of kalman filtering and softmax classification

parameter_comparison.py -- exploration of various parameters for UMAP

parser.py -- converts the edf files to one npz file

random_forests.py -- grid search for random forest

reservoir_computing.py -- a walkthrough of reservoir computing

reservoir_gridsearch.py -- finding optimal parameters for reservoir computing



