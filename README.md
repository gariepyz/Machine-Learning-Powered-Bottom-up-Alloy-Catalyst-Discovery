# Multi-component-alloy-catalyst-discovery
This is the code used to discover n+1 elementt catalyst from simpler n element catalyst. Code is scalable for larger element sizes. Code used in my paper "Machine learning assisted binary alloy catalyst design for the electroreduction of CO2 to C2 products" (DOI: 10.1039/d2ya00316c)

The main notebook import a dataset/model, performs configurational space exploration, predicted optimal catalyst structures and automatically generates VASP readable geometry files based on the ML guided optimization.

In this repo are the following files:

3_ele_optimization.ipynb: a notebook walking through the pipeline used to discover 3 element alloy catalysts from 2 element binary alloy catalysts (BAC). Models framework based on TensorFlow 2.0 MLP NN.

helpers.ipynb: help function to clean up the main notebook

Eads.csv: formatted dataset of 8 different Cu BACs. Needed for the notebook.
3_ele_preds.csv: example of optimization predictions results
3_ele_optimization: example of optimization structure format
test.txt: more optimization saves

saved_NN_val2 folder: saved weight/bias so you dont have to train a model yourself (conveniance only)
load_2.npy: train/test split for saved model

Cu_pure: sample structure for automated catalyst surface generation
