# Machine Learning Powered Bottom-up Alloy Catalyst Discovery
This is the code used to discover n+1 element catalysts from simpler n element catalyst. Code is scalable for larger element sizes as well. The paper is titled "Machine learning assisted binary alloy catalyst design for the electroreduction of CO2 to C2 products" (DOI: 10.1039/d2ya00316c)

The main notebook titled 'Catalyst_Discovery_Framework' performs configurational space exploration, predicts optimal catalyst structures and automatically generates VASP readable geometry files based on the ML guided optimization.

In this repo are the following files:

Catalyst_Discovery_Framework.ipynb: a notebook walking through the framework used to discover 3 element alloy catalysts from 2 element binary alloy catalysts (BAC). Models framework based on TensorFlow 2.0 MLP NN.

helpers.ipynb: helper function with the algorithms that power this framework

saved_NN_parameters folder: saved weight/bias so you dont have to train a model yourself (conveniance only)
saved_dataset.npy: train/test split for saved model

Cu_pure: sample structure for automated catalyst surface generation
