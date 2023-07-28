Machine Learning Powered Bottom-up Alloy Catalyst Discovery
============================================================

This repository provides the framework used to optimize and discover n+1 element catalysts from simpler n element catalyst. The code is scalable and can extrapolate to a wide range of oxide free catalysts. The full work is published in Energy Advances (RCS) - "Machine learning assisted binary alloy catalyst design for the electroreduction of CO2 to C2 products" (DOI: 10.1039/d2ya00316c).

Table of Contents
=================

<!--ts-->
   * [Scientific signifiance](#scientific-signifiance)
   * [Usage](#usage)
   * [Code structure](#code-structure)
      * [File contents](#file-contents)
      * [Dependencies](#dependencies)
<!--te-->

Scientific-signifiance
======================

In addition to the ML framework, this work performs the first ever investigation into the bidentate adsorption of COCOH*, a key intermediate in the CO<sub>2</sub>RR pathway towards its most valuable products (C<sub>2+</sub>). As seen below, literature exclusively focused on the monodentate pathway but this work shows the pathway energetics of bidentate dual CO co-adsorption followed by hydrogenation can be more favorable.

<p align="center" width="75%">
    <img width="50%" src="images/bidentate-nobg.png"> 
</p>

Data Science and ML signifiance
===============================

Usage
=====
The main notebook titled 'Catalyst_Discovery_Framework' performs configurational space exploration, predicts optimal catalyst structures and automatically generates VASP readable geometry files based on the ML guided optimization.

The exact pipeline used to discover catalysts is visualized below. The notebook elaborates on the design decisions, science and discovery featured in the publication.

<p align="center" width="75%">
    <img width="50%" src="images/pipeline.png"> 
</p>

Code structure
==============

File contents
-------------
In this repo are the following files:

Catalyst_Discovery_Framework.ipynb: a notebook walking through the framework used to discover 3 element alloy catalysts from 2 element binary alloy catalysts (BAC). Models framework based on TensorFlow 2.0 MLP NN.

helpers.ipynb: helper function with the algorithms that power this framework

saved_NN_parameters folder: saved weight/bias so you dont have to train a model yourself (conveniance only)
saved_dataset.npy: train/test split for saved model

Cu_pure: sample structure for automated catalyst surface generation

All code is thoroughly commented but if you have any questions, please feel free to reach out to me!

Dependencies
------------

(Insert require packages here July 28, 2023)
