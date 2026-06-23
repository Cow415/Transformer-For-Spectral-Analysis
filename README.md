# Transformer-For-Spectral-Analysis
## Introduction
This repository holds the codes to build and train spectral autoencoder model for Raman signal processing, desirably remove/preserve certain chemical signatures. 

Function implementations are in .py and work is done in model.ipynb

## Environment Dependencies:
  - python=3.13.12
  - ipykernel
  - numpy
  - scipy
  - matplotlib
  - jupyterlab
  - notebook
  - pip

## Organization
    Transformer/
    ├── model/              
    │   ├── augmentation.py 
    │   ├── loader.py       
    │   ├── preprocess.py   
    │   ├── augmentation.py 
    │   ├── structure.py    
    │   ├── loss.py         
    │   ├── training.py     
    │   ├── metrics.py      
    │   ├── visualization.py
    │   ├── benchmark.py    
    │   ├── inference.py    
    │   ├── utils.py        
    │   └── model.ipynb     
    ├── environment/        
    └── data/               
        ├── data_vis.ipynb  