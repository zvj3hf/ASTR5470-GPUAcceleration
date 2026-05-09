"""
Input/output helper for saving outputs and reading in data
"""

## IMPORTS ##
import os
import yaml
import numpy as np
import pandas as pd


def check_dir(path):
    """create directory if it doesn't exist"""
    os.makedirs(path, exist_ok = True)

def load_configuration(path):
    """loads in the configuration needed"""
    with open(path,"r") as f:
        return yaml.safe_load(f)

def save_array(path, array):
    """save numpy array to a .npy file"""
    np.save(path, array)


def save_csv(path, dataframe):
    """save pandas DF to a csv file"""
    dataframe.to_csv(path, index=False)

def save_text(path,text):
    """save text to a file"""
    with open(path, "w") as f:
        f.write(text)