#This mini-project is based on the 'Journey through data' Lab, given at the MLOPs specialization

import os
import shutil
import random
import zipfile
import tarfile
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import wget

# To ignore some warnings about Image metadata that Pillow prints out
import warnings
warnings.filterwarnings("ignore")

def downloadArtifacts():
    # # Download datasets
    #
    # # Cats and dogs
    # !wget https://storage.googleapis.com/mlep-public/course_1/week2/kagglecatsanddogs_3367a.zip
    #
    # # Caltech birds
    # !wget
    # https: // storage.googleapis.com / mlep -  public / course_1 / week2 / CUB_200_2011.tar
    #
    # # Download pretrained models and training histories
    # !wget - q - P / content / model - balanced / https: // storage.googleapis.com / mlep - public / course_1 / week2 / model - balanced / saved_model.pb
    # !wget - q - P / content / model - balanced / variables / https: // storage.googleapis.com / mlep - public / course_1 / week2 / model - balanced / variables / variables.data - 00000 - of - 00001
    # !wget - q - P / content / model - balanced / variables / https: // storage.googleapis.com / mlep - public / course_1 / week2 / model - balanced / variables / variables.index
    # !wget - q - P / content / history - balanced / https: // storage.googleapis.com / mlep - public / course_1 / week2 / history - balanced / history - balanced.csv
    #
    # !wget - q - P / content / model - imbalanced / https: // storage.googleapis.com / mlep - public / course_1 / week2 / model - imbalanced / saved_model.pb
    # !wget - q - P / content / model - imbalanced / variables / https: // storage.googleapis.com / mlep - public / course_1 / week2 / model - imbalanced / variables / variables.data - 00000 - of - 00001
    # !wget - q - P / content / model - imbalanced / variables / https: // storage.googleapis.com / mlep - public / course_1 / week2 / model - imbalanced / variables / variables.index
    # !wget - q - P / content / history - imbalanced / https: // storage.googleapis.com / mlep - public / course_1 / week2 / history - imbalanced / history - imbalanced.csv
    #
    # !wget - q - P / content / model - augmented / https: // storage.googleapis.com / mlep - public / course_1 / week2 / model - augmented / saved_model.pb
    # !wget - q - P / content / model - augmented / variables / https: // storage.googleapis.com / mlep - public / course_1 / week2 / model - augmented / variables / variables.data - 00000 - of - 00001
    # !wget - q - P / content / model - augmented / variables / https: // storage.googleapis.com / mlep - public / course_1 / week2 / model - augmented / variables / variables.index
    # !wget - q - P / content / history - augmented / https: // storage.googleapis.com / mlep - public / course_1 / week2 / history - augmented / history - augmented.csv

    print("hi")

if __name__ == "__main__":
    downloadArtifacts()