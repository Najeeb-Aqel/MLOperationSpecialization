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
import subprocess

# To ignore some warnings about Image metadata that Pillow prints out
import warnings
warnings.filterwarnings("ignore")


def download_artifacts():
    # Download datasets
    subprocess.call(['sh', './artifacts.sh'])


if __name__ == "__main__":
    if not os.path.isdir("./content"):
        download_artifacts()

    cats_and_dogs_zip = '/content/kagglecatsanddogs_3367a.zip'
    caltech_birds_tar = '/content/CUB_200_2011.tar'

    base_dir = '/content/raw_data'

    with zipfile.ZipFile(cats_and_dogs_zip, 'r') as my_zip:
        my_zip.extractall(base_dir)

    with tarfile.TarFile(caltech_birds_tar, 'r') as my_tar:
        my_tar.extractall(base_dir)
