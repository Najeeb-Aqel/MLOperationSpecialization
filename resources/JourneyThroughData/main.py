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

    cats_and_dogs_zip = './content/kagglecatsanddogs_3367a.zip'
    caltech_birds_tar = './content/CUB_200_2011.tar'
    base_dir = './content/raw_data'

    # Extract raw images
    if not os.path.isdir("./content/raw_data"):
        with zipfile.ZipFile(cats_and_dogs_zip, 'r') as my_zip:
            my_zip.extractall(base_dir)
        with tarfile.TarFile(caltech_birds_tar, 'r') as my_tar:
            my_tar.extractall(base_dir)

    # define base directories
    base_dogs_dir = os.path.join(base_dir, 'PetImages/Dog')
    base_cats_dir = os.path.join(base_dir, 'PetImages/Cat')
    base_birds_dir = os.path.join(base_dir, 'PetImages/Bird')

    # preprocess bird images
    if not os.path.isdir(base_birds_dir):
        os.mkdir(base_birds_dir)
        raw_birds_dir = './content/raw_data/CUB_200_2011/images'
        for subdir in os.listdir(raw_birds_dir):
            subdir_path = os.path.join(raw_birds_dir, subdir)
            for image in os.listdir(subdir_path):
                shutil.move(os.path.join(subdir_path, image), os.path.join(base_birds_dir))

    # count images in each class
    print(f"There are {len(os.listdir(base_birds_dir))} images of birds")
    print(f"There are {len(os.listdir(base_dogs_dir))} images of dogs")
    print(f"There are {len(os.listdir(base_cats_dir))} images of cats")
