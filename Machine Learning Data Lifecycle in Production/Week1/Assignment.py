# Import packages
import os
import pandas as pd
import tensorflow as tf
import tempfile, urllib, zipfile
import tensorflow_data_validation as tfdv


from tensorflow.python.lib.io import file_io
from tensorflow_data_validation.utils import slicing_util
from tensorflow_metadata.proto.v0.statistics_pb2 import DatasetFeatureStatisticsList, DatasetFeatureStatistics

# Set TF's logger to only display errors to avoid internal warnings being shown
tf.get_logger().setLevel('ERROR')

# Read CSV data into a dataframe and recognize the missing data that is encoded with '?' string as NaN
df = pd.read_csv('dataset_diabetes/diabetic_data.csv', header=0, na_values = '?')

# Preview the dataset
df.head()
