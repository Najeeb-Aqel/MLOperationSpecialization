import tensorflow as tf
import tensorflow_data_validation as tfdv
import pandas as pd

from sklearn.model_selection import train_test_split
from util import add_extra_rows

from tensorflow_metadata.proto.v0 import schema_pb2

print('TFDV Version: {}'.format(tfdv.__version__))
print('Tensorflow Version: {}'.format(tf.__version__))

# Read in the training and evaluation datasets
df = pd.read_csv('./data/adult.data', skipinitialspace=True)

# Split the dataset. Do not shuffle for this demo notebook.
train_df, eval_df = train_test_split(df, test_size=0.2, shuffle=False)

# add extra rows
eval_df = add_extra_rows(eval_df)

# Generate training dataset statistics
train_stats = tfdv.generate_statistics_from_dataframe(train_df)

# Visualize training dataset statistics
tfdv.visualize_statistics(train_stats)
