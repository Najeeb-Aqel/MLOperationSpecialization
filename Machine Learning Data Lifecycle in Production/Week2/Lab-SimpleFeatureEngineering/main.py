import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam

from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils

import pprint
import tempfile


def define_dummy_dataset():
    # define sample data
    raw_data = [
        {'x': 1, 'y': 1, 's': 'hello'},
        {'x': 2, 'y': 2, 's': 'world'},
        {'x': 3, 'y': 3, 's': 'hello'}
    ]

    return raw_data


def define_dataset_metadata():
    # define the schema as a DatasetMetadata object
    raw_data_metadata = dataset_metadata.DatasetMetadata(

        # use convenience function to build a Schema protobuf
        schema_utils.schema_from_feature_spec({

            # define a dictionary mapping the keys to its feature spec type
            'y': tf.io.FixedLenFeature([], tf.float32),
            'x': tf.io.FixedLenFeature([], tf.float32),
            's': tf.io.FixedLenFeature([], tf.string),
        }))
    return raw_data_metadata


def main():
    dummy_dataset = define_dummy_dataset()
    raw_data_metadata = define_dataset_metadata()


if __name__ == "__main__":
    main()
