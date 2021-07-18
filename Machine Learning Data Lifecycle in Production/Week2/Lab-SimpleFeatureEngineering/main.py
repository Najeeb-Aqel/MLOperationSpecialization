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


def main():
    dummy_dataset = define_dummy_dataset()


if __name__ == "__main__":
    main()