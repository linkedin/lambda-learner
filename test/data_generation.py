"""Data generation to be used in tests"""

import json
import logging
import os
import pathlib
from typing import Any, Dict, List

import numpy as np
from scipy import sparse

from linkedin.lambdalearnerlib.ds.feature import Feature
from linkedin.lambdalearnerlib.ds.indexed_dataset import IndexedDataset
from linkedin.lambdalearnerlib.ds.record import TrainingRecord

LOG = logging.getLogger(__name__)

MOCK_NTV_DATASET_PATH = os.path.join(pathlib.Path(__file__).parent, "resource/mock-ntv-data-1k.json")

MOVIE_LENS_TRAINING_DATA_PATH = os.path.join(pathlib.Path(__file__).parent, "resource/movie-lense-training-data-u359")
MOVIE_LENS_TEST_DATA_PATH = os.path.join(pathlib.Path(__file__).parent, "resource/movie-lense-test-data-u359")
MOVIE_LENS_INITIAL_MODEL_PATH = os.path.join(pathlib.Path(__file__).parent, "resource/movie-lense-initital-fixed-effect-model")
MOVIE_LENS_EXPECTED_TRAINED_MODEL_PATH = os.path.join(pathlib.Path(__file__).parent, "resource/movie-lense-final-model-u359")

OFFLINE_RESPONSE = "label"
WEIGHT = "weight"
FEATURES = "features"
OFFSET = "offset"

# Features are logged as name, term, value
FEATURE_NAME = "name"
FEATURE_TERM = "term"
FEATURE_VALUE = "value"


def read_ntv_records_from_json(num_examples: int = -1, data_path=MOCK_NTV_DATASET_PATH):
    """
    Read records from a file with one ntv json-formatted record per line.
    :param data_path: path to file
    :return: records
    """
    with open(data_path, "rb") as file:
        if num_examples > 0:
            records = [json.loads(next(file)) for _ in range(num_examples)]
        else:
            records = [json.loads(line) for line in file]
    return records


def read_model(file):
    with open(file, "r") as file:
        theta = [float(line) for line in file.readlines()]
    return np.array(theta)


def read_tsv_data(file):
    with open(file, "r") as file:
        example_list = [line.split("\t") for line in file.readlines()]
        num_examples = len(example_list)

        labels = []
        values = []
        cols = []
        rows = []
        offsets = [0] * num_examples  # temporary value
        weights = [1] * num_examples

        for idx, example in enumerate(example_list):
            timestamp, y, *feature_values = example
            timestamp = int(timestamp)
            y = float(y)
            feature_values = [float(v) for v in feature_values]

            # Add intercept
            values.append(1.0)
            cols.append(0)
            rows.append(idx)

            num_features = len(feature_values)
            feature_idx = range(1, num_features + 1)

            # Add other features
            values.extend(feature_values)
            cols.extend(feature_idx)
            rows.extend([idx] * num_features)

            labels.append(y)

        # Note: Why sparse and why CSC? CSR and CSC are better suited for linear algebra operations than DOK, DIA, LIL, etc.
        # In practice, performance testing has revealed little difference between CSR and CSC representations for the sorts
        # of operations lambda performs. However, sparse is considerably faster than dense for our typical data.
        # See go/ll-matrix-performance.
        data = sparse.coo_matrix((values, (rows, cols)), shape=(num_examples, 1 + num_features), dtype=np.float64).tocsc()

        offsets = np.array(offsets, dtype=np.float64)
        weights = np.array(weights, dtype=np.float64)
        labels = np.array(labels)

    return IndexedDataset(data, labels, weights, offsets)


def read_movie_lens_data():
    theta = read_model(MOVIE_LENS_INITIAL_MODEL_PATH)
    training_data = read_tsv_data(MOVIE_LENS_TRAINING_DATA_PATH)
    test_data = read_tsv_data(MOVIE_LENS_TEST_DATA_PATH)

    training_data.offsets = training_data.X * theta
    test_data.offsets = test_data.X * theta

    return training_data, test_data


def read_expected_movie_lens_training_result():
    return read_model(MOVIE_LENS_EXPECTED_TRAINED_MODEL_PATH)


def populate_features_from_avro_bag(features_avro: List[Any], feature_name_translation_map: Dict[str, str] = None) -> List[Feature]:
    features = []
    for feature_avro in features_avro:
        name, term, value = feature_avro[FEATURE_NAME], feature_avro[FEATURE_TERM], feature_avro[FEATURE_VALUE]
        if not feature_name_translation_map:
            # use all features as they are
            features.append(Feature(name, term, value))
        elif name in feature_name_translation_map:
            # use a subset of features, possibly translated to a different name
            # otherwise we drop this feature and don't use it in retraining
            translated_feature_name = feature_name_translation_map[name]
            features.append(Feature(translated_feature_name, term, value))
    return features


def from_offline_training_example_avro(record) -> TrainingRecord:
    """
    Parse an avro record in the TrainingExample format and construct a python object
    :param record:
    :return: a TrainingRecord object without feature indexing
    """
    features_avro = record[FEATURES]
    features = populate_features_from_avro_bag(features_avro)
    return TrainingRecord(label=record[OFFLINE_RESPONSE], weight=record[WEIGHT], offset=record[OFFSET], features=features)
