"""Data generation to be used in tests"""

import json
import os
import pathlib
from typing import Any, Dict, List

import numpy as np
from scipy import sparse

from linkedin.learner.ds.feature import Feature
from linkedin.learner.ds.indexed_dataset import IndexedDataset
from linkedin.learner.ds.record import TrainingRecord

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
    """Read records from a file with one ntv json-formatted record per line.

    :param num_examples: How many records to load? -1 means no limit.
    :param data_path: Path to data file.
    :return: List of loaded records.
    """
    with open(data_path, "rb") as file:
        if num_examples > 0:
            records = [json.loads(next(file)) for _ in range(num_examples)]
        else:
            records = [json.loads(line) for line in file]
    return records


def read_model(file):
    """Read an index-domain model file with one coefficient per line.

    :param file: Path to model file.
    :returns: NDArray of coefficients.
    """
    with open(file, "r") as file:
        theta = [float(line) for line in file.readlines()]
    return np.array(theta)


def read_tsv_data(file):
    """"Read a tab-separated value file of index-domain data, with one record per line.

    :param file: Path to data file.
    :returns: IndexedDataset representing the loaded data.
    """
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

        data = sparse.coo_matrix((values, (rows, cols)), shape=(num_examples, 1 + num_features), dtype=np.float64).tocsc()

        offsets = np.array(offsets, dtype=np.float64)
        weights = np.array(weights, dtype=np.float64)
        labels = np.array(labels)

    return IndexedDataset(data, labels, weights, offsets)


def read_movie_lens_data():
    """Read the movie lens data for testing.

    :returns: Movie Lens training and test datasets.
    """
    theta = read_model(MOVIE_LENS_INITIAL_MODEL_PATH)
    training_data = read_tsv_data(MOVIE_LENS_TRAINING_DATA_PATH)
    test_data = read_tsv_data(MOVIE_LENS_TEST_DATA_PATH)

    training_data.offsets = training_data.X * theta
    test_data.offsets = test_data.X * theta

    return training_data, test_data


def read_expected_movie_lens_training_result():
    """Read the expected movie lens training result (the trained model).

    :returns: NDArray representing the loaded model.
    """
    return read_model(MOVIE_LENS_EXPECTED_TRAINED_MODEL_PATH)


def features_from_json(features_json: List[Any], feature_name_translation_map: Dict[str, str] = None) -> List[Feature]:
    """Convert a bag of json-like features to Feature format.

    :param features_json: A list of features as parsed directly from json.
    :param feature_name_translation_map: (Optional) Mapping for using different names internally.
    :return: A list of Features.
    """
    feature_list = []
    for feature in features_json:
        name, term, value = feature[FEATURE_NAME], feature[FEATURE_TERM], feature[FEATURE_VALUE]
        if not feature_name_translation_map:
            # use all features as they are
            feature_list.append(Feature(name, term, value))
        elif name in feature_name_translation_map:
            # use a subset of features, possibly translated to a different name
            # otherwise we drop this feature and don't use it in retraining
            translated_feature_name = feature_name_translation_map[name]
            feature_list.append(Feature(translated_feature_name, term, value))
    return feature_list


def training_record_from_json(record_json) -> TrainingRecord:
    """Convert a json-like record in TrainingRecord format.

    :param record_json: A record as parsed directly from json.
    :return: A TrainingRecord object.
    """
    features_json = record_json[FEATURES]
    features = features_from_json(features_json)
    return TrainingRecord(label=record_json[OFFLINE_RESPONSE], weight=record_json[WEIGHT], offset=record_json[OFFSET], features=features)
