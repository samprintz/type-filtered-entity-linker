import json
import logging
import os

from preprocess import Preprocessor


_logger = logging.getLogger(__name__)


def get_aida_conll_train_dataset():
    return load_dataset('conll2003')['train']


def get_wikidata_disamb_dataset(dataset_dir, train, part):
    """
    Read Wikidata-Disamb dataset.
    """
    if not part or part is 'full':
        _logger.info(f'Reading {train} data...')
        dataset_path = os.path.join(dataset_dir, f'wikidata-disambig-{train}.json')
    else:
        _logger.info(f'Reading {train} data ({part})...')
        dataset_path = os.path.join(dataset_dir, f'wikidata-disambig-{train}.{part}.json')
    with open(dataset_path, encoding='utf8') as f:
        json_data = json.load(f)
    _logger.info(f'Read {len(json_data)} lines')
    return json_data


def get_wikidata_typerec_detailed_dataset(dataset_dir, train, part):
    """
    Read Wikidata-TypeRec-detailed dataset.
    """
    dataset_filename = f'wikidata-typerec-detailed-{train}.{part}.json'
    _logger.info(f'Reading {train} data ({part})...')
    dataset_path = os.path.join(dataset_dir, dataset_filename)
    with open(dataset_path, encoding='utf8') as f:
        json_data = json.load(f)
    _logger.info(f'Read {len(json_data)} lines')
    return json_data


def write_wikidata_typerec_detailed_dataset(dataset_dir, dataset, train, part):
    """
    Write Wikidata-TypeRec-detailed dataset to JSON file.
    """
    dataset_filename = f'wikidata-typerec-detailed-{train}.{part}.json'
    _logger.info(f'Writing {len(dataset)} lines {dataset_filename}')
    dataset_path = os.path.join(dataset_dir, dataset_filename)
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    _logger.info(f'Done')


def get_wikidata_typerec_dataset(dataset_dir, train, part):
    """
    Read Wikidata-TypeRec dataset.
    """
    dataset_filename = f'wikidata-typerec-{train}.{part}.json'
    _logger.info(f'Reading {train} data ({part})...')
    dataset_path = os.path.join(dataset_dir, dataset_filename)
    with open(dataset_path, encoding='utf8') as f:
        json_data = json.load(f)
    _logger.info(f'Read {len(json_data)} lines')
    return json_data


def write_wikidata_typerec_dataset(dataset_dir, dataset, train, part):
    """
    Write Wikidata-TypeRec dataset to JSON file.
    """
    dataset_filename = f'wikidata-typerec-{train}.{part}.json'
    _logger.info(f'Writing {len(dataset)} lines {dataset_filename}')
    dataset_path = os.path.join(dataset_dir, dataset_filename)
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    _logger.info(f'Done')


def load_typerec_train_datasets(dataset_dir, dataset_name, dataset_partial, features):
    """
    Loads the datasets for training i.e. the train and dev dataset.
    Loading includes reading the dataset from the file and preprocessing it.
    Features are the features the are actually required and should be reshaped
    in reshape().
    """
    datasets = []
    dataset_parts = ['train', 'dev']

    # Do for the train and the dev dataset
    for dataset_part in dataset_parts:
        _logger.info(f'=== Load {dataset_part} dataset ===')

        # Load data
        data_raw = get_wikidata_typerec_dataset(dataset_dir,
                dataset_part, dataset_partial)

        # Preprocess data
        preprocessor = Preprocessor()
        data_pre = preprocessor.prepare_typerec_dataset(data_raw)
        dataset = preprocessor.reshape_typerec_dataset(data_pre, features)
        datasets.append(dataset)

    return datasets


def load_typerec_test_dataset(dataset_dir, dataset_name, dataset_partial, features):
    """
    Loads the dataset for testing.
    Loading includes reading the dataset from the file and preprocessing it.
    Features are the features the are actually required and should be reshaped
    in reshape().
    """
    dataset_part = 'test'
    _logger.info(f'=== Load {dataset_part} dataset ===')

    # Load data
    data_raw = get_wikidata_typerec_dataset(dataset_dir,
            dataset_part, dataset_partial)

    # Preprocess data
    preprocessor = Preprocessor()
    data_pre = preprocessor.prepare_typerec_dataset(data_raw)
    dataset = preprocessor.reshape_typerec_dataset(data_pre, features)

    return dataset
