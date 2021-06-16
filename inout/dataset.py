import json
import logging
import os
from datasets import load_dataset


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
