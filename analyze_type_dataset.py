import datetime
from inout import dataset
from inout.wikidata import Wikidata
from generate_type_dataset import _entity_type_superclasses
import logging
import os
from tqdm import tqdm


dirs = {
    'logging' : os.path.join(os.getcwd(), 'log'),
    'models' : os.path.join(os.getcwd(), 'data', 'models'),
    'wikidata_disamb' : os.path.join(os.getcwd(), 'data', 'wikidata_disamb'),
    'wikidata_typerec' : os.path.join(os.getcwd(), 'data', 'wikidata_type_recognition'),
    'type_cache' : os.path.join(os.getcwd(), 'data', 'type_cache'),
    'subclass_cache' : os.path.join(os.getcwd(), 'data', 'subclass_cache')
    }

for path in dirs.values():
    if not os.path.exists(path):
        os.makedirs(path)


# Logging settings
log_level = logging.INFO
log_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_path = os.path.join(dirs['logging'], f'{log_filename}-typerec_analysis.log')
log_format = "%(asctime)s: %(levelname)-1.1s %(name)s:%(lineno)d] %(message)s"

_logger = logging.getLogger()
logging.basicConfig(level=log_level, format=log_format,
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])


def analyze_data(data_raw):
    """
    Analyze the frequency of high-level types.
    """
    _logger.debug(f'Analyze Wikidata TypeRec dataset ({len(data_raw)} lines)...')

    # Create dictionary to count high-level types (each initialized with 0)
    item_type_counts = dict.fromkeys(_entity_type_superclasses, 0)

    for line in tqdm(data_raw):
        item_types = line['item_types']
        # Increase the counter for each high-level type of the item in this data row
        for item_type in item_types:
            item_type_counts[item_type] += 1

    _logger.info('=== Result ===')
    _logger.info('')
    for item_type, counter in item_type_counts.items():
        _logger.info(f'{item_type}: {counter}')


def main():
    # Specify dataset
    dataset_train = 'train' # train/test/dev
    dataset_part = 'small' # small/medium/full

    # Load data
    data_raw = dataset.get_wikidata_typerec_dataset(dirs['wikidata_typerec'],
            dataset_train, dataset_part)

    # Analyze types
    analyze_data(data_raw)


if __name__ == '__main__':
    main()

