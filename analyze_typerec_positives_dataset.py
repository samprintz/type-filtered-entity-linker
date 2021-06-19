import datetime
import logging
import os
from tqdm import tqdm

from inout import dataset
from inout.wikidata import Wikidata
from typerec import types


dirs = {
    'logging' : os.path.join(os.getcwd(), 'log'),
    'models' : os.path.join(os.getcwd(), 'data', 'models'),
    'wikidata_disamb' : os.path.join(os.getcwd(), 'data', 'wikidata_disamb'),
    'wikidata_typerec' : os.path.join(os.getcwd(), 'data', 'wikidata_typerec'),
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
    Analyze the frequency of high-level types in Wikidata-
    TypeRec-Positives dataset.
    """
    _logger.info(f'Count frequency of types in Wikidata-TypeRec-Positives dataset ({len(data_raw)} lines)...')

    # Create dictionary to count high-level types (each initialized with 0)
    item_type_counts = dict.fromkeys(types.type_list, 0)

    for line in tqdm(data_raw):
        item_types = line['item_types']
        # Increase the counter for each high-level type of the item in this data row
        for item_type in item_types:
            item_type_counts[item_type] += 1

    _logger.info('=== Frequencies ===')
    for item_type, counter in item_type_counts.items():
        _logger.info(f'{item_type}: {counter}')

    return item_type_counts


def get_type_probability_distribution(data_raw):
    """
    Analyze the probability distribution of high-level types in Wikidata-
    TypeRec-Positives dataset.
    """
    _logger.info(f'Analyze probability distribution of types in Wikidata-TypeRec-Positives dataset ({len(data_raw)} lines)...')

    # Analyze the type frequencies
    item_type_counts = analyze_data(data_raw)

    # Sum of all types
    types_total = sum(item_type_counts.values())

    # Calculate distribution
    _logger.info('=== Probability distribution ===')
    item_type_probability = {}
    for item_type, counter in item_type_counts.items():
        probability = counter / types_total
        item_type_probability[item_type] = probability
        _logger.info(f'{item_type}: {probability}')

    return item_type_probability


def main():
    # Specify dataset
    dataset_train = 'test' # train/test/dev
    dataset_part = 'small' # small/medium/full

    # Load data
    data_raw = dataset.get_wikidata_typerec_positives_dataset(dirs['wikidata_typerec'],
            dataset_train, dataset_part)

    # Analyze types
    #analyze_data(data_raw)
    get_type_probability_distribution(data_raw)


if __name__ == '__main__':
    main()

