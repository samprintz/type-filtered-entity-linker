import datetime
import logging
import os
import random
from tqdm import tqdm

from inout import dataset
from inout.wikidata import Wikidata
import analyze_typerec_positives_dataset


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


def generate_typerec_dataset(data_raw):
    """
    Generates the Wikidata-TypeRec dataset by augmenting the Wikidata-TypeRec-
    Positives dataset with negative samples.
    The dataset is for single-label prediction, i.e. for predicting one type.
    """

    _logger.info(f'Generating Wikidata-TypeRec dataset ({len(data_raw)} lines)...')
    data = []
    row_count = 0
    sample_count = 0
    sample_count_failed = 0

    # Get frequencies of high-level types in whole dataset
    item_type_probabilities = analyze_typerec_positives_dataset.get_type_probability_distribution(data_raw)

    for row in data_raw:
        row_count += 1

        # If row has no type, skip it
        # TODO Introduce neutral type "Other"
        if 'item_types' not in row or len(row['item_types']) is 0:
            sample_count_failed += 1
            continue

        # Create a sample
        sample = {
            'text' : row['text'].strip(),
            'item_name' : row['item_name'],
            'item_id' : row['item_id']
            }

        # Use the first type as correct type
        correct_type = row['item_types'][0]

        # Get one wrong type randomly
        wrong_type = get_wrong_type(row['item_types'], item_type_probabilities)

        # Add the sample a second time with one correct types
        correct_sample = sample.copy() # deep copy
        correct_sample['item_type'] = correct_type
        correct_sample['answer'] = True
        data.append(correct_sample)
        sample_count += 1

        # Add the sample once with a randomly drawn wrong type
        wrong_sample = sample.copy() # deep copy
        wrong_sample['item_type'] = wrong_type
        wrong_sample['answer'] = False
        data.append(wrong_sample)
        sample_count += 1

    _logger.info(f'Created {sample_count} samples from {row_count} rows (skipped {sample_count_failed} without type)')

    return data


def get_wrong_type(item_types, item_type_probabilities):
    """
    Choose one type from the list of high-level types that is not in the given
    list of item_types. Choose it by using the probability distribution of the
    high-level types.
    """

    # Set of all types minus the set of the correct types
    #wrong_types = item_type_probabilities.keys().difference(item_types)

    # Set probabilities of correct types to 0
    for item_type in item_types:
        item_type_probabilities[item_type] = 0.0

    # Choose one by the probability distribution
    wrong_type_list = random.choices(list(item_type_probabilities.keys()), weights=item_type_probabilities.values())
    return wrong_type_list[0]


def main():
    # Specify dataset
    dataset_train = 'dev' # train/test/dev
    dataset_part = 'small' # small/medium/full

    # Load data
    data_raw = dataset.get_wikidata_typerec_positives_dataset(dirs['wikidata_typerec'],
            dataset_train, dataset_part)

    # Analyze types
    data = generate_typerec_dataset(data_raw)

    # Write dataset to file
    dataset.write_wikidata_typerec_dataset(dirs['wikidata_typerec'],
            data, dataset_train, dataset_part)


if __name__ == '__main__':
    main()

