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
log_format = "%(asctime)s: %(levelname)-1.1s %(name)s:%(lineno)d] %(message)s"

_logger = logging.getLogger()
logging.basicConfig(level=log_level, format=log_format,
        handlers=[logging.StreamHandler()])


# Wikidata adapter
_wikidata = Wikidata(dirs['type_cache'], dirs['subclass_cache'])

# Mappings
_entity_type_superclass_map = {} # subclass -> superclass


def convert_data(data_raw):
    """
    Read the raw JSON dataset and create for each line one sample for the
    correct Wikidata ID and another for the wrong ID.
    """

    _logger.debug(f'Converting dataset ({len(data_raw)} lines)...')
    data = []
    line_count = 0
    sample_count = 0
    sample_count_failed = 0

    for line in tqdm(data_raw):
        line_count += 1

        # Once for correct Wikidata item
        try:
            sample = convert_data_row(line['text'], line['string'], line['correct_id'])
            data.append(sample)
            sample_count += 1
        except Exception as e:
            _logger.info(str(e))
            sample_count_failed += 1

        # Once for wrong Wikidata item
        try:
            sample = convert_data_row(line['text'], line['string'], line['wrong_id'])
            data.append(sample)
            sample_count += 1
        except Exception as e:
            _logger.info(str(e))
            sample_count_failed += 1

    _logger.debug(f'Prepared {sample_count} samples from {line_count} lines (skipped {sample_count_failed} failed)')

    return data


def convert_data_row(text, string, item_id):
    """
    Convert one line of the JSON file like described in convert_data().
    """
    sample = {}

    sample['text'] = text
    sample['item_name'] = string
    sample['item_id'] = item_id
    sample['item_types'] = []
    sample['item_types_detailed'] = []

    return sample


def augment_data_with_entity_types(data_raw):
    """
    Augment a sample with two lists:
    1. List of entity types explicitly given in statements in Wikidata.
    2. List of derived high-level entity entity types (derived using Wikidata's
    ontology.
    """
    _logger.info(f'Augmenting dataset with entity types ({len(data_raw)} samples)...')
    data = []
    sample_count = 0
    sample_count_failed = 0

    for sample in tqdm(data_raw):
        try:
            sample = augment_sample_with_entity_types(sample)
            data.append(sample)
            sample_count += 1
        except Exception as e:
            _logger.info(str(e))
            sample_count_failed += 1

    _logger.info(f'Augmented {sample_count} samples with entity types (skipped {sample_count_failed} failed samples)')

    return data


def augment_sample_with_entity_types(sample):
    """
    Augment a sample with the two lists described in
    augment_data_with_entity_types().
    """
    item_id = sample['item_id']

    # Request the entity types explicitly given in statements in Wikidata
    entity_types_detailed = _wikidata.get_types_of_item(item_id)
    sample['item_types_detailed'] = entity_types_detailed

    # Derive high-level entity types from Wikidata
    for entity_type in entity_types_detailed:
        try:
            superclasses = get_type_superclass(entity_type['id'])
            for superclass in superclasses:
                if superclass not in sample['item_types']:
                    sample['item_types'].append(superclass)
        except KeyError:
            # TODO Add default class (class "other" or so)
            _logger.debug(f'Item {item_id}: Entity type {entity_type["id"]} does not match any superclass, skipping')

    _logger.debug(f'Item {item_id}: Found {len(sample["item_types"])} high-level types')
    return sample


def main():
    # Get entity subclass map
    global _entity_type_superclass_map # TODO
    _entity_type_superclass_map = types.get_entity_type_superclass_map(types.type_list)

    # Specify dataset
    dataset_train = 'dev' # train/test/dev
    dataset_part = 'small' # small/medium/full

    # Load data
    data_raw = dataset.get_wikidata_disamb_dataset(dirs['wikidata_disamb'],
            dataset_train, dataset_part)

    # Convert data
    data = convert_data(data_raw)
    data_with_types = augment_data_with_entity_types(data)

    # Write data to file
    dataset.write_wikidata_typerec_positives_dataset(dirs['wikidata_typerec'],
            data_with_types, dataset_train, dataset_part)


if __name__ == '__main__':
    main()

