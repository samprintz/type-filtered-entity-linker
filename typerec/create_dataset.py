import logging
import os
from tqdm import tqdm

from config import TypeRecDatasetConfig
from inout import dataset
from inout.wikidata import Wikidata
from typerec import types


# Model and training settings
settings = {
    'dataset_train' : 'train', # train/test/dev
    'dataset_part' : 'full', # small/medium/full
    'detailed_types' : False # True/False
    }

# Create config
_config = TypeRecDatasetConfig(settings)

# Logging settings
logging.basicConfig(level=_config.log_level, format=_config.log_format,
        handlers=[logging.FileHandler(_config.log_path),
        logging.StreamHandler()])
_logger = logging.getLogger()


# Wikidata adapter
_wikidata = Wikidata(_config.dirs['type_cache'], _config.dirs['subclass_cache'])

# Type superclass map
type_superclass_map = types.get_entity_type_superclass_map(types.type_dict.keys())


def create_typerec_dataset(data_raw):
    """
    Generates the Wikidata-TypeRec dataset.
    Read the raw JSON dataset and create for each line one sample by augmenting
    it with the type of the correct entity.

    The dataset is for single-label prediction, i.e. for predicting one type.
    """

    _logger.info(f'Generating Wikidata-TypeRec dataset ({len(data_raw)} lines)...')
    data = []
    line_count = 0
    sample_count = 0
    sample_count_failed = 0

    for line in tqdm(data_raw):
        line_count += 1

        try:
            # Only for the correct Wikidata item
            sample = process_data_row(line['text'], line['string'],
                    line['correct_id']) # don't use line['wrong_id']
            data.append(sample)
            sample_count += 1
        except Exception as e:
            _logger.info(str(e))
            sample_count_failed += 1

    _logger.info(f'Created {sample_count} samples from {line_count} lines ' \
            f'(skipped {sample_count_failed} without type)')

    return data


def process_data_row(text, string, item_id):
    """
    Convert one line of the JSON file like described in convert_data().
    """
    sample = {}

    sample['text'] = text.strip()
    sample['item_name'] = string
    sample['item_id'] = item_id
    sample['item_type'] = None # set by add_entity_types()
    sample['item_types'] = [] # set by add_entity_types()
    sample['item_types_detailed'] = [] # set by add_entity_types()

    # Augment the sample with the detailed types and the supertypes
    sample = add_entity_types(sample)

    return sample


def add_entity_types(sample):
    """
    Augment a sample with two lists:
    1. List of entity types explicitly given in statements in Wikidata.
    2. List of derived high-level entity types (derived using Wikidata's
    ontology). Currently its only one such supertype.
    """

    item_id = sample['item_id']

    # Request the entity types explicitly given in statements in Wikidata
    entity_types_detailed = _wikidata.get_types_of_item(item_id)

    # If entity has no types, skip the row
    if len(entity_types_detailed) is 0:
        raise ValueError(f'Item {item_id}: Has no type in Wikidata')

    # Add the types to the sample
    sample['item_types_detailed'] = entity_types_detailed

    # Derive high-level types from Wikidata
    all_supertypes = []
    for entity_type in entity_types_detailed:
        try:
            supertypes = types.get_type_superclass(type_superclass_map, entity_type['id'])
            for supertype in supertypes:
                if supertype not in all_supertypes: # avoid duplicates
                    all_supertypes.append(supertype)
        except KeyError:
            continue


    # If no supertype matched, add default supertype
    if len(all_supertypes) is 0:
        sample['item_type'] = types.default_supertype
        sample['item_types'] = [types.default_supertype] # for analysis
        _logger.info(f'Item {item_id}: Entity type {entity_type["id"]} ' \
                f'does not match any supertype, adding default type ' \
                f'({types.default_supertype})')
    else:
        # Use the first type as correct type
        sample['item_type'] = all_supertypes[0]
        sample['item_types'] = all_supertypes # for analysis
        type_label = types.get_type_label(sample['item_type'])
        if len(all_supertypes) is 1:
            _logger.info(f'Item {item_id}: Found {len(all_supertypes)} ' \
                    f'supertype, adding it ({type_label})')
        else:
            omitted = [types.get_type_label(t) for t in all_supertypes[1:]]
            _logger.info(f'Item {item_id}: Found {len(all_supertypes)} ' \
                    f'supertypes, adding first ({type_label}), ommiting ' \
                    f'{omitted}')

    return sample


def analyze_type_frequency(data):
    """
    Analyze the types in the Wikidata-TypeRec dataset.
    """
    _logger.info(f'Analyze the types in Wikidata-TypeRec ' \
            f'dataset ({len(data)} lines)...')

    # Create dictionary to count types
    item_type_counts = {}

    # Counter to count the average types per item
    item_types_counter = 0

    # How many items accidentially have aone of the supertypes directly as type
    items_with_supertype = 0

    for line in tqdm(data):
        item_types = line['item_types_detailed']
        # Increase counter for each type of the item in this data row
        for item_type in item_types:
            item_types_counter += 1
            if item_type['id'] in item_type_counts:
                item_type_counts[item_type['id']] += 1
            else:
                item_type_counts[item_type['id']] = 1
            if item_type['id'] in types.type_dict.keys():
                items_with_supertype += 1

    avg_types_per_item = item_types_counter / len(data)
    percentage_items_with_supertype = items_with_supertype / len(data)

    _logger.info('')
    _logger.info('=== Statistics ===')
    _logger.info(f'Number of different (low-level) types: {len(item_type_counts)}')
    _logger.info(f'Average number (low-level) types per item: {avg_types_per_item}')
    _logger.info(f'Items that accidentially have a supertype directly as type: {items_with_supertype} ({percentage_items_with_supertype})')


def analyze_supertype_frequency(data):
    """
    Analyze the frequency of high-level types in the Wikidata-TypeRec dataset.
    """
    _logger.info(f'Count frequency of types in Wikidata-TypeRec ' \
            f'dataset ({len(data)} lines)...')

    # Create dictionary to count high-level types (each initialized with 0)
    item_type_counts = dict.fromkeys(types.type_dict.keys(), 0)

    for line in tqdm(data):
        item_types = line['item_types']
        # Increase counter for each high-level type of the item in this data row
        for item_type in item_types:
            item_type_counts[item_type] += 1

    _logger.info('')
    _logger.info('=== Frequencies ===')
    max_label_length = len(max(types.type_dict.values(), key=len))
    for item_type, counter in item_type_counts.items():
        ljust_type_label = types.get_type_label(item_type).ljust(
                max_label_length + 1, " ")
        _logger.info(f'{ljust_type_label}: {counter}')

    return item_type_counts


def analyze_supertype_probability_distribution(dataset_train, dataset_part,
            data):
    """
    Analyze the probability distribution of high-level types in Wikidata-
    TypeRec dataset.
    """
    _logger.info(f'Analyze probability distribution of types in Wikidata-' \
            f'TypeRec dataset ({dataset_train}, {dataset_part}, ' \
            f'{len(data)} lines)...')

    # Analyze the type frequencies
    item_type_counts = analyze_supertype_frequency(data)

    # Sum of all types
    types_total = sum(item_type_counts.values()) # in case of overlapping types, this increases
    #types_total = len(data) # number of items (invariant to overlapping types)

    # Calculate distribution
    _logger.info('')
    _logger.info('=== Probability distribution ===')
    item_type_probability = {}
    max_label_length = len(max(types.type_dict.values(), key=len))
    for item_type, counter in item_type_counts.items():
        probability = counter / types_total
        item_type_probability[item_type] = probability
        ljust_type_label = types.get_type_label(item_type).ljust(
                max_label_length + 1, " ")
        _logger.info(f'{ljust_type_label}: {probability}')

    return item_type_probability


def remove_attribute(data, key):
    """
    Delete from all samples items with a given key.
    """
    for sample in data:
        del sample[key]
    return data



def main():
    #dataset_train = _config.dataset_train
    #dataset_part = _config.dataset_part

    # Generate all datasets for all sizes
    for dataset_train in ['all']:
    #for dataset_train in ['train', 'test', 'dev']:
        for dataset_part in ['full']: # ['small', 'medium', 'full']:

            # Load data
            data_raw = dataset.get_wikidata_disamb_dataset(
                    _config.dirs['wikidata_disamb'], dataset_train, dataset_part)

            # Create dataset
            typerec_dataset = create_typerec_dataset(data_raw)

            # Analyze data
            analyze_type_frequency(typerec_dataset)
            analyze_supertype_probability_distribution(dataset_train,
                    dataset_part, typerec_dataset)

            # Remove attributes
            typerec_dataset = remove_attribute(typerec_dataset, 'item_types')
            if not _config.detailed_types:
                typerec_dataset = remove_attribute(typerec_dataset, 'item_types_detailed')

            # Write data to file
            if _config.detailed_types:
                dataset.write_wikidata_typerec_detailed_dataset(
                        _config.dirs['wikidata_typerec'], typerec_dataset,
                        dataset_train, dataset_part)
            else:
                dataset.write_wikidata_typerec_dataset(
                        _config.dirs['wikidata_typerec'], typerec_dataset,
                        dataset_train, dataset_part)


if __name__ == '__main__':
    main()
