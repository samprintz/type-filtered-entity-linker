import logging
from tqdm import tqdm

from config import Config
from inout.wikidata import Wikidata


_logger = logging.getLogger()
_wikidata = Wikidata(Config.dirs['type_cache'], Config.dirs['subclass_cache'])


# List of high-level entity types
type_list = [
        'http://www.wikidata.org/entity/Q215627', # person
        'http://www.wikidata.org/entity/Q163875', # cardinal
        'http://www.wikidata.org/entity/Q838948', # work of art
        'http://www.wikidata.org/entity/Q13442814', # article in scholarly journal
        'http://www.wikidata.org/entity/Q571', # book
        'http://www.wikidata.org/entity/Q618123', # geographical feature
        'http://www.wikidata.org/entity/Q43229', # organization
        'http://www.wikidata.org/entity/Q811979', # architectural structure
        #'http://www.wikidata.org/entity/Q7187', # gene
        #'http://www.wikidata.org/entity/Q11173', # chemical compound
        #'http://www.wikidata.org/entity/Q6999', # astronomical object
        'http://www.wikidata.org/entity/Q16521', # taxon
        'http://www.wikidata.org/entity/Q1656682', # event
        'http://www.wikidata.org/entity/Q83620' # thoroughfare
]


def get_type_superclass(type_superclass_map, type_url):
    """
    Return all high-level entity types that match with the given entity type.
    """
    return type_superclass_map[type_url]


def get_entity_type_subclass_map(entity_types):
    """
    Returns for a list of (high-level) entity types a map, mapping each entity
    type to all of its subclasses:
    superclass -> subclasses
    """
    _logger.info(f'Requesting entity type subclass map from Wikidata ({len(entity_types)} types)...')

    subclass_map = {}
    for entity_type in entity_types:
        subclasses = _wikidata.get_type_subclasses(entity_type)
        subclass_map[entity_type] = subclasses

    _logger.info(f'Requested entity type subclass map from Wikidata')

    return subclass_map


def get_entity_type_superclass_map(entity_types):
    """
    Return superclass map, mapping
    subclass -> superclass
    """
    _logger.info(f'Creating entity type superclass map...')
    entity_type_subclass_map = get_entity_type_subclass_map(entity_types)
    return reverse_entity_type_subclass_map(entity_type_subclass_map)


def reverse_entity_type_subclass_map(entity_type_subclass_map):
    """
    Reverse the entity subclass map s.t. the mapping is
    subclass -> superclass
    """
    entity_type_superclass_map = {}

    for superclass, subclasses in tqdm(entity_type_subclass_map.items()):
        for subclass in subclasses:
            if subclass['id'] in entity_type_superclass_map:
                entity_type_superclass_map[subclass['id']].append(superclass)
            else:
                entity_type_superclass_map[subclass['id']] = [superclass]

    return entity_type_superclass_map


def get_type_index():
    """
    Create an index s.t. each type gets an ID (e.g. person -> 1,
    organization -> 2, ...
    """
    return dict(enumerate(type_list))


def get_index_of_type(entity_type):
    """
    Return the index of a given entity type.
    """
    return type_list.index(entity_type)


def get_type_by_index(index):
    """
    Return the type of a given index.
    """
    return type_list[index - 1] # TODO re-train model with new indices
