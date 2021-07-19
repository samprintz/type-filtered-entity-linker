import logging
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from config import Config
from inout.wikidata import Wikidata


_logger = logging.getLogger(__name__)
_wikidata = Wikidata(Config.dirs['type_cache'], Config.dirs['subclass_cache'])


# Default supertype for all types that don't match any of the supertypes in the dict above
default_supertype = 'OTHER'

# List of high-level entity types
type_dict = {
        'http://www.wikidata.org/entity/Q215627' : 'person',
        'http://www.wikidata.org/entity/Q163875' : 'cardinal',
        'http://www.wikidata.org/entity/Q838948' : 'work of art', # use the broader type "creative work" instead
        #'http://www.wikidata.org/entity/Q17537576' : 'creative work', # use the broader type "intellectaual work" instead
        #'http://www.wikidata.org/entity/Q15621286' : 'intellectual work', # use the broader type "work" instead
        #'http://www.wikidata.org/entity/Q386724' : 'work',
        'http://www.wikidata.org/entity/Q13442814' : 'article in scholarly journal', # only 0.025 %
        'http://www.wikidata.org/entity/Q571' : 'book', # only 0.047 %
        'http://www.wikidata.org/entity/Q618123' : 'geographical feature',
        'http://www.wikidata.org/entity/Q43229' : 'organization',
        'http://www.wikidata.org/entity/Q811979' : 'architectural structure',
        #'http://www.wikidata.org/entity/Q7187' : 'gene', # 1,004,657 subclasses
        #'http://www.wikidata.org/entity/Q11173' : 'chemical compound', # 877,579 subclasses
        #'http://www.wikidata.org/entity/Q6999' : 'astronomical object',
        'http://www.wikidata.org/entity/Q16521' : 'taxon', # only 0.75 %
        'http://www.wikidata.org/entity/Q1656682' : 'event',
        #'http://www.wikidata.org/entity/Q2424752' : 'product', # new
        'http://www.wikidata.org/entity/Q83620' : 'thoroughfare', # only 0.33 %
        default_supertype : default_supertype
}

# Types see https://spacy.io/models/en#en_core_web_sm
# Wikidata type manually identified
ner_type_to_wikidata_map = {
        'CARDINAL': ['http://www.wikidata.org/entity/Q163875'],
        'DATE': ['http://www.wikidata.org/entity/Q205892'],
        'EVENT': ['http://www.wikidata.org/entity/Q1656682'],
        'FAC': ['http://www.wikidata.org/entity/Q13226383'], # facility
        'GPE': ['http://www.wikidata.org/entity/Q618123'], # http://www.wikidata.org/entity/Q1048835
        'LANGUAGE': ['http://www.wikidata.org/entity/Q34770'],
        'LAW': ['http://www.wikidata.org/entity/Q7748'],
        'LOC': ['http://www.wikidata.org/entity/Q618123'],
        'MONEY': ['http://www.wikidata.org/entity/Q1368'],
        'NORP': ['http://www.wikidata.org/entity/Q41710'], # nationalities or religious or political groups
        'ORDINAL': ['http://www.wikidata.org/entity/Q191780'],
        'ORG': ['http://www.wikidata.org/entity/Q43229'],
        'PERCENT': ['http://www.wikidata.org/entity/Q11229'],
        'PERSON': ['http://www.wikidata.org/entity/Q215627'],
        'PRODUCT': ['http://www.wikidata.org/entity/Q2424752'],
        'QUANTITY': ['http://www.wikidata.org/entity/Q309314'],
        'TIME': ['http://www.wikidata.org/entity/Q11471'],
        'WORK_OF_ART': ['http://www.wikidata.org/entity/Q838948']
}

# Initialize one-hot encoder for entity types
one_hot_encoder = OneHotEncoder(sparse=False)
types_array = np.array(list(type_dict.keys())).reshape(-1, 1)
one_hot_encoder.fit(types_array)
#print(one_hot_encoder.categories_)


def get_type_label(type_url):
    """
    Return the label of a type.
    """
    return type_dict[type_url]


def get_ner_type_label(type_url):
    """
    Return the label of a NER type.
    """
    for ner_label, type_urls in ner_type_to_wikidata_map.items():
        if type_url in type_urls:
            return ner_label


def get_type_id(type_url):
    """
    Return the ID of a type.
    """
    # TODO
    return type_url


def get_type_superclass(type_superclass_map, type_url):
    """
    Return all high-level entity types that match with the given entity type.
    """
    # TODO: Dispense the first argument (by creating a Types object?)
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


#def get_type_index():
    """
    Create an index s.t. each type gets an ID (e.g. person -> 1,
    organization -> 2, ...
    """
    #return dict(enumerate(type_list))


#def get_index_of_type(entity_type):
    """
    Return the index of a given entity type.
    """
    #return type_list.index(entity_type)


#def get_type_by_index(index):
    """
    Return the type of a given index.
    """
    #return type_list[index - 1] # TODO re-train model with new indices


def get_type_by_onehot_vector(onehot_vector):
    """
    Return the type of a given index.
    """
    return one_hot_encoder.inverse_transform([onehot_vector])[0][0]


def get_types_count():
    """
    Return number of different types.
    """
    return len(type_dict.keys())
