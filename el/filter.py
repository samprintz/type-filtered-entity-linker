import logging

from config import Config
from inout.wikidata import Wikidata
from preprocess import Preprocessor
from typerec.model import TypeRecModel
from typerec import types

class BERTTypeFilter:

    def __init__(self, config, model_path):
        self._logger = logging.getLogger(__name__)
        self._preprocessor = Preprocessor()
        self._model = TypeRecModel(config) # TODO load model just in time before predict
        self._model.load(model_path)
        self._wikidata = Wikidata(Config.dirs['type_cache'], Config.dirs['subclass_cache'])
        self._entity_type_superclass_map = types.get_entity_type_superclass_map(types.type_dict.keys()) # subclass -> superclass


    def process(self, doc):
        for mention in doc['mentions']:
            mention['filtered_candidates'] = []

            # If the mention has no candidates
            if not mention['candidates']:
                mention['filtered_candidates'] = mention['candidates']
                self._logger.info(f'No candidates for {mention["sf"]}, skipping')
                continue

            # Preprocessing
            sample_raw = {'text' : doc['text'], 'item_name' : mention['sf']}
            sample_pre = self._preprocessor.prepare_typerec_sample(sample_raw, for_training=False)
            sample = self._preprocessor.reshape_typerec_dataset([sample_pre], features=
                    ['text_and_mention_tokenized', 'text_and_mention_attention_mask'])

            # Entity type prediction
            entity_type = self._model.predict(sample)
            mention['type'] = entity_type
            self._logger.info(f'Got type {mention["type"]} for {mention["sf"]}')

            # Filter candidate entities
            mention['filtered_candidates'] = []
            for candidate in mention['candidates']:

                # Get types of the candidate entity
                candidate['rdf_types'] = self._wikidata.get_types_of_item(candidate['item_id'])

                # If the candidate entity has no type on Wikidata
                if not candidate["rdf_types"]: # empty
                    self._logger.info(f'Removed candidate "{candidate["item_id"]}" having no type')
                    continue

                # For each type of the candidate entity
                candidate_has_correct_type = False
                for rdf_type in candidate['rdf_types']:

                    # get its supertypes
                    try:
                        supertypes = types.get_type_superclass(self._entity_type_superclass_map, rdf_type['id'])
                    except KeyError:
                        self._logger.debug(f'Found no supertypes for type {rdf_type["label"]} ({rdf_type["id"]})')
                        continue

                    # and add the candidate if one of its supertypes matches the (predicted) type of the mention
                    if mention['type'] in supertypes:
                        candidate_has_correct_type = True
                        mention['filtered_candidates'].append(candidate)
                        self._logger.debug(f'Added candidate "{candidate["item_id"]}" with correct type {mention["type"]}')
                        break

                # otherwise don't add it
                if not candidate_has_correct_type:
                    rdf_types_list_string = ', '.join([rdf_type['label'] for rdf_type in candidate['rdf_types']])
                    self._logger.debug(f'Removed candidate "{candidate["item_id"]}" with wrong types: {rdf_types_list_string}' )

            mention['unfiltered_candidates'] = mention['candidates']
            mention['candidates'] = mention['filtered_candidates']

            count_unfiltered = len(mention["unfiltered_candidates"])
            count_filtered = len(mention["filtered_candidates"])
            count_removed = count_unfiltered - count_filtered

            self._logger.info(f'Filtered to {count_filtered} candidates for mention "{mention["sf"]}" (removed {count_removed}/{count_unfiltered})')

        return doc


class NERTypeFilter:

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._wikidata = Wikidata(Config.dirs['type_cache'], Config.dirs['subclass_cache'])

        # Types see https://spacy.io/models/en#en_core_web_sm
        # Wikidata type manually identified
        # TODO add the other URLs (from https://lov.linkeddata.es/dataset/lov or directly Wikidata?)
        self._ner_type_to_wikidata_map = {
                'CARDINAL': ['https://www.wikidata.org/wiki/Q163875'],
                'DATE': ['TODO'],
                'EVENT': ['TODO'],
                'FAC': ['TODO'],
                'GPE': ['TODO'],
                'LANGUAGE': ['TODO'],
                'LAW': ['TODO'],
                'LOC': ['http://rdf.geospecies.org/ont/geospecies#Location'],
                'MONEY': ['TODO'],
                'NORP': ['http://www.wikidata.org/entity/Q41710'], # Nationalities or religious or political groups
                'ORDINAL': ['http://www.wikidata.org/entity/Q191780'],
                'ORG': ['http://xmlns.com/foaf/0.1/Organization'],
                'PERCENT': ['TODO'],
                'PERSON': ['http://xmlns.com/foaf/0.1/Person', 'http://www.wikidata.org/entity/Q5'],
                'PRODUCT': ['TODO'],
                'QUANTITY': ['TODO'],
                'TIME': ['TODO'],
                'WORK_OF_ART': ['TODO', 'http://www.wikidata.org/entity/Q11424']
            }


    def process(self, doc):
        for mention in doc['mentions']:
            mention['filtered_candidates'] = []

            # If the mention has no candidates
            if not mention['candidates']:
                mention['filtered_candidates'] = mention['candidates']
                self._logger.info(f'No candidates for {mention["sf"]}, skipping')
                continue

            # If the mention has no NER type
            if not 'ner_type' in mention:
                mention['filtered_candidates'] = mention['candidates']
                self._logger.info(f'No NER type for {mention["sf"]}, skipping')
                continue

            mention['filtered_candidates'] = []
            for candidate in mention['candidates']:
                candidate['rdf_types'] = self._wikidata.get_types_of_item(candidate['item_id'])

                if not candidate["rdf_types"]: # empty
                    self._logger.info(f'Removed candidate "{candidate["item_id"]}" having no NER type')

                # Is any RDF type related to a NER type?
                else:
                    for rdf_type in candidate['rdf_types']:
                        # Is the RDF type in the dictionary? If so, what is the key (the NER type)?
                        for ner_type in self._ner_type_to_wikidata_map.keys():
                            if rdf_type['id'] in self._ner_type_to_wikidata_map[ner_type]:
                                candidate['ner_type'] = ner_type
                                break

                # Add the candidate if it has the correct NER type
                if 'ner_type' in candidate and candidate['ner_type'] == mention['ner_type']:
                    mention['filtered_candidates'].append(candidate)
                    self._logger.debug(f'Added candidate "{candidate["item_id"]}" with correct NER type {candidate["ner_type"]}')
                else:
                    rdf_types_list_string = ', '.join([rdf_type['label'] for rdf_type in candidate['rdf_types']])
                    self._logger.debug(f'Removed candidate "{candidate["item_id"]}" with wrong NER types: {rdf_types_list_string}' )

            mention['unfiltered_candidates'] = mention['candidates']
            mention['candidates'] = mention['filtered_candidates']

            count_filtered = len(mention["filtered_candidates"])
            count_removed = len(mention["unfiltered_candidates"]) - count_filtered

            self._logger.info(f'Filtered to {count_filtered} candidates for mention "{mention["sf"]}" (removed {count_removed})')

        return doc
