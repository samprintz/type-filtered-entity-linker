import logging

class TypeFilter:

    def __init__(self, wikidata):
        self._logger = logging.getLogger(__name__)
        self._wikidata = wikidata

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
