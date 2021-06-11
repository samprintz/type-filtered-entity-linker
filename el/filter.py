import logging

class TypeFilter:

    def __init__(self, wikidata):
        self._logger = logging.getLogger(__name__)
        self._wikidata = wikidata

        # Types see https://spacy.io/models/en#en_core_web_sm
        # Wikidata type manually identified
        # TODO add the other URLs (from https://lov.linkeddata.es/dataset/lov or directly Wikidata?)
        self._ner_type_to_wikidata_map = {
                'CARDINAL': 'https://www.wikidata.org/wiki/Q163875',
                'DATE': 'TODO',
                'EVENT': 'TODO',
                'FAC': 'TODO',
                'GPE': 'TODO',
                'LANGUAGE': 'TODO',
                'LAW': 'TODO',
                'LOC': 'http://rdf.geospecies.org/ont/geospecies#Location',
                'MONEY': 'TODO',
                'NORP': 'http://www.wikidata.org/entity/Q41710', # Nationalities or religious or political groups
                'ORDINAL': 'http://www.wikidata.org/entity/Q191780',
                'ORG': 'http://xmlns.com/foaf/0.1/Organization',
                'PERCENT': 'TODO',
                'PERSON': 'http://xmlns.com/foaf/0.1/Person',
                'PRODUCT': 'TODO',
                'QUANTITY': 'TODO',
                'TIME': 'TODO',
                'WORK_OF_ART': 'TODO'
            }


    def process(self, doc):
        for mention in doc['mentions']:
            mention['filtered_candidates'] = []

            # If the mention has no candidates
            if not mention['candidates']:
                mention['unfiltered_candidates'] = mention['candidates']
                self._logger.info(f'No candidates for {mention["sf"]}, skipping')


            mention['filtered_candidates'] = []
            for candidate in mention['candidates']:
                candidate['rdf_types'] = self._wikidata.get_types_of_item(candidate['item_id'])

                # Is any RDF type related to a NER type
                for rdf_type in candidate['rdf_types']:
                    # Is the RDF type in the dictionary? If so, what is the key (the NER type)?
                    for ner_type in self._ner_type_to_wikidata_map.keys():
                        if rdf_type in self._ner_type_to_wikidata_map[ner_type]:
                            candidate['ner_type'] = ner_type
                            break

                # Remove the candidate if it has the wrong NER type
                if not candidate["rdf_types"]: # empty
                    self._logger.info(f'Removed candidate "{candidate["item_id"]}" having no NER type')
                elif 'ner_type' in candidate and candidate['ner_type'] == mention['ner_type']:
                    mention['filtered_candidates'].append(candidate)
                    self._logger.info(f'Added candidate "{candidate["item_id"]}" with correct NER type {candidate["ner_type"]}')
                else:
                    self._logger.info(f'Removed candidate "{candidate["item_id"]}" with wrong NER types: {candidate["rdf_types"]}')

            mention['unfiltered_candidates'] = mention['candidates']
            mention['candidates'] = mention['filtered_candidates']

            count_filtered = len(mention["filtered_candidates"])
            count_removed = len(mention["unfiltered_candidates"]) - count_filtered

            self._logger.info(f'Filtered to {count_filtered} candidates for mention "{mention["sf"]}" (removed {count_removed})')

        return doc