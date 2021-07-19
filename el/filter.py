import logging

from config import Config
from inout.wikidata import Wikidata
from preprocess import Preprocessor
from typerec.model import TypeRecModel
from typerec import types

class BERTTypeFilter:

    def __init__(self, config, model_path):
        self._logger = logging.getLogger(__name__)
        self._config = config
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
            self._logger.info(f'Predicting type of mention "{mention["sf"]}"...')
            entity_type = self._model.predict(sample)
            mention['type'] = entity_type
            self._logger.info(f'Predict type [{types.get_type_label(mention["type"])}] ({mention["type"]}) for {mention["sf"]}')

            # If type <OTHER> was predicted for the mention, just add all candidates
            if mention['type'] == types.default_supertype and not self._config.filter_default_type:
                mention['filtered_candidates'] = mention['candidates']
                self._logger.info(f'Add all candidates (predicted default type "{types.default_supertype}")')
                self._logger.info(f'To change this behavior, set settings.filter_default_type = True)')

            # Filter candidate entities
            else:
                mention['filtered_candidates'] = []
                for candidate in mention['candidates']:

                    # Get types of the candidate entity
                    candidate['rdf_types'] = self._wikidata.get_types_of_item(candidate['item_id'])

                    # If the candidate entity has no type on Wikidata
                    if not candidate["rdf_types"]: # empty
                        if self._config.filter_entities_without_type:
                            self._logger.info(f'Removed candidate "{candidate["item_id"]}" having no type')
                            continue
                        else:
                            candidate_has_correct_type = True
                            mention['filtered_candidates'].append(candidate)
                            self._logger.info(f'Added candidate "{candidate["item_id"]}" WITHOUT any type')
                            self._logger.info(f'To change this behavior, set settings.filter_entities_without_type = False)')

                    # For each type of the candidate entity
                    candidate_has_correct_type = False
                    for rdf_type in candidate['rdf_types']:

                        # get its supertypes
                        try:
                            supertypes = types.get_type_superclass(self._entity_type_superclass_map, rdf_type['id'])
                        except KeyError:
                            self._logger.info(f'Found no supertypes for [{rdf_type["label"]}] ({rdf_type["id"]})')
                            continue

                        # and add the candidate if one of its supertypes matches the (predicted) type of the mention
                        if mention['type'] in supertypes:
                            candidate_has_correct_type = True
                            mention['filtered_candidates'].append(candidate)
                            self._logger.info(f'Added candidate "{candidate["item_id"]}" with correct type: {types.get_type_label(mention["type"])}')
                            break

                    # otherwise don't add it
                    if not candidate_has_correct_type:
                        rdf_types_list_string = ', '.join([rdf_type['label'] for rdf_type in candidate['rdf_types']])
                        self._logger.info(f'Removed candidate "{candidate["item_id"]}" with wrong types: {rdf_types_list_string}' )

            mention['unfiltered_candidates'] = mention['candidates']
            mention['candidates'] = mention['filtered_candidates']

            count_unfiltered = len(mention["unfiltered_candidates"])
            count_filtered = len(mention["filtered_candidates"])
            count_removed = count_unfiltered - count_filtered

            self._logger.info(f'Filtered to {count_filtered}/{count_unfiltered} candidates for mention "{mention["sf"]}" (removed {count_removed})')

        return doc


class SpaCyTypeFilter:

    def __init__(self, config):
        self._logger = logging.getLogger(__name__)
        self._config = config
        self._wikidata = Wikidata(Config.dirs['type_cache'], Config.dirs['subclass_cache'])
        # Create supertype map from NER types
        types_list = []
        for entity_types in types.ner_type_to_wikidata_map.values():
            types_list.extend(entity_types)
        self._entity_type_superclass_map = types.get_entity_type_superclass_map(types_list) # subclass -> superclass

    def get_overlapping_mention(self, mention, mentions):
        """
        Given a mention m and a set of mention M, with which mentions from M
        does m overlap?
        Returns only the first match.
        """
        overlapping_mentions = []
        for mention2 in mentions:
            if mention2['start'] >= mention['start'] and mention2['end'] <= mention['end']:
                self._logger.info(f'Found overlapping mention for "{mention["sf"]}"')
                overlapping_mentions.append(mention2)

        # Warning if m overlaps with multiple mentions
        if len(overlapping_mentions) > 1:
            self._logger.warn(f'Mention "{mention["sf"]}" overlaps with multiple mentions, returning only first ({mention2["sf"]})')

        # Warning if m overlaps with no mention
        if len(overlapping_mentions) is 0:
            self._logger.info(f'Mention "{mention["sf"]}" does not overlap with any mention, returning None')
            return None

        return overlapping_mentions[0]


    def get_rdf_type_by_ner_type(self, ner_type):
        """
        Returns the RDF supertype given an NER type.
        """
        return types.ner_type_to_wikidata_map[ner_type]


    def process(self, doc):
        # Get mention types by spaCy
        # spaCy can only do full NER tagging and can't handle pre-marked spans
        # Hence, do full NER tagging with spaCy and use the NER types of those
        # spans that overlap with the pre-marked ones. Where spaCy didn't find
        # a span, use the default entity type (OTHER).
        from el.mention_detector import SpacyMentionDetectorTrf
        mention_detector = SpacyMentionDetectorTrf()
        # Copy doc
        # TODO Make the pipeline methods process() return a new object, not modify the old
        doc2 = {'text' : doc['text']}
        mention_detector.process(doc2)

        for mention in doc['mentions']:
            mention['filtered_candidates'] = []

            # If the mention has no candidates
            if not mention['candidates']:
                mention['filtered_candidates'] = mention['candidates']
                self._logger.info(f'No candidates for "{mention["sf"]}", skipping')
                continue

            # Find overlapping mentions in the spaCy result
            overlapping_mention = self.get_overlapping_mention(mention, doc2['mentions'])

            # If no spaCy mention overlaps, assign default type
            if not overlapping_mention:
                mention['types'] = [types.default_supertype]
            else:
                mention['types'] = self.get_rdf_type_by_ner_type(overlapping_mention['ner_type'])

            # If type <OTHER> was predicted for the mention, just add all candidates
            if mention['types'][0] == types.default_supertype and not self._config.filter_default_type:
                mention['filtered_candidates'] = mention['candidates']
                self._logger.info(f'Add all candidates (predicted default type "{types.default_supertype}")')
                self._logger.info(f'To change this behavior, set settings.filter_default_type = True)')

            # Filter candidate entities
            else:
                mention['filtered_candidates'] = []
                for candidate in mention['candidates']:

                    # Get types of the candidate entity
                    candidate['rdf_types'] = self._wikidata.get_types_of_item(candidate['item_id'])

                    # If the candidate entity has no type on Wikidata
                    if not candidate["rdf_types"]: # empty
                        if self._config.filter_entities_without_type:
                            self._logger.info(f'Removed candidate "{candidate["item_id"]}" having no type')
                            continue
                        else:
                            candidate_has_correct_type = True
                            mention['filtered_candidates'].append(candidate)
                            self._logger.info(f'Added candidate "{candidate["item_id"]}" WITHOUT any type')
                            self._logger.info(f'To change this behavior, set settings.filter_entities_without_type = False)')

                    # For each type of the candidate entity
                    candidate_has_correct_type = False
                    for rdf_type in candidate['rdf_types']:

                        # get its supertypes
                        try:
                            supertypes = types.get_type_superclass(self._entity_type_superclass_map, rdf_type['id'])
                        except KeyError:
                            self._logger.info(f'Found no supertypes for [{rdf_type["label"]}] ({rdf_type["id"]})')
                            continue

                        # and add the candidate if one of its supertypes matches the spaCy type of the mention
                        for mention_type in mention['types']:
                            if mention_type in supertypes:
                                candidate_has_correct_type = True
                                mention['filtered_candidates'].append(candidate)
                                self._logger.info(f'Added candidate "{candidate["item_id"]}" with correct type: {types.get_ner_type_label(mention_type)}')
                                break

                    # otherwise don't add it
                    if not candidate_has_correct_type:
                        rdf_types_list_string = ', '.join([rdf_type['label'] for rdf_type in candidate['rdf_types']])
                        self._logger.info(f'Removed candidate "{candidate["item_id"]}" with wrong types: {rdf_types_list_string}' )

            mention['unfiltered_candidates'] = mention['candidates']
            mention['candidates'] = mention['filtered_candidates']

            count_unfiltered = len(mention["unfiltered_candidates"])
            count_filtered = len(mention["filtered_candidates"])
            count_removed = count_unfiltered - count_filtered

            self._logger.info(f'Filtered to {count_filtered}/{count_unfiltered} candidates for mention "{mention["sf"]}" (removed {count_removed})')

        return doc



class NERTypeFilter:

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._wikidata = Wikidata(Config.dirs['type_cache'], Config.dirs['subclass_cache'])


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
                        for ner_type in types.ner_type_to_wikidata_map.keys():
                            if rdf_type['id'] in types.ner_type_to_wikidata_map[ner_type]:
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
