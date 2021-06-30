import logging

from preprocess import Preprocessor
from disamb.model import EDModel

class EntityDisambiguator:

    def __init__(self, model_path, model_checkpoint_type):
        self._logger = logging.getLogger(__name__)
        self._preprocessor = Preprocessor()
        # TODO load model just right before predict(), s.t. it's not loaded e.g. if there are no candidates
        self._model = EDModel()
        self._model.load(model_path, model_checkpoint_type)

    def process(self, doc):
        # If no mentions, return document
        if len(doc['mentions']) is 0:
            return doc

        # For each mention in the text
        for mention in doc['mentions']:

            # Skip if the mention has no candidates
            if len(mention['candidates']) is 0:
                self._logger.info(f'No candidates for "{mention["sf"]}", skipping')
                mention['entity'] = None
                continue

            # Predict how good of each its candidates matches with it (rank) by using the neural model
            candidates = []
            for candidate in mention['candidates']:

                # Preprocessing
                sample_raw = [doc['text'], mention['sf'], candidate['item_id']]
                try:
                    sample_pre = self._preprocessor.prepare_sample(sample_raw)
                except ValueError as e: # skip candidate when there is no embedding found
                    self._logger.info(str(e))
                    continue

                sample = self._preprocessor.reshape_dataset([sample_pre], for_training=False)

                # Predict matching score
                matching_score = self._model.predict(sample)
                self._logger.info(f'Score: "{mention["sf"]}" vs. {candidate["item_id"]}: {matching_score}')
                candidate['score'] = matching_score
                candidates.append(candidate)

            # If all candidates were removed (because they don't have an PBG embedding)
            if len(candidates) == 0:
                # take the first candidate
                entity = mention['candidates'][0]
                entity['score'] = 0.0 # default score
                self._logger.info(f'Not a single candidate had a PyTorch-BigGraph embedding. Could not calculate any matching score. Disambiguate by choosing first candidate:')
                self._logger.info(f'First as best candidate for "{mention["sf"]}": {entity["item_id"]}')

            else:
                # Winning entity to which the mention likely refers; mention is disambiguated now
                entity = self.__get_best_candidate(candidates)
                self._logger.info(f'Best candidate for "{mention["sf"]}": {entity["item_id"]} ({entity["score"]})')

            entity['item_url'] = self.__make_wikidata_url(entity['item_id'])
            mention['entity'] = entity

        return doc


    def __get_best_candidate(self, candidates):
        """
        Iterate the list of candidate entities and return the one with the highest score
        """
        self._logger.debug(f'Find best candidate:')
        best_candidate = None
        for candidate in candidates:
            if best_candidate is None or candidate['score'] > best_candidate['score']:
                self._logger.debug(f'Update best candidate: {candidate["item_id"]} ({candidate["score"]})')
                best_candidate = candidate
        return best_candidate


    def __make_wikidata_url(self, item_id):
        # TODO move to utils.py
        wikidata_url_prefix = 'http://www.wikidata.org/entity/'
        return f'{wikidata_url_prefix}{item_id}'


    def __make_dbpedia_url(self, entity):
        # TODO move to utils.py
        raise NotImplementedError
