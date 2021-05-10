import logging

from preprocess import Preprocessor
from el.model import ELModel

class EntityDisambiguator:

    def __init__(self, model_path):
        self._logger = logging.getLogger(__name__)
        self._preprocessor = Preprocessor()
        self._model = ELModel()
        self._model.load(model_path)

    def process(self, doc):
        # For each mention in the text
        for mention in doc['mentions']:

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

                #import pudb; pu.db

                sample = self._preprocessor.reshape_dataset([sample_pre], for_training=False)

                # matching score prediction
                matching_score = self._model.predict(sample)
                self._logger.info(f'Score: "{mention["sf"]}" vs. {candidate["item_id"]}: {matching_score}')
                candidate['score'] = matching_score
                candidates.append(candidate)

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
        wikidata_url_prefix = 'https://www.wikidata.org/wiki'
        return f'{wikidata_url_prefix}/{item_id}'


    def __make_dbpedia_url(self, entity):
        # TODO move to utils.py
        raise NotImplementedError
