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
                sample_raw = [doc['text'], mention['sf'], candidate]
                try:
                    sample_pre = self._preprocessor.prepare_sample(sample_raw)
                except ValueError as e: # skip candidate when there is no embedding found
                    self._logger.info(str(e))
                    continue

                #import pudb; pu.db

                sample = self._preprocessor.reshape_dataset([sample_pre], for_training=False)

                # matching score prediction
                matching_score = self._model.predict(sample)
                self._logger.info(f'Score: "{sample_pre["item_name"]}" vs. "{sample_pre["item_id"]}": {matching_score}')
                candidates.append((candidate, matching_score))

            # Winning entity to which the mention likely refers; mention is disambiguated now
            entity = self.__get_best_candidate(candidates)
            mention['entity'] = entity

        return doc


    def __get_best_candidate(self, candidates):
        # TODO
        pass
