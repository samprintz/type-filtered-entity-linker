import logging

from el.model import ELModel

class EntityDisambiguator:

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        # TODO initialize the neural model

    def process(self, text, candidates):
        entities = []
        for mention in candidates:
            # TODO preprocessing (here or already earlier?)
            # TODO disambiguation: model.predict()

            # TODO create some doc object (reuse from spacy?) and append candidates to that
            entities.append(candidates)
        return entities
