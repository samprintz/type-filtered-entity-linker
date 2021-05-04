import logging

from el.mention_detector import SpacyMentionDetector
from el.candidate_generator import WikidataSparqlCandidateGenerator
from el.entity_disambiguator import EntityDisambiguator

class EntityLinker:

    def __init__(self, model):
        self._model = model
        self._logger = logging.getLogger(__name__)


    def process(self, text):
        mentions = self.detect_mentions(text)
        candidates = self.generate_candidates(text, mentions)
        entities = self.disambiguate_entities(text, candidates)
        return mentions, candidates, entities


    def detect_mentions(self, text):
        self.__print_step_heading('Mention Detection')
        mention_detector = SpacyMentionDetector()
        mentions = mention_detector.process(text)
        self._logger.debug(f'Mentions: {mentions}')
        return mentions


    def generate_candidates(self, text, mentions):
        self.__print_step_heading('Candidate Generation')
        candidate_generator = WikidataSparqlCandidateGenerator()
        candidates = candidate_generator.process(text, mentions)
        self._logger.debug(f'Candidates: {candidates}')
        return candidates

    def disambiguate_entities(self, text, candidates):
        self.__print_step_heading('Entity Disambiguation')
        entity_disambiguator = EntityDisambiguator()
        entities = entity_disambiguator.process(text, candidates)
        self._logger.debug(f'Entities: {entities}')
        return entities


    def __print_step_heading(self, step_heading):
        self._logger.info('')
        self._logger.info(f'=== {step_heading} ===')

