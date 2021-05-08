import logging

from el.mention_detector import SpacyMentionDetector
from el.candidate_generator import WikidataSparqlCandidateGenerator
from el.entity_disambiguator import EntityDisambiguator

class EntityLinker:

    def __init__(self, config):
        self._logger = logging.getLogger(__name__)
        self._config = config


    def process(self, doc):
        doc_with_mentions = self.detect_mentions(doc)
        doc_with_candidates = self.generate_candidates(doc_with_mentions)
        doc_with_entities = self.disambiguate_entities(doc_with_candidates)
        return doc_with_mentions, doc_with_candidates, doc_with_entities


    def detect_mentions(self, doc):
        self.__print_step_heading('Mention Detection')
        mention_detector = SpacyMentionDetector()
        mention_detector.process(doc)
        return doc


    def generate_candidates(self, doc):
        self.__print_step_heading('Candidate Generation')
        candidate_generator = WikidataSparqlCandidateGenerator()
        candidate_generator.process(doc)
        return doc


    def disambiguate_entities(self, doc):
        self.__print_step_heading('Entity Disambiguation')
        entity_disambiguator = EntityDisambiguator(self._config['model_path'])
        entity_disambiguator.process(doc)
        return doc


    def __print_step_heading(self, step_heading):
        self._logger.info('')
        self._logger.info(f'=== {step_heading} ===')

