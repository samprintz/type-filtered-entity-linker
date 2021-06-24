import logging

from el.mention_detector import SpacyMentionDetectorSm, SpacyMentionDetectorTrf
from el.candidate_generator import WikidataSparqlCandidateGenerator
from el.filter import NERTypeFilter, BERTTypeFilter
from el.entity_disambiguator import EntityDisambiguator

class EntityLinker:

    def __init__(self, config):
        self._logger = logging.getLogger(__name__)
        self._config = config
        self._mention_detector = None
        self._candidate_generator = None
        self._candidate_filter = None
        self._entity_disambiguator = None


    def process(self, doc):
        doc_with_mentions = self.detect_mentions(doc)
        doc_with_candidates = self.generate_candidates(doc_with_mentions)
        if self._config.filter:
            doc_with_candidates = self.filter_candidates(doc_with_candidates)
        doc_with_entities = self.disambiguate_entities(doc_with_candidates)
        return doc_with_mentions, doc_with_candidates, doc_with_entities


    """
    GERBIL A2KB task
    Input: Document
    """
    def a2kb(self, doc):
        return process(doc)


    """
    GERBIL D2KB task
    Input: Document with already marked entities
    https://github.com/dice-group/gerbil/wiki/D2KB
    """
    def d2kb(self, doc_with_mentions):
        doc_with_candidates = self.generate_candidates(doc_with_mentions)
        if self.filter:
            doc_with_candidates = self.filter_candidates(doc_with_candidates)
        doc_with_entities = self.disambiguate_entities(doc_with_candidates)
        return doc_with_candidates, doc_with_entities


    def detect_mentions(self, doc):
        self.__print_step_heading('Mention Detection')
        if self._mention_detector is None:
            self._mention_detector = SpacyMentionDetectorTrf()
        self._mention_detector.process(doc)
        return doc


    def generate_candidates(self, doc):
        self.__print_step_heading('Candidate Generation')
        if self._candidate_generator is None:
            self._candidate_generator = WikidataSparqlCandidateGenerator(
                    self._config.candidates_limit)
        self._candidate_generator.process(doc)
        return doc


    def filter_candidates(self, doc):
        self.__print_step_heading('Type Filter')
        #self._filter = NERTypeFilter()
        self._filter = BERTTypeFilter(self._config, self._config.filter_model_path)
        self._filter.process(doc)
        return doc


    def disambiguate_entities(self, doc):
        self.__print_step_heading('Entity Disambiguation')
        if self._entity_disambiguator is None:
            self._entity_disambiguator = EntityDisambiguator(self._config.ed_model_path,
                    self._config.ed_model_checkpoint_type)
        self._entity_disambiguator.process(doc)
        return doc


    def __print_step_heading(self, step_heading):
        self._logger.info('')
        self._logger.info(f'=== {step_heading} ===')

