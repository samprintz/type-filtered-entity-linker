import logging
import spacy # TODO To analyze the document length; remove again

from el.mention_detector import SpacyMentionDetectorSm, SpacyMentionDetectorTrf
from el.candidate_generator import WikidataSparqlCandidateGenerator
from el.filter import SpaCyTypeFilter, BERTTypeFilter
from el.entity_disambiguator import EntityDisambiguator

class EntityLinker:

    def __init__(self, config):
        self._logger = logging.getLogger(__name__)
        self._config = config
        self._mention_detector = None
        self._candidate_generator = None
        self._candidate_filter = None
        self._entity_disambiguator = None

        # TODO To analyze the document length; remove again
        self._nlp = spacy.load("en_core_web_sm")


    def process(self, doc):
        doc_with_mentions = self.detect_mentions(doc)
        doc_with_candidates = self.generate_candidates(doc_with_mentions)
        # Apply type-filter
        if self._config.filter:
            doc_with_candidates = self.filter_candidates(doc_with_candidates)
        # Limit candidates
        if self._config.candidates_limit:
            doc_with_candidates = self.limit_candidates(doc_with_candidates)
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

        # TODO To analyze the document length; remove again
        spacy_doc = self._nlp(doc_with_mentions['text'])
        spacy_doc_no_punctuation = [n.lemma_ for n in spacy_doc if not n.is_punct]
        self._logger.info(f'Document length: {len(spacy_doc_no_punctuation)}')

        doc_with_candidates = self.generate_candidates(doc_with_mentions)
        # Apply type-filter
        if self._config.filter:
            doc_with_candidates = self.filter_candidates(doc_with_candidates)
        # Limit candidates
        if self._config.candidates_limit:
            doc_with_candidates = self.limit_candidates(doc_with_candidates)
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
            self._candidate_generator = WikidataSparqlCandidateGenerator()
        self._candidate_generator.process(doc)
        return doc


    def filter_candidates(self, doc):
        self.__print_step_heading('Type Filter')
        if self._candidate_filter is None:
            if self._config.filter == 'spacy':
                self._candidate_filter = SpaCyTypeFilter(self._config)
            elif self._config.filter == 'bert':
                self._candidate_filter = BERTTypeFilter(self._config, self._config.filter_model_path)
        self._candidate_filter.process(doc)
        return doc


    def limit_candidates(self, doc):
        self.__print_step_heading('Limit Candidates')
        limit = self._config.candidates_limit
        limit_applied = False
        for mention in doc['mentions']:
            if len(mention['candidates']) > limit:
                self._logger.info(f'Limit candidates from ' \
                        f'{len(mention["candidates"])} to {limit} for ' \
                        f'"{mention["sf"]}"')
                limit_applied = True
                mention['candidates'] = mention['candidates'][:limit]
        if not limit_applied:
            self._logger.info(f'The size of all candidate entity sets is ' \
                    f'smaller than the limit ({limit})')
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

