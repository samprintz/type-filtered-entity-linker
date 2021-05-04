import logging

from inout.wikidata import Wikidata

class WikidataSparqlCandidateGenerator:

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._wikidata = Wikidata()

    def process(self, text, mentions):
        candidates = []
        for mention in mentions:
            # TODO create some doc object (reuse from spacy?) and append candidates to that
            candidates.append(self._wikidata.get_items_by_label(mention))
        return candidates
