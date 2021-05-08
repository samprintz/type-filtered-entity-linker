import logging

from inout.wikidata import Wikidata

class WikidataSparqlCandidateGenerator:

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._wikidata = Wikidata()

    def process(self, doc):
        for mention in doc['mentions']:
            candidates = self._wikidata.get_items_by_label(mention['sf'])
            mention['candidates'] = candidates
            self._logger.info(f'Generated {len(mention["candidates"])} candidates for mention "{mention["sf"]}"')
        return doc
