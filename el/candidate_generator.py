import logging

from config import Config
from inout.wikidata import Wikidata

class WikidataSparqlCandidateGenerator:

    def __init__(self, limit=None):
        self._logger = logging.getLogger(__name__)
        self._wikidata = Wikidata(Config.dirs['type_cache'], Config.dirs['subclass_cache'])
        self._limit = limit


    def process(self, doc):
        for mention in doc['mentions']:
            mention['candidates'] = []
            candidate_item_ids = self._wikidata.get_items_by_label(mention['sf'])
            for candidate_item_id in candidate_item_ids:
                candidate = {'item_id': candidate_item_id}
                mention['candidates'].append(candidate)
            self._logger.info(f'Generated {len(mention["candidates"])} candidates for mention "{mention["sf"]}"')

        # Limit results
        if self._limit:
            for mention in doc['mentions']:
                if len(mention['candidates']) > self._limit:
                    self._logger.info(f'Limit candidates from {len(mention["candidates"])} to {self._limit} for "{mention["sf"]}"')
                    mention['candidates'] = mention['candidates'][:self._limit]

        return doc
