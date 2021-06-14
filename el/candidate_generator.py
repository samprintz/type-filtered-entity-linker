import logging

from el.filter import TypeFilter
from inout.wikidata import Wikidata

class WikidataSparqlCandidateGenerator:

    def __init__(self, use_filter=False, type_cache_dir=None, limit=None):
        self._logger = logging.getLogger(__name__)
        self._wikidata = Wikidata(type_cache_dir)
        self._use_filter = use_filter
        self._limit = limit

        if self._use_filter:
            self._filter = TypeFilter(self._wikidata)


    def process(self, doc):
        for mention in doc['mentions']:
            mention['candidates'] = []
            candidate_item_ids = self._wikidata.get_items_by_label(mention['sf'])
            for candidate_item_id in candidate_item_ids:
                candidate = {'item_id': candidate_item_id}
                mention['candidates'].append(candidate)
            self._logger.info(f'Generated {len(mention["candidates"])} candidates for mention "{mention["sf"]}"')

        # Filter by NER type
        if self._use_filter:
            doc = self._filter.process(doc)

        # Limit results
        if self._limit:
            for mention in doc['mentions']:
                if len(mention['candidates']) > self._limit:
                    self._logger.info(f'Limit candidates from {len(mention["candidates"])} to {self._limit} for "{mention["sf"]}"')
                    mention['candidates'] = mention['candidates'][:self._limit]

        return doc
