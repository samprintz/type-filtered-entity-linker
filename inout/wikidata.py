import logging
import requests

class Wikidata:
    _url_sparql = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
    _query_get_by_label = '''
            SELECT DISTINCT ?s {
                ?s ?label "%s"@en .
                ?s ?p ?o .
            }
            '''

    def __init__(self):
        self._logger = logging.getLogger(__name__)


    def get_neighborhood(self, item_id, hops=1):
        pass


    def get_items_by_label(self, label):
        self._logger.debug(f'Get items with label "{label}" from Wikidata SPARQL endpoint')

        items = []

        response = requests.get(self._url_sparql, params={'query': self._query_get_by_label % label, 'format': 'json'})

        if not response.status_code == 200:
            self._logger.debug(f'Request for "{label}" failed (status code {response.status_code} ({response.reason}))')
            return items

        try:
            data = response.json()
        except Exception as e:
            self._logger.debug(f'Failed to read following JSON response:')
            self._logger.debug(f'{response.text}')
            return items

        for row in data['results']['bindings']:
            item = self.__translate_from_url(row['s']['value'])
            if not item.startswith("Q"):
                continue
            items.append(item)
        self._logger.debug(f'Found {len(items)} items for the label "{label}"')
        return items


    def __translate_from_url(self, url):
        if '/' in url and '-' not in url:
            item = url.split('/')[-1]
        elif '/' in url and '-' in url:
            item = url.split('/')[-1].split('-')[0]
        else:
            item = url
        return item
