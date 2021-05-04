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
        self._logger.debug(f'Get items with label {label} from Wikidata SPARQL endpoint')
        data = requests.get(self._url_sparql, params={
                'query': self._query_get_by_label % label,
                'format': 'json'}).json()

        items = []
        for row in data['results']['bindings']:
            # TODO try except this:
            item = self.__translate_from_url(row['s']['value'])

            items.append(item)
            self._logger.debug(f'Found {item} with label {label}')
        return items


    def __translate_from_url(self, url):
        if '/' in url and '-' not in url:
            item = url.split('/')[-1]
        elif '/' in url and '-' in url:
            item = url.split('/')[-1].split('-')[0]
        else:
            item = url
        return item
