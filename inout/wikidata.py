import logging
import requests

class Wikidata:
    _url_sparql = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
    # TODO Second line required?
    # TODO replace or filter ?label with rdfs:label or skos:altLabel
    _query_get_by_label = '''
            SELECT DISTINCT ?s {
                ?s ?label "%s"@en .
                ?s ?p ?o .
            }
            '''
    _query_get_types_by_id = '''
            SELECT DISTINCT ?type ?typeLabel {
                %s wdt:P31 ?type .
                ?type rdfs:label ?typeLabel
                FILTER (lang(?typeLabel) = 'en')
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
            # Filter to the ones that start with "Q" (Q123, Q...)
            if not item.startswith("Q"):
                continue
            items.append(item)
        self._logger.debug(f'Found {len(items)} items for the label "{label}"')
        return items


    def __translate_from_url(self, url):
        # TODO move to utils
        if '/' in url and '-' not in url:
            item = url.split('/')[-1]
        elif '/' in url and '-' in url:
            item = url.split('/')[-1].split('-')[0]
        else:
            item = url
        return item


    def __translate_to_wikidata_entity(self, item_id):
        # TODO move to utils.py
        wikidata_namespace_prefix = 'wd'
        return f'{wikidata_namespace_prefix}:{item_id}'


    def get_types_of_item(self, item_id):
        self._logger.debug(f'Get RDF types of "{item_id}" from Wikidata SPARQL endpoint')

        rdf_types = []

        query = self._query_get_types_by_id % self.__translate_to_wikidata_entity(item_id)
        response = requests.get(self._url_sparql, params={'query': query, 'format': 'json'})

        if not response.status_code == 200:
            self._logger.debug(f'Request for "{item_id}" failed (status code {response.status_code} ({response.reason}))')
            return rdf_types

        try:
            data = response.json()
        except Exception as e:
            self._logger.debug(f'Failed to read following JSON response:')
            self._logger.debug(f'{response.text}')
            return rdf_types

        for row in data['results']['bindings']:
            rdf_type = row['type']['value']
            rdf_types.append(rdf_type)
        self._logger.debug(f'Found {len(rdf_types)} RDF types for the item ID "{item_id}"')
        return rdf_types
