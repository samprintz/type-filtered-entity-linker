import logging
import os
import requests
import time

class Wikidata:
    _url_sparql = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
    _request_headers = headers = {"User-Agent": "FilteredELBot/0.0"}
    _retry_after_time = 10 # 10 seconds
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

    def __init__(self, type_cache_dir):
        self._logger = logging.getLogger(__name__)
        self._type_cache_dir = type_cache_dir


    def get_neighborhood(self, item_id, hops=1):
        pass


    def get_items_by_label(self, label):
        self._logger.debug(f'Get items with label "{label}" from Wikidata SPARQL endpoint')

        items = []

        response = requests.get(self._url_sparql, headers=self._request_headers, params={'query': self._query_get_by_label % label, 'format': 'json'})

        if not response.status_code == 200:
            self._logger.info(f'Request for "{label}" failed (status code {response.status_code} ({response.reason}))')
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


    '''
    Get Wikidata entity type of a given item ID. Tries to read from cache first.
    Then queries the Wikidata SPARQL endpoint (saves result to cache if successful).
    '''
    def get_types_of_item(self, item_id):
        try:
            # Check cache
            return self.__get_types_from_cache(self._type_cache_dir, item_id)
        except FileNotFoundError:
            # Request from Wikidata SPARQL endpoint
            rdf_types = self.__get_types_from_sparql_endpoint(item_id)
            # Write always to cache, also if length is 0, s.t. items without type are queried again
            #if len(rdf_types) > 0:
            self.__write_types_to_cache(self._type_cache_dir, item_id, rdf_types)
            return rdf_types


    '''
    Read Wikidata entity types from local cache. Throws FileNotFoundError when not cached yet.
    '''
    def __get_types_from_cache(self, cache_dir, item_id):
        rdf_types = []

        cache_file = f'{self._type_cache_dir}/{item_id}.nt'

        if os.path.exists(cache_file):
            self._logger.debug(f'Read RDF types of {item_id} from cache')

            with open(cache_file) as cache:
                for line in cache:
                    rdf_type_id = line.strip().split('\t')[0]
                    rdf_type_label = line.strip().split('\t')[1]
                    rdf_type = {'id' : rdf_type_id, 'label' : rdf_type_label}
                    rdf_types.append(rdf_type)

            return rdf_types

        else:
            raise FileNotFoundError(f'No file found in type cache for "{item_id}"')


    '''
    Write Wikidata entity types to local cache
    '''
    def __write_types_to_cache(self, cache_dir, item_id, rdf_types):
        cache_file = f'{self._type_cache_dir}/{item_id}.nt'

        with open(cache_file, "w+") as cache_file:
            self._logger.debug(f'Write graph of {item_id} to cache')

            for rdf_type in rdf_types:
                line = f'{rdf_type["id"]}\t{rdf_type["label"]}\n'
                cache_file.write(line)


    '''
    Request Wikidata entity types from Wikidata SPARQL endpoint.
    '''
    def __get_types_from_sparql_endpoint(self, item_id):
        rdf_types = []

        # Loop in case of HTTP errors
        first_try = True
        while True:

            if first_try:
                self._logger.debug(f'Get RDF types of {item_id} from Wikidata SPARQL endpoint')
            else:
                self._logger.debug(f'Get RDF types of {item_id} from Wikidata SPARQL endpoint (retry)')

            query = self._query_get_types_by_id % self.__translate_to_wikidata_entity(item_id)
            response = requests.get(self._url_sparql, params={'query': query, 'format': 'json'})

            if response.status_code == 200:
                break # continue below with getting the JSON
            else:
                self._logger.info(f'Request for "{item_id}" failed (status code {response.status_code} ({response.reason}))')
                first_try = False

                if response.status_code == 403: # banned
                    self._logger.info(f'Banned by Wikidata (HTTP 403), exiting')
                    sys.exit(1)

                elif response.status_code == 429: # too many requests
                    try:
                        retry_after_time = int(response.headers["Retry-After"])
                    except KeyError:
                        self._logger.info(f'Could not find "Retry-After" time in header, using default time ({self._retry_after_time} seconds)')
                        retry_after_time = self._retry_after_time

                    self._logger.info(f'Sleep for {retry_after_time} seconds...')
                    time.sleep(retry_after_time)
                    self._logger.info(f'Continue')

        # Read JSON
        try:
            data = response.json()
        except Exception as e:
            self._logger.debug(f'Failed to read following JSON response:')
            self._logger.debug(f'{response.text}')
            raise e

        for row in data['results']['bindings']:
            rdf_type_id = row['type']['value']
            rdf_type_label = row['typeLabel']['value']
            rdf_type = {'id' : rdf_type_id, 'label' : rdf_type_label}
            rdf_types.append(rdf_type)

        self._logger.debug(f'Found {len(rdf_types)} RDF types for the item ID "{item_id}"')

        return rdf_types
