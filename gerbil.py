import datetime
from flask import Flask, request, Response
import logging
import os
import pprint
import sys

from config import ELConfig
from el.entity_linker import EntityLinker
from inout import nif
import utils

app = Flask(__name__)


# Entity linking settings
settings = {
    'ed_model_type' : 'bert_pbg',
    'ed_model_name' : 'model-20210529-1',
    'ed_model_checkpoint_epoch' : 15,
    'ed_model_checkpoint_type' : 'model', # model/weights
    'filter' : 'spacy', # spacy/bert/none
    'filter_model_name' : 'model-20210625-2',
    'filter_model_checkpoint_epoch' : 18,
    'filter_entities_without_type' : False,
    'filter_default_type' : False,
    'candidates_limit' : 500
    }

# Create config
config = ELConfig(settings, log_suffix='gerbil')

# Logging settings
logging.basicConfig(level=config.log_level, format=config.log_format,
        handlers=[logging.FileHandler(config.log_path), logging.StreamHandler()])
logger = logging.getLogger()

# Initialize linker
linker = EntityLinker(config)


@app.route('/', methods=['get', 'post'])
def index():
    logger.info("=== New request === ")

    nif_data = request.data
    doc = nif.read_nif(nif_data)

    logger.info(doc['uri'])
    logger.info(doc['text'])

    if app.config.get('experiment_type') is 'd2kb':
        doc_with_candidates, doc_with_entities = linker.d2kb(doc)
    else:
        doc_with_mentions, doc_with_candidates, doc_with_entities = linker.process(doc)

    result = nif.generate_nif(doc)

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        print(result)

    logger.info("Response sent")
    return Response(result, content_type='application/x-turtle')

    #print(request.data)
    #return "Hello World"

class LoggingMiddleware(object):
    def __init__(self, app):
        self._app = app

    def __call__(self, env, resp):
        errorlog = env['wsgi.errors']
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            pprint.pprint(('REQUEST', env), stream=errorlog)

        def log_response(status, headers, *args):
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                pprint.pprint(('RESPONSE', status, headers), stream=errorlog)
            return resp(status, headers, *args)

        return self._app(env, log_response)

#if __name__ == '__main__':
#   app.run()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'd2kb':
            app.config['experiment_type'] = 'd2kb'
            settings['experiment_type'] = 'd2kb'
        else:
            logger.warn(f'Unknown command line argument "{sys.argv[1]}"')
            settings['experiment_type'] = 'a2kb'
    else:
        settings['experiment_type'] = 'a2kb'
    utils.log_experiment_settings(settings=settings, mode="PRODUCTION")
    app.wsgi_app = LoggingMiddleware(app.wsgi_app)
    app.run()
