import datetime
from flask import Flask, request, Response
import logging
import os
import pprint

from el.model import ELModel
from el.entity_linker import EntityLinker
from inout import nif

app = Flask(__name__)


# TODO The code below is copied from run.py Merge/reuse code?

dirs = {
    'logging' : os.path.join(os.getcwd(), 'log'),
    'models' : os.path.join(os.getcwd(), 'data', 'models')
    }

for path in dirs.values():
    if not os.path.exists(path):
        os.makedirs(path)

# Model settings
model_type = 'pbg'
model_name = 'model-20210503-2'
model_checkpoint = 20

# Logging settings
log_level = logging.INFO
log_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_path = os.path.join(dirs['logging'], f'{log_filename}.log')
log_format = "%(asctime)s: %(levelname)-1.1s %(name)s:%(lineno)d] %(message)s"

logger = logging.getLogger()
logging.basicConfig(level=log_level, format=log_format,
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

# Entity linking settings
config = {
    'model_path' : os.path.join(dirs['models'], model_type, model_name, f'cp-{model_checkpoint:04d}.ckpt'),
    'use_filter' : True
    }

# Initialize linker
linker = EntityLinker(config)

# TODO The code above is copied from run.py Merge/reuse code?


@app.route('/', methods=['get', 'post'])
def index():
    logger.info("=== New request === ")

    nif_data = request.data
    doc = nif.read_nif(nif_data)

    logger.info(doc['uri'])
    logger.info(doc['text'])

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
    app.wsgi_app = LoggingMiddleware(app.wsgi_app)
    app.run()
