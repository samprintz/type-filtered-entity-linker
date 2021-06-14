import datetime
import logging
import os
import sys
from inout import dataset
import preprocess
from el.model import ELModel
from el.entity_linker import EntityLinker


dirs = {
    'logging' : os.path.join(os.getcwd(), 'log'),
    'models' : os.path.join(os.getcwd(), 'data', 'models'),
    'type_cache' : os.path.join(os.getcwd(), 'data', 'type_cache')
    }

for path in dirs.values():
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    try:
        text = sys.argv[1]
    except:
        text = "Napoleon was the first emperor of the French empire."

    doc = {'text' : text}

    # Model settings

    # working
    #model_type = 'bert_rnn'
    #model_name = 'model-20210428-1'
    #model_checkpoint_epoch = 60
    #model_checkpoint_type = 'model'

    # working
    #model_type = 'bert_pbg'
    #model_name = 'model-20210503-2'
    #model_checkpoint_epoch = 20
    #model_checkpoint_type = 'model'

    # working
    model_type = 'bert_pbg'
    model_name = 'model-20210529-1'
    model_checkpoint_epoch = 60
    model_checkpoint_type = 'model'

    # not working, would require the model in el/model.py to be identical with the
    # trained model that should be loaded
    #model_type = 'gru_gcn'
    #model_name = 'model-20210529-1'
    #model_checkpoint_epoch = 60
    #model_checkpoint_type = 'weights'

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
        'model_path' : os.path.join(dirs['models'], model_type, model_name, f'cp-{model_checkpoint_epoch:04d}.ckpt'),
        'model_checkpoint_type' : model_checkpoint_type,
        'type_cache_dir' : dirs['type_cache'],
        'candidates_limit' : 100,
        'use_filter' : True
        }

    # Initialize linker and do the entity linking
    linker = EntityLinker(config)
    doc_with_mentions, doc_with_candidates, doc_with_entities = linker.process(doc)
    logger.info("Done")


if __name__ == '__main__':
    main()

