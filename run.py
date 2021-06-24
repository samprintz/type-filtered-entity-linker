import datetime
import logging
import os
import sys

from config import ELConfig
from inout import dataset
from el.model import ELModel
from el.entity_linker import EntityLinker
import preprocess


def main():
    try:
        text = sys.argv[1]
    except:
        text = "Napoleon was the first emperor of the French empire."

    doc = {'text' : text}

    # Entity linking settings
    settings = {
        'ed_model_type' : 'bert_pbg',
        'ed_model_name' : 'model-20210529-1',
        'ed_model_checkpoint_epoch' : 60,
        'ed_model_checkpoint_type' : 'model', # model/weights
        'filter' : 'bert', # spacy/bert/none
        'filter_model_name' : 'model-20210621-2',
        'filter_model_checkpoint_epoch' : 20,
        'candidates_limit' : 100
        }

    # Create config
    config = ELConfig(settings)

    # Logging settings
    logging.basicConfig(level=config.log_level, format=config.log_format,
            handlers=[logging.FileHandler(config.log_path), logging.StreamHandler()])
    logger = logging.getLogger()

    # Initialize linker and do the entity linking
    linker = EntityLinker(config)
    doc_with_mentions, doc_with_candidates, doc_with_entities = linker.process(doc)
    logger.info("Done")


if __name__ == '__main__':
    main()
