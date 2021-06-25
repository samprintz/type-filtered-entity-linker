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
        #text = "Napoleon was the first emperor of the French empire."
        text = "private university in Nanjing, China which was founded in 1888 and sponsored by American churches. It's originally named the Nanking University, the first school officially named University."

    doc = {'text' : text}

    # Entity linking settings
    settings = {
        'ed_model_type' : 'bert_pbg',
        'ed_model_name' : 'model-20210529-1',
        'ed_model_checkpoint_epoch' : 60,
        'ed_model_checkpoint_type' : 'model', # model/weights
        'filter' : 'bert', # spacy/bert/none
        'filter_model_name' : 'model-20210625-1',
        'filter_model_checkpoint_epoch' : 5,
        'candidates_limit' : 100
        }

    # Create config
    config = ELConfig(settings, log_suffix='el-run')

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
