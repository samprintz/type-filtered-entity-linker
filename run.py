import datetime
import logging
import os
import sys

from config import ELConfig
from inout import dataset
from el.entity_linker import EntityLinker
import preprocess
import utils


def main():
    try:
        text = sys.argv[1]
    except:
        #text = "Napoleon was the first emperor of the French empire."
        #text = "private university in Nanjing, China which was founded in 1888 and sponsored by American churches. It's originally named the Nanking University, the first school officially named University."
        text = "Prokhorov said the visit would serve as a cornerstone for future interaction between players and coaches from the Nets and young Russians, with the aim of developing basketball in Russia, where the sport is a distant third in popularity behind soccer and hockey."

    doc = {'text' : text}
    #doc['mentions'] = [{'start' : 0, 'end' : 9, 'sf' : 'Prokhorov'}]
    #doc['mentions'] = [{'start' : 180, 'end' : 186, 'sf' : 'Russia'}]
    doc['mentions'] = [{'start' : 227, 'end' : 236, 'sf' : 'popularity'}]

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
    config = ELConfig(settings, log_suffix='run')

    # Logging settings
    logging.basicConfig(level=config.log_level, format=config.log_format,
            handlers=[logging.FileHandler(config.log_path), logging.StreamHandler()])
    logger = logging.getLogger()

    # Initialize linker and do the entity linking
    utils.log_experiment_settings(settings=settings, mode="TEST RUN")
    linker = EntityLinker(config)
    #doc_with_mentions, doc_with_candidates, doc_with_entities = linker.process(doc)
    doc_with_candidates, doc_with_entities = linker.d2kb(doc)
    logger.info("Done")


if __name__ == '__main__':
    main()
