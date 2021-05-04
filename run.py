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
    'models' : os.path.join(os.getcwd(), 'data', 'models')
    }

for path in dirs.values():
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    try:
        text = sys.argv[1]
    except:
        text = "Napoleon was the first emperor of the French empire."

    # Model settings
    model_type = 'rnn'
    model_name = 'model-20210428-1'
    model_checkpoint = 60

    # Logging settings
    log_level = logging.DEBUG
    log_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(dirs['logging'], f'{log_filename}.log')
    log_format = "%(asctime)s: %(levelname)-1.1s %(name)s:%(lineno)d] %(message)s"

    logger = logging.getLogger()
    logging.basicConfig(level=log_level, format=log_format,
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    # Load model
    # TODO load model entity disambiguator (where it is actually needed)
    model_path = os.path.join(dirs['models'], model_type, model_name, f'cp-{model_checkpoint:04d}.ckpt')
    model = ELModel()
    model.load(model_path)

    # Initialize linker and do the entity linking
    linker = EntityLinker(model)
    mentions, candidates, entities = linker.process(text)
    logger.info("Done")


if __name__ == '__main__':
    main()

