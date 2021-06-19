import datetime
import logging
import os
from tqdm import tqdm

from inout import dataset
from typerec.model import TypeRecModel
from utils import log_experiment_settings


dirs = {
    'logging' : os.path.join(os.getcwd(), 'log'),
    'models' : os.path.join(os.getcwd(), 'data', 'models'),
    'wikidata_typerec' : os.path.join(os.getcwd(), 'data', 'wikidata_typerec')
    }

for path in dirs.values():
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    # Model and training settings
    config = {
        'dataset_partial' : 'small', # small/medium/full
        'model_type' : 'typerec',
        'model_name' : 'model-20210619-2',
        'epochs' : 3,
        'batch_size' : 32,
        'dropout' : 0.5
        }

    # Path for saving model checkpoints
    config['saving_dir'] = os.path.join(dirs['models'], config['model_type'],
            config['model_name'])

    # Logging settings
    log_level = logging.INFO
    # include model_type and model_name in log file name
    log_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(dirs['logging'], f'{log_filename}-test.log')
    log_format = "%(asctime)s: %(levelname)-1.1s %(name)s:%(lineno)d] %(message)s"

    logger = logging.getLogger()
    logging.basicConfig(level=log_level, format=log_format,
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    # Load data
    features = ['text_and_mention_tokenized', 'text_and_mention_attention_mask', 'item_type_index']
    dataset_test = dataset.load_typerec_test_dataset(
            dataset_dir = dirs['wikidata_typerec'],
            dataset_name = 'wikidata-typerec',
            dataset_partial = config['dataset_partial'],
            features = features)
    config['dataset_length_test'] = len(next(iter(dataset_test)))

    # Initialize the model and train it
    log_experiment_settings(settings=config, is_test=True)
    model = TypeRecModel()
    model.test(dataset_test,
        saving_dir=config['saving_dir'],
        epochs=config['epochs'],
        batch_size=config['batch_size'])


if __name__ == '__main__':
    main()
