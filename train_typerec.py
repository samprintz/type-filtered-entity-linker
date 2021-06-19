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
            config['model_name']) #, 'cp-{model_checkpoint:04d}.ckpt') TODO

    # Logging settings
    log_level = logging.INFO
    # include model_type and model_name in log file name
    log_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(dirs['logging'], f'{log_filename}-train.log')
    log_format = "%(asctime)s: %(levelname)-1.1s %(name)s:%(lineno)d] %(message)s"

    logger = logging.getLogger()
    logging.basicConfig(level=log_level, format=log_format,
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    # Load dataset
    features = ['text_and_mention_tokenized', 'text_and_mention_attention_mask', 'item_type_index']
    dataset_train, dataset_dev = dataset.load_typerec_train_datasets(
            dataset_dir = dirs['wikidata_typerec'],
            dataset_name = 'wikidata-typerec',
            dataset_partial = config['dataset_partial'],
            features = features)
    config['dataset_length_train'] = len(next(iter(dataset_train)))
    config['dataset_length_dev'] = len(next(iter(dataset_dev)))

    # Initialize the model and train it
    log_experiment_settings(settings=config, is_test=False)
    model = TypeRecModel()
    model.train((dataset_train, dataset_dev),
        saving_dir=config['saving_dir'],
        epochs=config['epochs'],
        batch_size=config['batch_size'])


if __name__ == '__main__':
    main()
