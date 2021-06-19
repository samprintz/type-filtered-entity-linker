import datetime
import logging
import os
from tqdm import tqdm

from inout import dataset
from preprocess import Preprocessor
from typerec.model import TypeRecModel


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
        'dataset' : 'test', # train/test/dev
        'dataset_part' : 'small', # small/medium/full
        'model_type' : 'typerec',
        'model_name' : 'model-20210619-1',
        'epochs' : 3,
        'batch_size' : 2, # TODO 32
        'dropout' : 0.5
        }

    config['saving_dir'] = os.path.join(dirs['models'], config['model_type'],
            config['model_name'])

    # Logging settings
    log_level = logging.INFO
    # include model_type and model_name in log file name
    log_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(dirs['logging'], f'{log_filename}.log')
    log_format = "%(asctime)s: %(levelname)-1.1s %(name)s:%(lineno)d] %(message)s"

    logger = logging.getLogger()
    logging.basicConfig(level=log_level, format=log_format,
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    # Load data
    data_raw = dataset.get_wikidata_typerec_dataset(dirs['wikidata_typerec'],
            config['dataset'], config['dataset_part'])

    # Preprocess data
    preprocessor = Preprocessor()
    data_pre = preprocessor.prepare_typerec_dataset(data_raw)
    features = ['text_and_mention_tokenized', 'text_and_mention_attention_mask', 'item_type_index']
    data = preprocessor.reshape_typerec_dataset(data_pre, features)

    # Initialize the model and train it
    model = TypeRecModel()
    # TODO Log configuration of the model
    model.test(data,
        saving_dir=config['saving_dir'],
        epochs=config['epochs'],
        batch_size=config['batch_size'])


if __name__ == '__main__':
    main()
