import os
import logging

from disamb.model import EDModel
from inout import dataset
import preprocess
import utils


dirs = {
    'logging' : os.path.join(os.getcwd(), 'log'),
    'models' : os.path.join(os.getcwd(), 'data', 'models')
    }

for path in dirs.values():
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    # Model settings
    model_type = 'pbg'
    model_name = 'model-20210503-2'

    # Model and training settings
    config = {
        # TODO include model settings here and use them for better log file name
        'save_path' : os.path.join(dirs['models'], model_type, model_name, 'cp-{model_checkpoint:04d}.ckpt'),
        'epochs' : 20,
        'batch_size' : 32,
        'dropout' : 1.0
        }

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
    data_raw = dataset.get_wikidata_disamb_dataset('train', 'small')

    # Preprocess data
    preprocessor = Preprocessor()
    data_pre = preprocessor.prepare_dataset(data_raw)
    data = preprocessor.reshape_dataset(data_pre)

    # Initialize the model and train it
    ed_model = EDModel()
    # TODO Log configuration of the model
    ed_model.train(data,
        save_path=config['save_path'],
        epochs=config['epochs'],
        batch_size=config['batch_size'])


if __name__ == '__main__':
    main()

