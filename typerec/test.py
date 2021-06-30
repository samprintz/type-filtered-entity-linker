import datetime
import logging
import os
from tqdm import tqdm

from config import ModelConfig
from inout import dataset
from typerec import types
from typerec.model import TypeRecModel
import utils


def main():
    # Model and training settings
    settings = {
        'dataset_partial' : 'small', # small/medium/full
        'model_type' : 'typerec',
        'model_name' : 'model-20210625-2',
        'epochs' : 3,
        'batch_size' : 1,
        'dropout_bert' : 0.2,
        'dropout_bert_attention' : 0.2
        }

    # Create config
    config = ModelConfig(settings, 'test')

    # Logging settings
    logging.basicConfig(level=config.log_level, format=config.log_format,
            handlers=[logging.FileHandler(config.log_path), logging.StreamHandler()])
    logger = logging.getLogger()

    # Load data
    features = ['text_and_mention_tokenized', 'text_and_mention_attention_mask',
            'item_type_onehot', 'text', 'item_name', 'item_id'] # text, item_name, item_id for logging
    dataset_test = dataset.load_typerec_test_dataset(
            dataset_dir=config.dirs['wikidata_typerec'],
            dataset_name='wikidata-typerec',
            dataset_partial=settings['dataset_partial'],
            features=features)

    # Set dataset properties
    config.dataset_length_test = utils.get_dataset_length(dataset_test)
    config.steps_per_epoch_test = utils.get_steps_per_epoch(
            config.dataset_length_test, config.batch_size)
    config.types_count = types.get_types_count()

    # Initialize the model and test it
    utils.log_experiment_settings(settings=settings, mode='TEST')
    model = TypeRecModel(config)
    model.test(dataset_test,
        saving_dir=config.model_saving_dir,
        epochs=config.epochs,
        batch_size=config.batch_size)


if __name__ == '__main__':
    main()
