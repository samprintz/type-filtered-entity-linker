import datetime
import logging
import os
from tqdm import tqdm

from config import Config
from inout import dataset
from typerec.model import TypeRecModel
import utils


def main():
    # Model and training settings
    settings = {
        'dataset_partial' : 'full', # small/medium/full
        'model_type' : 'typerec',
        'model_name' : 'model-20210621-2',
        'epochs' : 20,
        'batch_size' : 128,
        'dropout_bert' : 0.2,
        'dropout_bert_attention' : 0.2
        }

    # Create config
    config = Config(settings, is_test=False)

    # Logging settings
    logging.basicConfig(level=config.log_level, format=config.log_format,
            handlers=[logging.FileHandler(config.log_path), logging.StreamHandler()])
    logger = logging.getLogger()

    # Load dataset
    features = ['text_and_mention_tokenized', 'text_and_mention_attention_mask',
            'item_type_index']
    dataset_train, dataset_dev = dataset.load_typerec_train_datasets(
            dataset_dir=config.dirs['wikidata_typerec'],
            dataset_name='wikidata-typerec',
            dataset_partial=settings['dataset_partial'],
            features=features)

    # Set dataset properties
    config.dataset_length_train = utils.get_dataset_length(dataset_train)
    config.dataset_length_dev = utils.get_dataset_length(dataset_dev)
    config.steps_per_epoch_train = utils.get_steps_per_epoch(
            config.dataset_length_train, config.batch_size)
    config.steps_per_epoch_dev = utils.get_steps_per_epoch(
            config.dataset_length_dev, config.batch_size)

    # Initialize the model and train it
    utils.log_experiment_settings(settings=settings, is_test=False)
    model = TypeRecModel(config)
    model.train(dataset_train, dataset_dev,
        saving_dir=config.model_saving_dir,
        epochs=config.epochs,
        batch_size=config.batch_size)


if __name__ == '__main__':
    main()
