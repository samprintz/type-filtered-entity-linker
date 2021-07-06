import os
import logging
import datetime

import utils

class Config:

    # Directories
    dirs = {
        'logging' : os.path.join(os.getcwd(), 'log'),
        'models' : os.path.join(os.getcwd(), 'data', 'models'),
        'wikidata_disamb' : os.path.join(os.getcwd(), 'data', 'wikidata_disamb'),
        'wikidata_typerec' : os.path.join(os.getcwd(), 'data', 'wikidata_typerec'),
        'type_cache' : os.path.join(os.getcwd(), 'data', 'type_cache'),
        'subclass_cache' : os.path.join(os.getcwd(), 'data', 'subclass_cache')
        }

    # Create directories
    for path in dirs.values():
        if not os.path.exists(path):
            print(f'Create directory {path}')
            os.makedirs(path)

    def __init__(self, log_suffix='log'):
        # Logging
        self.log_level = logging.INFO
        self.log_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_filename = f'{self.log_timestamp}-{log_suffix}'
        self.log_path = os.path.join(self.dirs['logging'], f'{self.log_filename}.log')
        self.log_format = "%(asctime)s: %(levelname)-1.1s %(name)s:%(lineno)d] %(message)s"


class ModelConfig(Config):

    def __init__(self, settings, log_suffix):
        # Settings for neural network training/testing
        self.model_name = settings['model_name']
        self.model_type = settings['model_type']
        self.epochs = settings['epochs']
        self.batch_size = settings['batch_size']
        self.dropout_bert = settings['dropout_bert']
        self.dropout_bert_attention = settings['dropout_bert_attention']
        log_suffix = f'{self.model_name}-{log_suffix}'

        # Path for saving model checkpoints
        self.model_saving_dir = os.path.join(self.dirs['models'], self.model_type,
                self.model_name)

        super(ModelConfig, self).__init__(log_suffix)


class ELConfig(Config):

    def __init__(self, settings, log_suffix):
        # Settings for entity disambiguation model
        self.ed_model_name = settings['ed_model_name']
        self.ed_model_type = settings['ed_model_type']
        self.ed_model_checkpoint_epoch = settings['ed_model_checkpoint_epoch']
        self.ed_model_checkpoint_type = settings['ed_model_checkpoint_type']
        self.ed_model_path = utils.get_model_path(self.dirs['models'], self.ed_model_type,
                self.ed_model_name, self.ed_model_checkpoint_epoch)

        # Settings for type filter model
        if settings['filter'] == 'bert':
            self.filter = settings['filter']
            self.filter_model_name = settings['filter_model_name']
            self.filter_model_checkpoint_epoch = settings['filter_model_checkpoint_epoch']
            self.filter_model_path = utils.get_model_path(self.dirs['models'], 'typerec',
                    self.filter_model_name, self.filter_model_checkpoint_epoch)
            self.filter_entities_without_type = settings['filter_entities_without_type']
            self.filter_default_type = settings['filter_default_type']
        elif settings['filter'] == 'spacy':
            self.filter = settings['filter']
            self.filter_entities_without_type = settings['filter_entities_without_type']
            self.filter_default_type = settings['filter_default_type']
        else:
            self.filter = False

        # Other settings
        self.candidates_limit = settings['candidates_limit']
        log_suffix = f'el-{log_suffix}'

        super(ELConfig, self).__init__(log_suffix)



class TypeRecDatasetConfig(Config):

    def __init__(self, settings):
        self.dataset_train = settings['dataset_train']
        self.dataset_part = settings['dataset_part']
        self.detailed_types = settings['detailed_types']
        log_suffix = f'dataset-{self.dataset_train}-{self.dataset_part}'

        super(TypeRecDatasetConfig, self).__init__(log_suffix)
