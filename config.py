import os
import logging
import datetime

class Config:

    def __init__(self, settings, is_test):
        self.model_name = settings['model_name']
        self.model_type = settings['model_type']
        self.epochs = settings['epochs']
        self.batch_size = settings['batch_size']
        self.dropout_bert = settings['dropout_bert']
        self.dropout_bert_attention = settings['dropout_bert_attention']
        self.is_test = is_test


        # Directories
        self.dirs = {
            'logging' : os.path.join(os.getcwd(), 'log'),
            'models' : os.path.join(os.getcwd(), 'data', 'models'),
            'wikidata_typerec' : os.path.join(os.getcwd(), 'data', 'wikidata_typerec')
            }

        # Create directories
        for path in self.dirs.values():
            if not os.path.exists(path):
                print(f'Create directory {path}')
                os.makedirs(path)

        # Path for saving model checkpoints
        self.model_saving_dir = os.path.join(self.dirs['models'], self.model_type,
                self.model_name)

        # Logging
        log_suffix = 'test' if self.is_test else 'train'
        self.log_level = logging.INFO
        self.log_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_filename = f'{self.log_timestamp}-{self.model_name}-{log_suffix}'
        self.log_path = os.path.join(self.dirs['logging'], f'{self.log_filename}.log')
        self.log_format = "%(asctime)s: %(levelname)-1.1s %(name)s:%(lineno)d] %(message)s"
