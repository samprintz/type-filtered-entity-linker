import logging
import numpy as np
from transformers import DistilBertTokenizer
from tqdm import tqdm

from inout.pbg import PBG


class Preprocessor:
    _bert_version = 'distilbert-base-uncased'
    _max_text_length = 512
    _embedding_size = 768


    def __init__(self, sample_mode=False, use_cache=True):
        self._logger = logging.getLogger(__name__)
        self._sample_mode = sample_mode
        self._use_cache = use_cache
        self._tokenizer = self.__init_tokenizer()
        self._pbg = self.__init_pbg(sample_mode=sample_mode, use_cache=use_cache)


    def __init_tokenizer(self):
        return DistilBertTokenizer.from_pretrained(
            self._bert_version,
            do_lower_case=True,
            add_special_tokens=True,
            max_length=self._max_text_length,
            pad_to_max_length=True)


    def __init_pbg(self, sample_mode, use_cache):
        return PBG(sample_mode, use_cache)


    def prepare_sample(self, sample_raw):
        """
        Preprocess a sample consisting of the text, one mention and one candidate.
        Requires preprocess.reshape() before feeding it into model.predict().
        """

        text, mention, candidate = sample_raw
        self._logger.debug(f'Preprocessing sample ("{mention}" vs. "{candidate}")')

        sample = {}
        sample['text'] = text
        sample['text_tokenized'] = None # set by add_tokens()
        sample['text_attention_mask'] = None # set by add_tokens()
        sample['item_name'] = mention
        self.add_tokens(sample)
        sample['text_mention_mask'] = None # set by add_mention_mask()
        self.add_mention_mask(sample)
        sample['item_id'] = candidate
        sample['item_pbg'] = self._pbg.get_item_embedding(candidate)
        sample['item_glove'] = np.empty((1, 900)) # TODO
        return sample


    def prepare_dataset(self, data_raw):
        """
        Proprocess the raw JSON dataset by creating for each line one sample for the correct Wikidata ID and another for the wrong ID. Additionally add the tokens for the text and the embeddings for the Wikidata items.
        Requires preprocess.reshape() before feeding it into model.train().
        """

        self._logger.debug(f'Preparing dataset ({len(data_raw)} lines)...')
        data = []
        line_count = 0
        sample_count = 0
        sample_count_failed = 0

        for line in tqdm(data_raw):
            line_count += 1
            #self._logger.debug(f'Line {line_count}/{len(data_raw)}')

            try:
                # TODO Call prepare_sample() here?
                sample = {}

                sample['text'] = line['text']
                sample['text_tokenized'] = None # set by add_tokens()
                sample['text_attention_mask'] = None # set by add_tokens()
                sample['item_name'] = line['string']
                self.add_tokens(sample)
                sample['text_mention_mask'] = None # set by add_mention_mask()
                self.add_mention_mask(sample)

                # Once for correct Wikidata item
                sample['item_id'] = line['correct_id']
                sample['item_pbg'] = self._pbg.get_item_embedding(line['correct_id'])
                sample['item_glove'] = np.empty((1, 900)) # TODO
                sample['answer'] = True
                data.append(sample)
                sample_count += 1

                # Once for wrong Wikidata item
                sample['item_id'] = line['wrong_id']
                sample['item_pbg'] = self._pbg.get_item_embedding(line['wrong_id'])
                sample['item_glove'] = np.empty((1, 900)) # TODO
                sample['answer'] = False
                data.append(sample)
                sample_count += 1

            except ValueError as e: # skip sample when there is no embedding found
                self._logger.info(str(e))
                sample_count_failed += 1
                continue

        self._logger.debug(f'Prepared {sample_count} samples from {line_count} lines (skipped {sample_count_failed} failed)')

        return data


    def add_tokens(self, sample):
        """
        Add BERT tokens, attention masks and padding of the input text to sample.
        """
        # Text
        inputs = self._tokenizer.encode_plus(sample['text'],
                add_special_tokens=True,
                max_length=self._max_text_length,
                padding='max_length', # TODO padding here or in model (together with item_glove)?
                return_attention_mask=True)
        sample['text_tokenized'] = inputs['input_ids']
        sample['text_attention_mask'] = inputs['attention_mask']
        # Item name (mention/surface form)
        inputs = self._tokenizer.encode(sample['item_name'],
                add_special_tokens=False)
        sample['item_name_tokenized'] = inputs


    def add_mention_mask(self, sample):
        """
        Add a mask to the sample for masking the output of BERT to the position of the mention.
        """
        mention_mask = []
        for token in sample['text_tokenized']:
            mention_mask.append([1. if token in sample['item_name_tokenized'] else 0.] * self._embedding_size)
        # Padding
        while len(mention_mask) < self._max_text_length:
            mention_mask.append(np.zeros(self._embedding_size))
        sample['text_mention_mask'] = mention_mask


    def reshape_dataset(self, data_pre, for_training=True):
        """
        Reshapes a preprocessed dataset (list of samples) to a suitable input (dict of features) for model.train() or
        model.predict().
        """

        if not isinstance(data_pre, list):
            raise ValueError(f'reshape_dataset() requires a list as input')

        self._logger.debug(f'Reshaping dataset ({len(data_pre)} samples)...')
        data = {}

        keys = ['text_tokenized', 'text_attention_mask', 'text_mention_mask', 'item_pbg', 'item_glove']
        if for_training:
            keys.append('answer')

        for key in keys:
            data[key] = []

        for sample in tqdm(data_pre):
            for key in keys:
                data[key].append(sample[key])

        for key in keys:
            data[key] = np.asarray(data[key])

        self._logger.debug(f'Reshaped dataset')

        return data


