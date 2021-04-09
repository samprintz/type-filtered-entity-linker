from inout.pbg import PBG
from transformers import DistilBertTokenizer
from tqdm import tqdm

import pudb # TODO

_bert_version = 'distilbert-base-uncased'
_max_sentence_length = 512
_embedding_size = 768


def init_tokenizer():
    return DistilBertTokenizer.from_pretrained(
        _bert_version,
        do_lower_case=True,
        add_special_tokens=True,
        max_length=_max_sentence_length,
        pad_to_max_length=True)


def init_pbg(sample_mode, use_cache):
    return PBG(sample_mode, use_cache)


def prepare_dataset(data_raw, sample_mode=False, use_cache=False):
    """
    Proprocess the raw JSON dataset by creating for each line one sample for the correct Wikidata ID and another for the wrong ID. Additionally add the tokens for the text and the embeddings for the Wikidata items.
    """

    tokenizer = init_tokenizer()
    pbg = init_pbg(sample_mode, use_cache)

    print(f'Preparing dataset ({len(data_raw)} lines)...')
    data = []
    line_count = 0
    sample_count = 0
    sample_count_failed = 0

    for line in tqdm(data_raw):
        line_count += 1
        #print(f'Line {line_count}/{len(data_raw)}')

        try:
            sample = {}

            sample['text'] = line['text']
            sample['text_tokenized'] = None # set by add_tokens()
            sample['text_attention_mask'] = None # set by add_tokens()
            sample['item_name'] = line['string']
            add_tokens(tokenizer, sample)
            sample['text_sf_mask'] = None # set by add_sf_mask()
            add_sf_mask(sample)

            # Once for correct Wikidata item
            sample['item_id'] = line['correct_id']
            sample['item_embedded'] = pbg.get_item_embedding(line['correct_id'])
            sample['answer'] = True
            data.append(sample)
            sample_count += 1

            # Once for wrong Wikidata item
            sample['item_id'] = line['wrong_id']
            sample['item_embedded'] = pbg.get_item_embedding(line['wrong_id'])
            sample['answer'] = False
            data.append(sample)
            sample_count += 1

        except Exception as e:
            print(str(e))
            sample_count_failed += 1

    print(f'Prepared {sample_count} samples from {line_count} lines (skipped {sample_count_failed} failed)')

    return data


def add_tokens(tokenizer, sample):
    """
    Add BERT tokens, attention masks and padding of the to dict.
    """
    # Text
    inputs = tokenizer.encode_plus(sample['text'],
            add_special_tokens=True,
            max_length=_max_sentence_length,
            padding='max_length',
            return_attention_mask=True)
    sample['text_tokenized'] = inputs['input_ids']
    sample['text_attention_mask'] = inputs['attention_mask']
    # Item name (mention/surface form)
    inputs = tokenizer.encode(sample['item_name'],
            add_special_tokens=False)
    sample['item_name_tokenized'] = inputs


def add_sf_mask(sample):
    sf_mask = []
    for token in sample['text_tokenized']:
        sf_mask.append([1. if token in sample['item_name_tokenized'] else 0.] * _embedding_size)
    # Padding
    while len(sf_mask) < _max_sentence_length:
        sf_mask.append(np.zeros(_embedding_size))
    sample['text_sf_mask'] = sf_mask


def reshape_dataset(data_pre):
    print(f'Reshaping dataset ({len(data_pre)} samples)...')
    data = {}

    keys = ['text_tokenized', 'text_attention_mask', 'text_sf_mask', 'item_embedded', 'answer']
    for key in keys:
        data[key] = []

    for sample in tqdm(data_pre):
        for key in keys:
            data[key].append(sample[key])

    print(f'Reshaped dataset')

    return data


