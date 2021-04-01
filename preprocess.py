from inout.pbg import PBG
from transformers import DistilBertTokenizer

import pudb # TODO

_bert_version = 'distilbert-base-uncased'
_max_sentence_length = 512


def init_tokenizer():
    return DistilBertTokenizer.from_pretrained(
        _bert_version,
        do_lower_case=True,
        add_special_tokens=True,
        max_length=_max_sentence_length,
        pad_to_max_length=True)

def init_pbg(sample_mode):
    return PBG(sample_mode)


def prepare_dataset(data_raw):
    """
    Proprocess the raw JSON dataset by creating for each line one sample for the correct Wikidata ID and another for the wrong ID. Additionally add the tokens for the text and the embeddings for the Wikidata items.
    """
    tokenizer = init_tokenizer()
    pbg = init_pbg(False)


    print(f'Preparing dataset ({len(data_raw)} lines)...')
    data = []
    line_count = 0
    sample_count = 0
    sample_count_failed = 0

    for line in data_raw:
        line_count += 1
        print(f'Line {line_count}/{len(data_raw)}')

        try:
            sample = {}

            sample['text'] = line['text']
            sample['text_tokenized'] = None # set by add_tokens()
            sample['text_attention_mask'] = None # set by add_tokens()
            add_tokens(tokenizer, sample)

            sample['item_name'] = line['string']

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
    inputs = tokenizer.encode_plus(sample['text'],
            add_special_tokens=True,
            max_length=_max_sentence_length,
            padding='max_length',
            return_attention_mask=True)
    sample['text_tokenized'] = inputs['input_ids']
    sample['text_attention_mask'] = inputs['attention_mask']

