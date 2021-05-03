import json
import os
from datasets import load_dataset

_path = os.path.dirname(__file__)

def get_aida_conll_train_dataset():
    return load_dataset('conll2003')['train']

def get_wikidata_disamb_dataset(train, part):
    if not part:
        print(f'Reading {train} data...')
        dataset_path = os.path.join(_path, '../data/wikidata_disamb/wikidata-disambig-{train}.json')
    else:
        print(f'Reading {train} data ({part})...')
        dataset_path = os.path.join(_path, f'../data/wikidata_disamb/wikidata-disambig-{train}.{part}.json')
    with open(dataset_path, encoding='utf8') as f:
        json_data = json.load(f)
    print(f'Read {len(json_data)} lines')
    return json_data

