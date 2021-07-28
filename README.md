# Type-filtered Entity Linker

This is an entity linker using a type filter based on BERT to filter the set of candidate entities and accelerate entity disambiguation.

## Installation

1. Create virtual environment

```bash
python -m venv ve
source ./ve/bin/activate
pip install -r requirements.txt
```

2. Download spaCy <tt>small</tt> and <tt>transformer</tt> models

```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
```

3. Download Wikidata-Disamb dataset from [GitHub](https://github.com/ContextScout/ned-graphs/tree/master/dataset) and copy it to <tt>./data/wikidata_disamb</tt>

4. Download Wikidata-TypeRec dataset from [GitHub](https://github.com/samprintz/wikidata-typerec-dataset) and copy it to <tt>./data/wikidata_typerec</tt>

5. Download GloVe

```bash
mkdir data/glove
cd data/glove
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
echo "2196017 300" | cat - glove.840B.300d.txt > glove_2.2M.txt
cd ../..
```

6. Download PyTorch-BigGraph embeddings

```bash
mkdir data/pbg
cd data/pbg
wget https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1_names.json.gz
gunzip wikidata_translation_v1_names.json.gz
wget https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1_vectors.npy.gz
gunzip wikidata_translation_v1_vectors.json.gz
```

7. Train the ED model "BERT+PBG" and copy it to <tt>./data/models/bert_pbg</tt>

See https://github.com/samprintz/ed-with-kg-structure#entity-disambiguation-with-knowledge-graph-structure

8. Train the type classifier TypeRec-BERT and copy it to <tt>./data/models/typerec</tt>

See https://github.com/samprintz/type-filtered-entity-linker#type-classifier-typerec-bert


## Run Type-Filtered Entity Linker

To execute a test run of the type-filtered entity linker

```bash
python -m run
```

To run the NIF service for the GERBIL evaluation of the type-filtered entity linker

```bash
python -m gerbil
```

To run it for the GERBIL evaluation on the D2KB task

```bash
python -m gerbil d2kb
```

## Type Classifier TypeRec-BERT

To train the type classifier TypeRec-BERT

```bash
python -m typerec.train
```

To evaluate it (it might be necessary to update <tt>model_name</tt> in <tt>typerec/test.py</tt>)

```bash
python -m typerec.test
```
