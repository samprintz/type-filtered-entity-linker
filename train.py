from inout import dataset
import preprocess

# TODO Adapter to GERBIL
# TODO Load data: Load training set (receive from GERBIL)
# TODO Load data: Load Wikidata item embeddings from Wikidata PyTorch Big Graph keyed vector index
# TODO Candidate Generation
# TODO Part Ia: BERT for surface form embedding
# TODO Part Ib: BERT for surface form NER classification
# TODO Filter entity candidates by NER type
# TODO Part II: PyTorch Big Graph embeddings for entity candidate
# TODO Part III: Comparator MLP
# TODO Training
# TODO Testing/prediction


def main():
    data_raw = dataset.get_wikidata_disamb_train_dataset('sample')
    data = preprocess.prepare_dataset(data_raw)

    model = ELModel()
    model.train(data,
        epochs=2,
        batch_size=32)



if __name__ == '__main__':
    main()

