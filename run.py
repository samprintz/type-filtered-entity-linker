import sys
from inout import dataset
import preprocess
from el.model import ELModel
from el.entity_linker import EntityLinker


def main():
    text = sys.argv[1]

    # Load model
    #model = ELModel()
    #model.load('model')

    # TODO Workaround: re-train model, as loading of the saved model doesn't work
    #train_data_raw = dataset.get_wikidata_disamb_dataset('train', 'small')
    #train_data_pre = preprocess.prepare_dataset(train_data_raw,
    #        sample_mode=True,
    #        use_cache=True)
    #train_data = preprocess.reshape_dataset(train_data_pre)

    model = ELModel()
    #model.train(train_data,
    #    epochs=2,
    #    batch_size=32)

    # Initialize linker and do the entity linking
    linker = EntityLinker(model)
    mentions, candidates, entities = linker.process(text)
    print("Done")


if __name__ == '__main__':
    main()

