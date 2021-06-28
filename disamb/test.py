from inout import dataset
import preprocess
from disamb.model import EDModel


def main():
    data_raw = dataset.get_wikidata_disamb_dataset('test', 'small')
    data_pre = preprocess.prepare_dataset(data_raw,
            sample_mode=True,
            use_cache=True)
    data = preprocess.reshape_dataset(data_pre)

    #model = ELModel()
    #model.load('model')

    # TODO Workaround: re-train model, as loading of the saved model doesn't work
    train_data_raw = dataset.get_wikidata_disamb_dataset('train', 'small')
    train_data_pre = preprocess.prepare_dataset(train_data_raw,
            sample_mode=True,
            use_cache=True)
    train_data = preprocess.reshape_dataset(train_data_pre)

    model = EDModel()
    model.train(train_data,
        epochs=2,
        batch_size=32)

    model.test(data,
        batch_size=32)


if __name__ == '__main__':
    main()

