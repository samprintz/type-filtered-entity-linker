from inout import dataset
import preprocess
from el.model import ELModel


def main():
    data_raw = dataset.get_wikidata_disamb_train_dataset('small')
    data_pre = preprocess.prepare_dataset(data_raw,
            sample_mode=True,
            use_cache=True)
    data = preprocess.reshape_dataset(data_pre)

    model = ELModel()
    model.train(data,
        epochs=2,
        batch_size=32)

    model.save('model')


if __name__ == '__main__':
    main()

