import os
import sys
from inout import dataset
import preprocess
from el.model import ELModel
from el.entity_linker import EntityLinker


def main():
    try:
        text = sys.argv[1]
    except:
        text = "Napoleon was the first emperor of the French empire."

    # Load model
    model_type = 'rnn'
    model_name = 'model-20210428-1'
    model_checkpoint = 60
    filepath = os.path.join(os.getcwd(), 'data', 'models', model_type, model_name, f'cp-{model_checkpoint:04d}.ckpt')
    model = ELModel()
    model.load(filepath)

    # Initialize linker and do the entity linking
    linker = EntityLinker(model)
    mentions, candidates, entities = linker.process(text)
    print("Done")


if __name__ == '__main__':
    main()

