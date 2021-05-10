from pynif import NIFCollection

def generate_nif(doc):
    collection = NIFCollection()

    context = collection.add_context(
            uri="http://test.org/doc1", # TODO
            mention=doc['text'])

    for mention in doc['mentions']:
        context.add_phrase(
                beginIndex=mention['start'],
                endIndex=mention['end'],
                taClassRef=[], # TODO
                score=mention['entity']['score'], # TODO format?
                annotator='http://test.org/', # TODO
                taIdentRef=mention['entity']['item_url'],
                taMsClassRef='') # TODO

    generated_nif = collection.dumps(format='turtle')
    print(generated_nif)
    return generated_nif

