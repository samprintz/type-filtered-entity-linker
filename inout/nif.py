from pynif import NIFCollection

def generate_nif(doc):
    collection = NIFCollection()

    context = collection.add_context(
            uri=doc['uri'],
            mention=doc['text'])

    for mention in doc['mentions']:
        if mention['entity'] is not None:
            context.add_phrase(
                    beginIndex=mention['start'],
                    endIndex=mention['end'],
                    taClassRef=[], # TODO
                    score=mention['entity']['score'], # TODO format?
                    annotator='http://test.org/', # TODO
                    taIdentRef=mention['entity']['item_url'],
                    taMsClassRef='') # TODO
        else: # if a mention was detected, but no entity was found
            context.add_phrase(
                    beginIndex=mention['start'],
                    endIndex=mention['end'],
                    annotator='http://test.org/') # TODO

    generated_nif = collection.dumps(format='turtle')
    return generated_nif


def read_nif(nif_data):
    # TODO try catch?
    doc = {}
    parsed = NIFCollection.loads(nif_data, format='turtle')

    doc['uri'] = parsed.contexts[0].uri.toPython()
    doc['text'] = parsed.contexts[0].mention
    doc['mentions'] = []
    for phrase in parsed.contexts[0].phrases:
        doc['mentions'].append(gerbil_phrase_to_mention(phrase))

    return doc


def gerbil_phrase_to_mention(phrase):
    return {
        'start' : phrase.beginIndex,
        'end' : phrase.endIndex,
        'sf' : phrase.mention
        }
