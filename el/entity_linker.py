class EntityLinker:

    def __init__(self):
        pass

    def process(self, text):
        mentions = detect_mentions(text)
        candidates = generate_candidates(text, mentions)
        entities = disambiguate_entities(text, candidates)
        return entities


    def detect_mentions(self, text):
        # TODO
        return mentions


    def generate_candidates(self, text, mentions):
        # TODO
        return candidates


    def disambiguate_entities(self, text, candidates):
        # TODO
        return entities

