from el.mention_detector import SpacyMentionDetector

class EntityLinker:

    def __init__(self, model):
        self._model = model

    def process(self, text):
        mentions = self.detect_mentions(text)
        candidates = self.generate_candidates(text, mentions)
        entities = self.disambiguate_entities(text, candidates)
        return mentions, candidates, entities


    def detect_mentions(self, text):
        self.__print_step_heading('Mention Detection')
        mention_detector = SpacyMentionDetector()
        mentions = mention_detector.process(text)
        print(mentions)
        return mentions


    def generate_candidates(self, text, mentions):
        self.__print_step_heading('Candidate Generation')
        # TODO
        candidates = mentions
        print("TODO")
        #print(candidates)
        return candidates


    def disambiguate_entities(self, text, candidates):
        self.__print_step_heading('Entity Disambiguation')
        # TODO
        entities = candidates
        print("TODO")
        #print(entities)
        return entities

    def __print_step_heading(self, step_heading):
        print(f'=== {step_heading} ===')

