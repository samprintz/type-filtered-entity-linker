import logging
import spacy

class SpacyMentionDetector:

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._nlp = spacy.load("en_core_web_sm")

    def process(self, text):
        doc = self._nlp(text)
        mentions = doc.ents
        for mention in doc.ents:
            self._logger.debug(f'[{mention.start_char}:{mention.end_char}] {mention.text} ({mention.label_})')
        return mentions

