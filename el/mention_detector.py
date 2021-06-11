import logging
import spacy

class SpacyMentionDetector:

    def __init__(self):
        self._logger = logging.getLogger(__name__)


    def process(self, doc):
        spacy_doc = self._nlp(doc['text'])
        doc['mentions'] = []
        for ent in spacy_doc.ents:
            self._logger.debug(f'[{ent.start_char}:{ent.end_char}] {ent.text} ({ent.label_})')
            mention = self.__spacy_ent_to_mention(ent)
            doc['mentions'].append(mention)
        self._logger.info(f'Detected {len(doc["mentions"])} mentions')
        return doc


    def __spacy_ent_to_mention(self, ent):
        return {
            'start' : ent.start_char,
            'end' : ent.end_char,
            'sf' : ent.text,
            'ner_type' : ent.label_
            }


class SpacyMentionDetectorSm(SpacyMentionDetector):

    def __init__(self):
        self._nlp = spacy.load("en_core_web_sm")
        super(SpacyMentionDetectorSm, self).__init__()


class SpacyMentionDetectorTrf(SpacyMentionDetector):

    def __init__(self):
        self._nlp = spacy.load("en_core_web_trf")
        super(SpacyMentionDetectorTrf, self).__init__()
