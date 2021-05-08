import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # INFO messages are not printed
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, GRU, LSTM, Bidirectional, Activation, Dropout, Concatenate, BatchNormalization
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, DistilBertConfig, TFDistilBertModel


class ELModel:
    _max_text_length = 512
    _item_pbg_vocab_size = 200
    _item_glove_vocab_size = 300 * 3

    _text_vector_size = 150
    _item_vector_size = 150

    _bert_embedding_size = 768
    _hidden_layer2_size = 250
    _output_size = 1

    _distil_bert = 'distilbert-base-uncased'
    _memory_dim = 100
    _stack_dimension = 2


    def __init__(self):
        self._logger = logging.getLogger(__name__)

        tf.compat.v1.disable_eager_execution()
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.compat.v1.reset_default_graph()

        # Part I: Question text sequence -> BERT
        config = DistilBertConfig(dropout=0.2, attention_dropout=0.2)
        config.output_hidden_states = False
        transformer_model = TFDistilBertModel.from_pretrained(self._distil_bert, config=config)

        input_text_tokenized = Input(shape=(self._max_text_length,), dtype='int32', name='text_tokenized')
        input_text_attention_mask = Input(shape=(self._max_text_length,), dtype='int32', name='text_attention_mask')
        input_text_sf_mask = Input(shape=(self._max_text_length, self._bert_embedding_size), dtype='float32', name='text_sf_mask')

        embedding_layer = transformer_model.distilbert(input_text_tokenized, attention_mask=input_text_attention_mask)[0]
        #cls_token = embedding_layer[:,0,:]
        sf_token = tf.math.reduce_mean(tf.math.multiply(embedding_layer, input_text_sf_mask), axis=1)
        question_outputs = BatchNormalization()(sf_token)
        question_outputs = Dense(self._text_vector_size)(question_outputs)
        question_outputs = Activation('relu')(question_outputs)

        # Part II-A: Entity -> PyTorch Big Graph embedding
        input_item_pbg = Input(shape=(self._item_pbg_vocab_size), name='item_pbg')
        item_pbg_outputs = Dense(self._item_vector_size)(input_item_pbg)
        item_pbg_outputs = Activation('relu')(item_pbg_outputs)

        # Part II-B: Entity -> GCN/GAT
        # TODO copy from other model

        # Part II-C: Entity graph node (as text) -> Bi-LSTM
        fw_lstm = LSTM(self._memory_dim)
        bw_lstm = LSTM(self._memory_dim, go_backwards=True)

        # TODO rename to input_item_glove?
        input_item_embedded = Input(shape=(None, self._item_glove_vocab_size), name='item_embedded')
        item_embedded_outputs = Dense(self._item_vector_size)(input_item_embedded)
        item_embedded_outputs = Activation('relu')(item_embedded_outputs)

        # Part III: Comparator -> MLP
        concatenated = Concatenate(axis=1)([question_outputs, item_pbg_outputs])
        mlp_outputs = Dense(self._hidden_layer2_size)(concatenated)
        mlp_outputs = Activation('relu')(mlp_outputs)
        mlp_outputs = Dropout(0.2)(mlp_outputs)
        mlp_outputs = Dense(self._output_size)(mlp_outputs) # 2-dim. output
        mlp_outputs = Activation('softmax')(mlp_outputs)

        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        self._model = tf.keras.models.Model(inputs=[input_text_tokenized, input_text_attention_mask, input_text_sf_mask, input_item_pbg, input_item_embedded], outputs=mlp_outputs)
        self._model.get_layer('distilbert').trainable = False # make BERT layers untrainable
        self._model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        #self._model.summary()


    def train(self, data, epochs=20, batch_size=32):
        # TODO use generator
        text_tokenized = data['text_tokenized']
        text_attention_masks = data['text_attention_mask']
        text_sf_masks = data['text_sf_mask']
        item_pbg = data['item_pbg']
        # TODO rename to item_glove?
        item_embedded = data['item_embedded']
        answer = data['answer']
        #ner_tags = data['ner_tags']

        self._model.fit([text_tokenized, text_attention_masks, text_sf_masks, item_pbg, item_embedded],
                answer, epochs=epochs, batch_size=batch_size)

       
    def test(self, data, batch_size=32):
        text_tokenized = data['text_tokenized']
        text_attention_masks = data['text_attention_mask']
        text_sf_masks = data['text_sf_mask']
        item_pbg = data['item_pbg']
        item_embedded = data['item_embedded']
        answer = data['answer']
        #ner_tags = data['ner_tags']

        results = self._model.evaluate([text_tokenized, text_attention_masks, text_sf_masks, item_pbg, item_embedded],
                answer, batch_size=batch_size)

        self._logger.info(f'Loss: {results[0]}')
        self._logger.info(f'Accuracy: {results[1]}')


    def predict(self, data, batch_size=32):
        text_tokenized = data['text_tokenized']
        text_attention_masks = data['text_attention_mask']
        text_sf_masks = data['text_sf_mask']
        item_pbg = data['item_pbg']
        item_embedded = data['item_embedded']

        # TODO Dependending on the model, results is either a 2-dim. array (load(Cetoli)) or one value (__init__()).
        # How to handle this? Change completely to my models and don't use Cetoli's anymore? Have two ELModel() classes, one for the loaded models, one for mine?
        results = self._model.predict([text_tokenized, text_attention_masks, text_sf_masks, item_pbg, item_embedded],
                batch_size=batch_size)

        self._logger.debug(f'Prediction: {results}')
        score = self.__standardize_result(results[0])
        self._logger.debug(f'Score: {score}')

        return score

    def __standardize_result(self, result):
        """
        In Cetoli's models, the first index represents irrelevance (max. [0., 1.]), the second index relevance (max. [1., 0.]).
        If the relevance is higher than the irrelevance, return the relevance score (= matching score). Otherwise 0 (not matching).
        """
        if result[0] < result[1]:
            return result[1] # matching score
        return 0. # not matching


    def save(self, filepath):
        self._logger.info(f'Saving into {filepath}')
        self._model.save(filepath)

    def load(self, filepath):
        """
        Load a model from a file. This overwrites the model built in __init__() stored at self._model.
        """
        self._logger.info(f'Load model from {filepath}')
        self._model = tf.keras.models.load_model(filepath)

