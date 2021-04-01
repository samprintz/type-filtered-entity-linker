import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, GRU, LSTM, Bidirectional, Activation, Dropout, Concatenate, BatchNormalization
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, DistilBertConfig, TFDistilBertModel


tf.compat.v1.disable_eager_execution()

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)


class Model():
    _max_sentence_length = 512
    _nodes_vocab_size = 300 * 3
    _question_vocab_size = 300
    _nodes_vector_size = 150
    _question_vector_size = 150
    _bert_embedding_size = 768
    _hidden_layer2_size = 250
    _output_size = 2

    _distil_bert = 'distilbert-base-uncased'
    _memory_dim = 100
    _stack_dimension = 2


    def __init__():
    tf.config.experimental.set_memory_growth(gpu, True)


    def train(self, data, epochs=20, batch_size=32):
        tokens = data['tokens']
        ner_tags = data['ner_tags']

        # Tokenize
        tokenizer = DistilBertTokenizer.from_pretrained(self._distil_bert, do_lower_case=True, add_special_tokens=True,
                max_length=self._max_sentence_length, pad_to_max_length=True)
        questions, attention_masks, segments = self.__tokenize(sentences, tokenizer, self._max_sentence_length)

        # Part I: Question text sequence -> BERT
        config = DistilBertConfig(dropout=0.2, attention_dropout=0.2)
        config.output_hidden_states = False
        transformer_model = TFDistilBertModel.from_pretrained(self._distil_bert, config=config)

        input_question = Input(shape=(self._max_sentence_length,), dtype='int32')
        input_attention_mask = Input(shape=(self._max_sentence_length,), dtype='int32')
        input_sf_mask = Input(shape=(self._max_sentence_length, self._bert_embedding_size), dtype='float32')

        embedding_layer = transformer_model.distilbert(input_question, attention_mask=input_attention_mask)[0]
        #cls_token = embedding_layer[:,0,:]
        sf_token = tf.math.reduce_mean(tf.math.multiply(embedding_layer, input_sf_mask), axis=1)
        question_outputs = BatchNormalization()(sf_token)
        question_outputs = Dense(self._question_vector_size)(question_outputs)
        question_outputs = Activation('relu')(question_outputs)

        # Part II: Entity graph node (as text) -> Bi-LSTM
        fw_lstm = LSTM(self._memory_dim)
        bw_lstm = LSTM(self._memory_dim, go_backwards=True)

        input_nodes = Input(shape=(None, self._nodes_vocab_size))
        nodes_outputs = Bidirectional(layer=fw_lstm, backward_layer=bw_lstm)(input_nodes)
        nodes_outputs = Dense(self._nodes_vector_size)(nodes_outputs)
        nodes_outputs = Activation('relu')(nodes_outputs)

        # Part III: Comparator -> MLP
        # concatenation size = _nodes_vector_size + _question_vector_size
        concatenated = Concatenate(axis=1)([question_outputs, nodes_outputs])
        mlp_outputs = Dense(self._hidden_layer2_size)(concatenated)
        mlp_outputs = Activation('relu')(mlp_outputs)
        mlp_outputs = Dropout(0.2)(mlp_outputs)
        mlp_outputs = Dense(self._output_size)(mlp_outputs) # 2-dim. output
        mlp_outputs = Activation('softmax')(mlp_outputs)

        # Compile and fit the model
        self._model = tf.keras.models.Model(inputs=[input_question, input_attention_mask, input_sf_mask, input_nodes], outputs=mlp_outputs)
        self._model.get_layer('distilbert').trainable = False # make BERT layers untrainable
        self._model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
        self._model.summary()
        self._model.fit([questions, attention_masks, sf_mask, node_X], y, epochs=epochs, batch_size=batch_size)

    def tokenize(self, sentences, tokenizer, max_length):
        input_ids, input_masks, input_segments = [],[],[]
        for sentence in sentences:
            inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=max_length,
                    padding='max_length', return_attention_mask=True, return_token_type_ids=True)
            input_ids.append(inputs['input_ids'])
            input_masks.append(inputs['attention_mask'])
            input_segments.append(inputs['token_type_ids'])
        return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(input_segments, dtype='int32')

       

    def test():


