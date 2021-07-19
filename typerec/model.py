import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # INFO messages are not printed

import random
import sys
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, DistilBertConfig, TFDistilBertModel

from typerec import types
import utils


class TypeRecModel:
    _max_text_length = 512
    _text_vector_size = 150
    _distil_bert = 'distilbert-base-uncased'


    def __init__(self, config):
        self._config = config

        # Logging
        self._logger = logging.getLogger(__name__) # own logger
        self._tf_logger = tf.get_logger() # TensorFlow logger
        self._tf_logger.handlers = [] # remove the original handler from the TensorFlow logger
        logging.basicConfig(level=self._config.log_level, format=self._config.log_format,
                handlers=[logging.FileHandler(self._config.log_path), logging.StreamHandler()])

        # Disable eager execution
        tf.compat.v1.disable_eager_execution()

        # Use GPU (not CPU)
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)

        # Reset default graph
        tf.compat.v1.reset_default_graph()


    def __initialize(self, config):
        # Part I: Question text sequence | mention -> BERT
        bert_config = DistilBertConfig(dropout=self._config.dropout_bert,
                attention_dropout=self._config.dropout_bert_attention)
        bert_config.output_hidden_states = False
        transformer_model = TFDistilBertModel.from_pretrained(self._distil_bert, config=bert_config)

        input_text_and_mention_tokenized = Input(shape=(self._max_text_length,),
                dtype='int32', name='text_and_mention_tokenized')
        input_text_and_mention_attention_mask = Input(shape=(self._max_text_length,),
                dtype='int32', name='text_and_mention_attention_mask')

        embedding_layer = transformer_model.distilbert(input_text_and_mention_tokenized,
                attention_mask=input_text_and_mention_attention_mask)[0]
        cls_token = embedding_layer[:,0,:]
        bert_outputs = BatchNormalization()(cls_token)
        bert_outputs = Dense(self._text_vector_size)(bert_outputs)
        bert_outputs = Activation('relu')(bert_outputs)
        bert_outputs = Dense(self._config.types_count)(bert_outputs)
        bert_outputs = Activation('softmax')(bert_outputs)

        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        self._model = tf.keras.models.Model(
                inputs=[input_text_and_mention_tokenized, input_text_and_mention_attention_mask],
                outputs=bert_outputs)
        self._model.get_layer('distilbert').trainable = False # make BERT layers untrainable
        self._model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=[
                tf.keras.metrics.CategoricalAccuracy()])
        #self._model.summary()


    def __generate_data(self, dataset, batch_size):
        # https://stackoverflow.com/questions/46493419/use-a-generator-for-keras-model-fit-generator
        i = 0
        while True:
            # Get a batch from the shuffled dataset, preprocess it, and give it to the model
            batch = {
                    'text_and_mention_tokenized': [],
                    'text_and_mention_attention_mask': [],
                    'item_type_onehot': []
                    }

            # Draw a (ordered) batch from the (shuffled) dataset
            for b in range(batch_size):
                dataset_length = len(next(iter(dataset)))
                if i == dataset_length: # re-shuffle when processed whole dataset
                    i = 0
                    lists = list(zip(dataset['text_and_mention_tokenized'],
                            dataset['text_and_mention_attention_mask'], dataset['item_type_onehot']))
                    random.shuffle(lists)
                    dataset['text_and_mention_tokenized'], dataset['text_and_mention_attention_mask'], dataset['item_type_onehot'] = zip(*lists)
                    #TODO rather stop iteration?
                    # raise StopIteration

                # Add sample
                for key in dataset.keys():
                    batch[key].append(dataset[key][i])

                i += 1

            # Preprocess batch (array, pad, tokenize)
            # Already done in preprocess()/prepare(), but could be done again here if the dataset is too large
            X = {}
            X['text_and_mention_tokenized'] = np.asarray(batch['text_and_mention_tokenized'])
            X['text_and_mention_attention_mask'] = np.asarray(batch['text_and_mention_attention_mask'])
            y = np.asarray(batch['item_type_onehot'])

            yield X, y


    def train(self, dataset_train, dataset_dev, saving_dir, epochs=32, batch_size=32):
        saving_path = saving_dir + "/cp-{epoch:04d}.ckpt"

        # Initialize model
        self.__initialize(self._config)

        # Initialize callbacks
        save_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path,
                save_weights_only=False)
        logging_callback = tf.keras.callbacks.LambdaCallback(
                on_epoch_end = lambda epoch, logs: utils.log_epoch_metrics(epoch, logs))

        # Train the model
        history = self._model.fit(
                self.__generate_data(dataset_train, batch_size),
                epochs = epochs,
                steps_per_epoch=self._config.steps_per_epoch_train,
                validation_data=self.__generate_data(dataset_dev, batch_size),
                validation_steps=self._config.steps_per_epoch_dev,
                callbacks=[save_model_callback, logging_callback]
        )

        return history


    def test(self, data, saving_dir, epochs=32, batch_size=32):
        # Save metrics of all epochs in dicts
        precision_list = {}
        recall_list = {}
        f1_list = {}

        # For each epoch, load the model and evaluate it
        for epoch in range(1, epochs + 1):
            self._logger.info('')
            self._logger.info(f'--- Epoch {str(epoch)}/{str(epochs)} ---')

            # Load model from checkpoint
            self.load(os.path.join(saving_dir, f'cp-{epoch:04d}.ckpt'))
            results = self._model.evaluate([
                            data['text_and_mention_tokenized'],
                            data['text_and_mention_attention_mask']],
                    data['item_type_onehot'],
                    batch_size=batch_size)

            # For analyzing use predict(), as it returns the prediction
            entity_types = self._model.predict(data)

            # For logging
            count_all = utils.get_dataset_length(data)
            for index, pred in enumerate(entity_types):
                # Predicted type
                pred_type_onehot_vector = entity_types[index] # one-hot vector [1,0,...,0]
                pred_type_uri = types.get_type_by_onehot_vector(pred_type_onehot_vector) # Wikidata URI
                pred_type_name = types.type_dict[pred_type_uri] # Wikidata label of the type
                pred_type_index = np.argmax(pred_type_onehot_vector) # integer encoding (e.g. 1->PER, 2->ORG)
                # True type
                true_type_onehot_vector = data["item_type_onehot"][index] # one-hot vector [1,0,...,0]
                true_type_uri = types.get_type_by_onehot_vector(true_type_onehot_vector) # Wikidata URI
                true_type_name = types.type_dict[true_type_uri] # Wikidata label of the type
                true_type_index = np.argmax(true_type_onehot_vector) # integer encoding (e.g. 1->PER, 2->ORG)
                # Mention and context
                item_id = data["item_id"][index]
                text = data["text"][index]
                mention = data["item_name"][index]

                if pred_type_index == true_type_index:
                    self._logger.info(f'{str(index)}/{str(count_all)}: [TRUE] Predicted [{pred_type_name}] for "{mention}" ({item_id}) in "{text}"')
                else:
                    self._logger.info(f'{str(index)}/{str(count_all)}: [FALSE] Predicted [{pred_type_name}] instead of [{true_type_name}] for "{mention}" ({item_id}) in "{text}"')

            # Get integer label encoding from one-hot vectors (true value)
            true_labels = np.argmax(data["item_type_onehot"], axis=1)
            # Get integer label encoding from softmax probability distribution (predicted value)
            pred_labels = np.argmax(entity_types, axis=1)

            # Print metrics
            self._logger.info(f'evaluate():')
            self._logger.info(f'- Loss: {results[0]}')
            self._logger.info(f'- Categorical accuracy: {results[1]}')

            for average in ['micro', 'macro', None]:
                #average = 'macro' # None: return scores for each class; others: micro, macro, weighted
                labels = list(range(self._config.types_count)) # [0,1,2,3,...,9,10]
                zero_division = 1 # return 1 when there is a zero division
                precision = precision_score(true_labels, pred_labels,
                        average=average, labels=labels, zero_division=zero_division)
                recall = recall_score(true_labels, pred_labels,
                        average=average, labels=labels, zero_division=zero_division)
                # TODO seems to be wrong, at least not the harmonic mean of prec and recall
                f1 = f1_score(true_labels, pred_labels,
                        average=average, labels=labels, zero_division=zero_division)

                self._logger.info(f'predict() [{average}]:')
                self._logger.info(f'- Precision: {precision}')
                self._logger.info(f'- Recall:    {recall}')
                self._logger.info(f'- F1:        {f1}')

            ## Save metrics of the epochs
            precision_list[epoch] = precision
            recall_list[epoch] = recall
            f1_list[epoch] = f1

        # Calculate average of all epochs
        f1_avg = sum(f1_list.values()) / len(f1_list.values())
        precision_avg = sum(precision_list.values()) / len(precision_list.values())
        recall_avg = sum(recall_list.values()) / len(recall_list.values())
        self._logger.info('')
        self._logger.info(f'--- Average metrics for all epochs ({epochs} epochs) ---')
        self._logger.info(f'Precision (average): {precision_avg}')
        self._logger.info(f'Recall (average):    {recall_avg}')
        self._logger.info(f'F1 (average):        {f1_avg}')


    def predict(self, data, batch_size=32):
        text_and_mention_tokenized = data['text_and_mention_tokenized']
        text_and_mention_attention_mask = data['text_and_mention_attention_mask']

        # Explanation cf. comment in load()
        global sess
        global graph
        with graph.as_default():
            tf.compat.v1.keras.backend.set_session(sess)
            results = self._model.predict([text_and_mention_tokenized, text_and_mention_attention_mask],
                    batch_size=batch_size, verbose=0)

        onehot_vector = results[0]
        entity_type = types.get_type_by_onehot_vector(onehot_vector)
        type_index = np.argmax(onehot_vector)
        self._logger.info(f'Entity type: {types.get_type_label(entity_type)} ({entity_type}, index {type_index})')

        return entity_type


    def load(self, filepath, checkpoint_type='model'):
        """
        Load a model from a file. This overwrites the model built in __init__() stored at self._model.
        """
        self._logger.info(f'Load model from {filepath}')

        # Flask creates a thread for each request. As the model is generated during the first request,
        # it is not visible in the next request. Therefore, create a global session that is used throughout
        # all following requests.
        global sess
        global graph
        sess = tf.compat.v1.Session()
        graph = tf.compat.v1.get_default_graph()
        tf.compat.v1.keras.backend.set_session(sess)

        if checkpoint_type == 'weights':
            #model = ELModel()
            #model._model.load_weights(filepath)
            self._logger.error(f'Checkpoint type {checkpoint_type} not supported')
        else: # == 'model'
            self._model = tf.keras.models.load_model(filepath)
            #self._config = config # this is done already on __init__() and it shouldn't change
