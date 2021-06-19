import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # INFO messages are not printed

import random
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Concatenate, BatchNormalization
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, DistilBertConfig, TFDistilBertModel


class TypeRecModel:
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

        # Part I: Question text sequence | mention -> BERT
        config = DistilBertConfig(dropout=0.2, attention_dropout=0.2) # TODO dropout as config parameter
        config.output_hidden_states = False
        transformer_model = TFDistilBertModel.from_pretrained(self._distil_bert, config=config)

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
        bert_outputs = Dense(self._output_size)(bert_outputs)
        bert_outputs = Activation('sigmoid')(bert_outputs)

        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        self._model = tf.keras.models.Model(
                inputs=[input_text_and_mention_tokenized, input_text_and_mention_attention_mask],
                outputs=bert_outputs)
        self._model.get_layer('distilbert').trainable = False # make BERT layers untrainable
        self._model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[
                tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        #self._model.summary()


    def __generate_data(self, dataset, batch_size):
        # https://stackoverflow.com/questions/46493419/use-a-generator-for-keras-model-fit-generator
        i = 0
        while True:
            # Get a batch from the shuffled dataset, preprocess it, and give it to the model
            batch = {
                    'text_and_mention_tokenized': [],
                    'text_and_mention_attention_mask': [],
                    'item_type_index': []
                    }

            # Draw a (ordered) batch from the (shuffled) dataset
            for b in range(batch_size):
                dataset_length = len(next(iter(dataset)))
                if i == dataset_length: # re-shuffle when processed whole dataset
                    i = 0
                    lists = list(zip(dataset['text_and_mention_tokenized'],
                            dataset['text_and_mention_attention_mask'], dataset['item_type_index']))
                    random.shuffle(lists)
                    dataset['text_and_mention_tokenized'], dataset['text_and_mention_attention_mask'], dataset['item_type_index'] = zip(*lists)
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
            y = np.asarray(batch['item_type_index'])

            yield X, y


    def train(self, datasets, saving_dir, epochs=32, batch_size=32):
        saving_path = saving_dir + "/cp-{epoch:04d}.ckpt"

        # Initialize callbacks
        save_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path,
                save_weights_only=False)
        logging_callback = tf.keras.callbacks.LambdaCallback(
                on_epoch_end = lambda epoch, logs: self._logger.info(f'\nEpoch {epoch + 1}: loss: {logs["loss"]} - accuracy: {logs["accuracy"]} - val_loss: {logs["val_loss"]} - val_accuracy: {logs["val_accuracy"]}')
        )

        # Train dataset
        dataset_train = datasets[0]
        dataset_length_train = len(next(iter(dataset_train)))
        steps_per_epoch = dataset_length_train // batch_size
        if steps_per_epoch < 1:
            steps_per_epoch = 1

        # Validation dataset
        dataset_val = datasets[1]
        dataset_length_val = len(next(iter(dataset_val)))
        validation_steps_per_epoch = dataset_length_val // batch_size
        if validation_steps_per_epoch < 1:
            validation_steps_per_epoch = 1

        # Print training settings
        self._logger.info('')
        self._logger.info('=== Training settings ===')
        self._logger.info(f'epochs={epochs}, batch_size={batch_size}, dataset_length_train={dataset_length_train}, dataset_length_val={dataset_length_val}, steps_per_epoch={steps_per_epoch}')

        # Train the model
        history = self._model.fit(
                self.__generate_data(dataset_train, batch_size),
                epochs = epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=self.__generate_data(dataset_val, batch_size),
                validation_steps=validation_steps_per_epoch,
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
            self.load(os.path.join(saving_dir, f'cp-{epoch:04d}.ckpt'))
            results = self._model.evaluate([
                            data['text_and_mention_tokenized'],
                            data['text_and_mention_attention_mask']],
                    data['item_type_index'],
                    batch_size=batch_size)

            # Get metrics
            precision = results[2]
            recall = results[3]
            f1 = 2 * 1 / (1 / results[2] + 1 / results[3])

            self._logger.info(f'Loss: {results[0]}')
            self._logger.info(f'Accuracy: {results[1]}')
            self._logger.info(f'Precision: {precision}')
            self._logger.info(f'Recall: {recall}')
            self._logger.info(f'F1: {f1}')

            # Save metrics of the epochs
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
        self._logger.info(f'Recall (average): {recall_avg}')
        self._logger.info(f'F1 (average): {f1_avg}')


    def predict(self, data, batch_size=32):
        text_tokenized = data['text_tokenized']
        text_attention_masks = data['text_attention_mask']
        text_mention_masks = data['text_mention_mask']
        item_pbg = data['item_pbg']
        item_glove = data['item_glove']

        # Explanation cf. comment in load()
        global sess
        global graph
        with graph.as_default():
            tf.compat.v1.keras.backend.set_session(sess)
            results = self._model.predict([text_tokenized, text_attention_masks, text_mention_masks, item_pbg, item_glove],
                    batch_size=batch_size, verbose=0)

        self._logger.debug(f'Prediction: {results}')
        score = results[0]
        self._logger.debug(f'Score: {score}')

        return score


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
