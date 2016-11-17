"""Module for breaking words into syllables."""
from datetime import datetime
import os
import re
from time import time
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.nn import bidirectional_dynamic_rnn as brnn
from tensorflow.python.ops.nn import dynamic_rnn as rnn


class SyllableParser(object):
    """Tensorflow model for parsing syllables."""

    def __init__(self, num_epochs=100, batch_size=20, hidden_size=128,
                 cell_type='lstm', net_type='brnn', num_layers=3, treshold=0.5):
        """Init SyllableParser.

        Args:
            num_epochs: A non-negative integer, which represents number of
                training epochs. Defaults to 100.
            batch_size: A non-negative integer, which represents size of
                training batch. Defaults to 20.
            hidden_size: A non-negative integer, which represents size of RNN
                hidden vector. Defaults to 128.
            cell_type: A string, which represents type of RNN cell used.
                Can be either 'lstm' or 'gru'. Defaults to 'lstm'.
            net_type: A string, which represents type of RNN used.
                Can be either 'rnn' or 'brnn'. 'rnn' denotes unidirectional rnn
                and 'brnn' denotes bidirectional rnn.
            num_layers: A non-negative integer, which represents number of
                hidden layers in RNN. Defaults to 3.
            treshold: A float in range [0, 1], which represents syllable
                probability treshold. If probablity of syllable gap at some
                point in word is greater than treshold, then we break the word
                at that point.
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.net_type = net_type
        self.treshold = treshold

    def encode(self, word):
        """Encode word according to mapping from letters to indices.

        Note: self.mapping must be constructed before calling this function.
        Args:
            word: A string to encode.
        Returns:
            word encoding: A list of integers of length equal to number of
                letters in word.
        Raises:
            ValueError: if word contains characters, which are not present in
                mapping.
        """
        return list(map(lambda x: self.mapping.index(x),
                        list(word.strip().lower())))

    def decode(self, encoded_word):
        """Decode list of integers according to mapping from letters to indices.

        Args:
            encoded_word: A list of integers to decode as returned by encode
                function.
        Returns:
            word: A decoded string.
        Raises:
            IndexError: if integer in encoded_word is not in range
                [0, len(self.mapping)].
        """
        return ''.join(map(lambda x: self.mapping[x],
                           list(map(int, encoded_word))))

    def encode_syllables(self, syllables):
        """Encode all syllables in list.

        Args:
            syllables: A list of strings to be encoded.
        Returns:
            encoded_syllables: A list of integer lists - one for each syllable.
        """
        encoded_syllables = []
        for syllable in syllables:
            if len(syllable.strip()) > 0:
                encoded_syllables.extend(([0] * (len(syllable.strip()) - 1) +
                                          [1]))
        return encoded_syllables

    def pad_into_matrix(self, rows, padding_value=0):
        """Pad rows with padding_value, so that they all have the same length.

        Args:
            rows: list of lists or np.arrays.
            padding_value: value to pad rows with. Defaults to 0. Type of
                padding value should match type of elements in rows.
        Returns:
            matrix: np.array of shape (num_rows, max_row_length), containing
                padded rows.
        """
        matrix = []
        lengths = np.array(list(map(len, rows)))
        matrix_width = np.max(lengths)

        for row in rows:
            if len(row) < matrix_width:
                matrix.append(np.hstack((np.array(row),
                              np.array([padding_value] * (matrix_width -
                                                          len(row))))))
            else:
                matrix.append(np.array(row))
        matrix = np.vstack(matrix)
        return matrix, lengths

    def fit_data(self, filename):
        r"""Parse and encode training data.

        Args:
            filename: A string, which represents name of file with training
                data. Each line of the file should be in format:
                'word\tsyl lab les'.
        """
        with open(filename) as fin:
            self.mapping = list(set(list(''.join(fin.readlines()))))
            fin.seek(0)
            X = []
            y = []
            for idx, line in enumerate(fin):
                if idx == 0:
                    continue  # this is csv header
                tokens = re.split(r'\t', line)
                encoded_word = self.encode(tokens[0])
                encoded_syllables = self.encode_syllables(re.split(r'\s+',
                                                                   tokens[1]))
                X.append(encoded_word)
                y.append(encoded_syllables)
            self.X, self.lengths = X, list(map(len, X))
            self.y = y

    def prepare_test_data(self, filename):
        """Parse and encode test data.

        Args:
            filename: A string, which represents a name of file with test data.
                Each line of the file should contain one word.
        """
        X = []
        with open(filename) as fin:
            for line in fin:
                X.append(self.encode(line))
        self.X_test, self.test_lengths = X, list(map(len, X))

    def decode_prediction(self, words, predicted_labels, lengths):
        """Construct human-readable syllables out of predicted syllable labels.

        Args:
            words: A list of integer lists or np.arrays or 2d np.array, which
                contains encoded words as returned by encode.
            predicted_labels: A 2d np.array, which contains syllable labels for
                every letter for every padded word in words.
            lengths: A list or 1d np.array of lengths for every padded word
                in words.
        """
        items = []
        for i in range(len(words)):
            word, pred, length = words[i], predicted_labels[i], lengths[i]
            syllables = []
            syllable = []
            for ch, idx in zip(list(self.decode(word[:length])), pred[:length]):
                if idx == 0:
                    syllable.append(ch)
                if idx == 1:
                    syllable.append(ch)
                    syllables.append(''.join(syllable))
                    syllable = []
            if len(syllable) > 0:
                syllables.append(''.join(syllable))
            items.append((self.decode(word[:length]), syllables))
        return items

    def get_batches(self, mode):
        """Generate batches of training, testing or validation data.

        Args:
            mode: A string, representing type of data. Can be 'train', 'test'
                or 'val'.
        Yields:
            X_batch: A 2d np.array with encoded words.
            y_batch: A 2d np.array with syllable labels for each word in
                X_batch.
            lengths_batch: A 1d np.array with lengths for each word in X_batch.
        Raises:
            ValueError: Unknown mode.
        """
        train_size = int(0.9 * len(self.X))
        if mode == 'train':
            X, y = self.X[:train_size], self.y[:train_size]
            lengths = self.lengths[:train_size]
        elif mode == 'val':
            X, y = self.X[train_size:], self.y[train_size:]
            lengths = self.lengths[train_size:]
        elif mode == 'test':
            X, y = self.X_test, self.y[:len(self.X_test)]
            lengths = self.test_lengths
        else:
            raise ValueError('Unknown mode.')
        X_batch, y_batch, lengths_batch = [], [], []
        for idx, x_sample in enumerate(X):
            if idx > 0 and idx % self.batch_size == 0:
                X_batch, lengths_batch = self.pad_into_matrix(X_batch, 0)
                y_batch, _ = self.pad_into_matrix(y_batch, 0)
                yield X_batch, y_batch, lengths_batch
                X_batch, y_batch, lengths_batch = [], [], []
            X_batch.append(x_sample)
            y_batch.append(y[idx])
            lengths_batch.append(lengths[idx])

    def accuracy(self, true_syllables, pred_syllables, seq_lengths):
        """Compute masked accuracy.

        Args:
            true_syllables: 2d np.array with true syllable labels.
            pred_syllables: 2d np.array with predicted syllable labels.
            seq_lengths: 1d np.array with lengths for each syllable
                labels sequence.
        Returns:
            accuracy: masked accuracy of prediction (float).
        """
        num_true_predictions = 0
        for i in range(len(true_syllables)):
            length = seq_lengths[i]
            true_syllable = true_syllables[i, :length]
            pred_syllable = pred_syllables[i, :length]
            if np.all(np.equal(true_syllable, pred_syllable)):
                num_true_predictions += 1
        # results = np.all(np.equal(true_syllables, pred_syllables), axis=1)
        # true_predictions = len(results[results == True])
        return num_true_predictions / float(len(true_syllables))

    def construct_graph(self):
        """Construct Tensorflow graph."""
        self.graph = tf.Graph()
        hidden_state_size = self.hidden_size
        if self.net_type == 'brnn':
            hidden_state_size *= 2
        with self.graph.as_default():
            self.words = tf.placeholder(tf.int32, shape=(self.batch_size, None),
                                        name='words')
            self.syllable_labels = tf.placeholder(tf.int32,
                                                  shape=(self.batch_size, None),
                                                  name='syllable_labels')
            self.seq_lengths = tf.placeholder(tf.int32, shape=(self.batch_size),
                                              name='lengths')
            W = tf.Variable(tf.truncated_normal([hidden_state_size, 2]),
                            dtype=tf.float32)
            b = tf.Variable(np.zeros([2]), dtype=tf.float32)
            embedding_matrix = tf.Variable(tf.truncated_normal(
                                           [len(self.mapping),
                                            self.hidden_size],
                                           stddev=np.sqrt(2.0 / self.hidden_size
                                                          )))
            embedding = tf.nn.embedding_lookup(embedding_matrix, self.words)
            treshold = tf.Variable(np.array([self.treshold]), dtype=tf.float32,
                                   name='treshold')
            if self.cell_type == 'lstm':
                cell = rnn_cell.LSTMCell(self.hidden_size)
            elif self.cell_type == 'gru':
                cell = rnn_cell.GRUCell(self.hidden_size)
            else:
                raise ValueError('Unknown cell type.')
            rnn_multicell = rnn_cell.MultiRNNCell([cell] * self.num_layers)
            if self.net_type == 'rnn':
                self.outputs, _ = rnn(rnn_multicell, embedding,
                                      sequence_length=self.seq_lengths,
                                      dtype=tf.float32, swap_memory=True)
            elif self.net_type == 'brnn':
                self.outputs, _ = brnn(rnn_multicell, rnn_multicell,
                                       embedding,
                                       sequence_length=self.seq_lengths,
                                       dtype=tf.float32, swap_memory=True)
                self.outputs = tf.concat(2, self.outputs)
            # print(self.outputs.get_shape())
            outputs_reshape = tf.reshape(self.outputs, [-1, hidden_state_size])
            # print(outputs_reshape.get_shape())
            # print(W.get_shape())
            logits = tf.matmul(outputs_reshape, W) + b
            self.logits = tf.reshape(logits, [self.batch_size, -1, 2])
            # print(self.logits.get_shape())
            # print(self.syllable_labels.get_shape())
            # self.prediction = tf.argmax(self.logits, 2)
            probs = tf.nn.softmax(self.logits)
            # print(probs.get_shape())
            sliced_probs = tf.slice(probs, [0, 0, 1], [-1, -1, -1])
            greater = tf.greater(sliced_probs, treshold)
            # print(greater.get_shape())
            self.separation_indices = tf.where(greater)
            self.prediction = tf.zeros_like(greater)
            # print(sliced_probs.get_shape())
            # print(self.prediction.get_shape())
            unmasked_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        self.logits, self.syllable_labels)
            ce_mask = tf.sequence_mask(self.seq_lengths,
                                       tf.reduce_max(self.seq_lengths),
                                       dtype=tf.float32)
            lengths_mask = tf.sequence_mask(self.seq_lengths,
                                            tf.reduce_max(self.seq_lengths),
                                            dtype=tf.float32)
            self.loss = (tf.reduce_sum(unmasked_ce * ce_mask) /
                         tf.reduce_sum(lengths_mask))
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
            self.saver = tf.train.Saver()

    def run_session(self, checkpoints_dir):
        """Run Tensorflow model training.

        Args:
            checkpoints_dir: A string, which represents path to directory,
                where to save model checkpoints.
        """
        with self.graph.as_default():
            self.session = tf.Session()
            self.session.run(tf.initialize_all_variables())
            print("Checking for checkpoints...")
            latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
            if latest_checkpoint is not None:
                print("Found checkpoints, using them.")
                self.saver.restore(self.session, latest_checkpoint)
            else:
                print("No checkpoints found, starting training from scratch.")
            accuracies = []
            for epoch in range(self.num_epochs):
                print("Starting epoch {}".format(epoch))
                batch_losses = []
                start = time()
                for (words_batch,
                     syllable_labels_batch,
                     lengths_batch) in self.get_batches('train'):
                    feed_dict = {self.words: words_batch,
                                 self.syllable_labels: syllable_labels_batch,
                                 self.seq_lengths: lengths_batch}
                    if words_batch.shape != syllable_labels_batch.shape:
                        prediction = self.decode_prediction(
                                     words_batch,
                                     syllable_labels_batch, lengths_batch)
                        print("Bad train examples:")
                        for word, syllables in prediction:
                            print('\t', word, ' '.join(syllables))

                    batch_loss, _ = self.session.run([self.loss,
                                                      self.optimizer],
                                                     feed_dict=feed_dict)
                    batch_losses.append(batch_loss)
                end = time()
                epoch_result = 'Epoch {} done. Loss: {}. Training took {} sec.'
                print(epoch_result.format(epoch, np.mean(batch_losses),
                                          end - start))
                val_losses = []
                for (val_words_batch,
                     val_syllable_label_batch,
                     val_lengths_batch) in self.get_batches('val'):
                    feed_dict = {self.words: val_words_batch,
                                 self.syllable_labels: val_syllable_label_batch,
                                 self.seq_lengths: val_lengths_batch}
                    (pred, indices,
                     val_loss) = self.session.run([self.prediction,
                                                   self.separation_indices,
                                                   self.loss],
                                                  feed_dict=feed_dict)
                    pred[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
                    val_losses.append(val_loss)
                    pred = pred.reshape((pred.shape[0],
                                         pred.shape[1])).astype(np.int32)
                print('Validation loss: %f' % np.mean(val_losses))
                accuracy = self.accuracy(val_syllable_label_batch,
                                         pred, val_lengths_batch)
                accuracies.append(accuracy)
                print('Accuracy: %f' % accuracy)
                indices = np.random.choice(np.arange(len(val_words_batch)), 3)
                prediction = self.decode_prediction(val_words_batch[indices],
                                                    pred[indices],
                                                    val_lengths_batch[indices])
                true_values = self.decode_prediction(val_words_batch[indices],
                                                     val_syllable_label_batch[
                                                     indices],
                                                     val_lengths_batch[indices])
                # print(pred[sample_indices],
                #       val_syllable_label_batch[sample_indices])
                print("Sample predictions:")
                for ((word, syllables),
                     (word, true_syllables)) in zip(prediction, true_values):
                    print('\t', word, ' '.join(syllables), ' | ',
                          ' '.join(true_syllables))
                if epoch % 10 == 0:
                    print("Saving model...")
                    checkpoint_name = ("checkpoint" +
                                       datetime.now().strftime("_%d.%m.%y_%H:%M"
                                                               ))
                    save_path = self.saver.save(self.session,
                                                os.path.join(checkpoints_dir,
                                                             checkpoint_name))
                    print("Saved in " + save_path)
            return np.mean(accuracies)

    def train(self, filename, checkpoints_dir):
        """Train model.

        Args:
            filename: A string which represents a name of file with training
                data.
            checkpoints_dir: A string which represents a path to directory,
                where to save checkpoints.
        Returns:
            session: Tensorflow session object, which can be used for sampling.
        """
        self.fit_data(filename)
        self.construct_graph()
        mean_val_accuracy = self.run_session(checkpoints_dir)
        return mean_val_accuracy, self.session  # for further sampling

    def sample(self, session, filename, out_file='output.txt'):
        """Sample syllables for each word in file.

        Note: graph should be already constructed.
        Args:
            session: Tensorflow session object.
            filename: A string, which represents a name of file with testing
                words, one word per line.
            out_file: A string, which represents a name of output file, where to
            save syllables. Defaults to 'output.txt'
        """
        self.prepare_test_data(filename)
        with open(out_file, 'w') as fout:
            with self.graph.as_default():
                print('Sampling...')
                for (words_batch, syllable_labels_batch,
                     lengths_batch) in self.get_batches('test'):
                    feed_dict = {self.words: words_batch,
                                 self.syllable_labels: syllable_labels_batch,
                                 self.seq_lengths: lengths_batch}
                    pred, indices = session.run([self.prediction,
                                                 self.separation_indices],
                                                feed_dict=feed_dict)
                    pred[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
                    prediction = self.decode_prediction(words_batch, pred,
                                                        lengths_batch)
                    for word, syllables in prediction:
                        print(word, ' '.join(syllables))
                        fout.write(word + ' ' + ' '.join(syllables) + '\n')
