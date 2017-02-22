"""Module for breaking words into syllables."""
from datetime import datetime
import os
import re
from time import time
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.contrib.rnn.python.ops import core_rnn_cell as rnn_cell
from tensorflow.python.ops.nn import bidirectional_dynamic_rnn as dynamic_brnn
from tensorflow.python.ops.nn import dynamic_rnn as dynamic_rnn
from utils import top_k_indices


class SyllableParser(object):
    """Tensorflow model for parsing syllables."""

    def __init__(self, num_epochs=100, hidden_size=128,
                 cell_type='lstm', net_type='brnn', num_layers=3, treshold=0.5):
        """Init SyllableParser.

        Args:
            num_epochs: A non-negative integer, which represents number of
                training epochs. Defaults to 100.
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
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.net_type = net_type
        self.treshold = treshold

    def _encode(self, word):
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

    def _decode(self, encoded_word):
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

    def _encode_syllables(self, syllables):
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

    def _pad_into_matrix(self, rows, padding_value=0):
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

    def _fit_data(self, filename):
        r"""Parse and encode training data.

        Args:
            filename: A string, which represents name of file with training
                data. Each line of the file should be in format:
                'word\tsyl la bles'.
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
                encoded_word = self._encode(tokens[0])
                encoded_syllables = self._encode_syllables(re.split(r'\s+',
                                                                    tokens[1]))
                X.append(encoded_word)
                y.append(encoded_syllables)
            self.X, self.lengths = X, list(map(len, X))
            self.y = y

    def _prepare_test_data(self, filename):
        """Parse and encode test data.

        Args:
            filename: A string, which represents a name of file with test data.
                Each line of the file should contain one word.
        """
        X = []
        y = []
        with open(filename) as fin:
            for line in fin:
                y.append(self._count_syllables(line))
                X.append(self._encode(line))
        self.X_test, self.test_lengths = X, list(map(len, X))
        self.y_test = y

    def _decode_prediction(self, words, predicted_labels, lengths):
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
            for ch, idx in zip(list(self._decode(word[:length])), pred[:length]):
                if idx == 0:
                    syllable.append(ch)
                if idx == 1:
                    syllable.append(ch)
                    syllables.append(''.join(syllable))
                    syllable = []
            if len(syllable) > 0:
                syllables.append(''.join(syllable))
            items.append((self._decode(word[:length]), syllables))
        return items

    def _count_syllables(self, line):
        y = []
        for symbol in line.strip():
            if symbol in 'уеыаоэёяию':
                y.append(1)
            else:
                y.append(0)
        return y

    def _get_batches(self, mode):
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
            X, y = self.X_test, self.y_test
            lengths = self.test_lengths
        else:
            raise ValueError('Unknown mode.')
        X_batch, y_batch, lengths_batch = [], [], []
        for idx, x_sample in enumerate(X):
            if idx > 0 and idx % self.batch_size == 0:
                X_batch, lengths_batch = self._pad_into_matrix(X_batch, 0)
                y_batch, _ = self._pad_into_matrix(y_batch, 0)
                yield X_batch, y_batch, lengths_batch
                X_batch, y_batch, lengths_batch = [], [], []
            X_batch.append(x_sample)
            y_batch.append(y[idx])
            lengths_batch.append(lengths[idx])

    def _accuracy(self, true_syllables, pred_syllables, seq_lengths):
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

    def _construct_graph(self):
        """Construct Tensorflow graph."""
        self.graph = tf.Graph()
        hidden_state_size = self.hidden_size
        if self.net_type == 'brnn':
            hidden_state_size *= 2
        with self.graph.as_default():
            self.words = tf.placeholder(tf.int32, shape=(None, None),
                                        name='words')
            self.syllable_labels = tf.placeholder(tf.int32,
                                                  shape=(None, None),
                                                  name='syllable_labels')
            self.seq_lengths = tf.placeholder(tf.int32, shape=(None),
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
            self.num_syllables = tf.reduce_sum(tf.cast(self.syllable_labels, tf.float32) *
                                               tf.sequence_mask(self.seq_lengths,
                                                                tf.reduce_max(self.seq_lengths),
                                                                dtype=tf.float32), 1)
            if self.cell_type == 'lstm':
                cell = rnn_cell.LSTMCell(self.hidden_size)
            elif self.cell_type == 'gru':
                cell = rnn_cell.GRUCell(self.hidden_size)
            elif self.cell_type == 'block_lstm':
                cell = tf.contrib.rnn.LSTMBlockCell(self.hidden_size)
            else:
                raise ValueError('Unknown cell type.')
            rnn_multicell = rnn_cell.MultiRNNCell([cell] * self.num_layers)
            if self.net_type == 'rnn':
                self.outputs, _ = dynamic_rnn(rnn_multicell, embedding,
                                              sequence_length=self.seq_lengths,
                                              dtype=tf.float32,
                                              swap_memory=True)
            elif self.net_type == 'brnn':
                self.outputs, _ = dynamic_brnn(rnn_multicell, rnn_multicell,
                                               embedding,
                                               sequence_length=self.seq_lengths,
                                               dtype=tf.float32,
                                               swap_memory=True)
                self.outputs = tf.concat(self.outputs, 2)
            outputs_reshape = tf.reshape(self.outputs, [-1, hidden_state_size])
            logits = tf.matmul(outputs_reshape, W) + b
            self.logits = tf.reshape(logits, [self.batch_size, -1, 2])
            probs = tf.nn.softmax(self.logits)
            # probabilities only for positive class:
            self.sliced_probs = tf.slice(probs, [0, 0, 1], [-1, -1, -1])
            self.sliced_probs = tf.squeeze(self.sliced_probs, axis=2)
            greater = tf.greater(self.sliced_probs, treshold)
            self.separation_indices = tf.where(greater)
            self.prediction = tf.zeros_like(greater, dtype=tf.float32)
            unmasked_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        logits=self.logits, labels=self.syllable_labels)
            mask = tf.sequence_mask(self.seq_lengths,
                                    tf.reduce_max(self.seq_lengths),
                                    dtype=tf.float32)
            self.loss = tf.reduce_sum(unmasked_ce * mask) / tf.reduce_sum(mask)
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
            self.saver = tf.train.Saver()

    def _run_session(self, checkpoints_dir):
        """Run Tensorflow model training.

        Args:
            checkpoints_dir: A string, which represents path to directory,
                where to save model checkpoints.
        """
        with self.graph.as_default():
            # config = tf.ConfigProto()
            # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            self.session = tf.Session()  # config=config)
            self.session.run(tf.global_variables_initializer())
            print("Checking for checkpoints...")
            latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
            if latest_checkpoint is not None:
                print("Found checkpoints, using them.")
                self.saver.restore(self.session, latest_checkpoint)
            else:
                print("No checkpoints found, starting training from scratch.")
                self._dump_parameters(checkpoints_dir)
            accuracies = []
            for epoch in range(self.num_epochs):
                print("Starting epoch {}".format(epoch))
                batch_losses = []
                start = time()
                for (words_batch,
                     syllable_labels_batch,
                     lengths_batch) in self._get_batches('train'):
                    feed_dict = {self.words: words_batch,
                                 self.syllable_labels: syllable_labels_batch,
                                 self.seq_lengths: lengths_batch}
                    if words_batch.shape != syllable_labels_batch.shape:
                        prediction = self._decode_prediction(
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
                epoch_result = 'Epoch {} done. Train loss: {}. Training took {} sec.'
                print(epoch_result.format(epoch, np.mean(batch_losses),
                                          end - start))
                val_losses = []
                for (val_words_batch,
                     val_syllable_label_batch,
                     val_lengths_batch) in self._get_batches('val'):
                    feed_dict = {self.words: val_words_batch,
                                 self.syllable_labels: val_syllable_label_batch,
                                 self.seq_lengths: val_lengths_batch}
                    (pred, indices,
                     nums_of_syllables,
                     probs, val_loss) = self.session.run([self.prediction,
                                                          self.separation_indices,
                                                          self.num_syllables,
                                                          self.sliced_probs,
                                                          self.loss],
                                                         feed_dict=feed_dict)
                    # pred[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
                    for idx, (k, word_probs, length) in enumerate(zip(nums_of_syllables, probs,
                                                                      val_lengths_batch)):
                        indices = top_k_indices(word_probs[:length], k=k)
                        pred[idx][indices] = 1
                        # if idx % 100 == 0:
                        #     print(self.decode(val_words_batch[idx][:length]))
                        #    print(f'\n\tk: {k}\n\tlength: {length}\n\tprobs: {word_probs}\n\tpred:'
                        #           f' {pred[idx]}\n'
                        #           f'\tindices:{indices}\n')
                    val_losses.append(val_loss)
                    # pred = pred.reshape((pred.shape[0],
                    #                     pred.shape[1])).astype(np.int32)
                print('Validation loss: %f' % np.mean(val_losses))
                accuracy = self.accuracy(val_syllable_label_batch,
                                         pred, val_lengths_batch)
                accuracies.append(accuracy)
                print('Accuracy: %f' % accuracy)
                indices = np.random.choice(np.arange(len(val_words_batch)), 3)
                prediction = self._decode_prediction(val_words_batch[indices],
                                                     pred[indices],
                                                     val_lengths_batch[indices])
                true_values = self._decode_prediction(val_words_batch[indices],
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

    def _dump_parameters(self, checkpoints_dir):
        with open(os.path.join(checkpoints_dir, 'params'), 'w') as params_file:
            pickle.dump(self.__dict__, params_file)

    def _load_parameters(self, checkpoints_dir):
        with open(os.path.join(checkpoints_dir, 'params')) as params_file:
            self.__dict__ = pickle.load(params_file)

    def restore(self, checkpoints_dir):
        if os.path.exists(checkpoints_dir) and len(list(os.listdir(checkpoints_dir))) > 1:
            self._load_parameters(checkpoints_dir)
        self._construct_graph()
        with self.graph.as_default():
            session = tf.Session()
            session.run(tf.global_variables_initializer())
            latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
            if latest_checkpoint is not None:
                self.saver.restore(session, latest_checkpoint)
                return session  # for sampling
            else:
                print("No checkpoints found")

    def train(self, filename, checkpoints_dir, batch_size=20):
        """Train model.

        Args:
            filename: A string which represents a name of file with training
                data.
            checkpoints_dir: A string which represents a path to directory,
                where to save checkpoints.
            batch_size: A non-negative integer, which represents size of
                training batch. Defaults to 20.
        Returns:
            session: Tensorflow session object, which can be used for sampling.
        """
        if os.path.exists(checkpoints_dir) and len(list(os.listdir(checkpoints_dir))) > 1:
            # that is there are checkpoints worth restoring
            self._load_parameters(checkpoints_dir)

        self.batch_size = batch_size
        self._fit_data(filename)
        self._construct_graph()
        os.makedirs(checkpoints_dir, exist_ok=True)
        mean_val_accuracy = self._run_session(checkpoints_dir)
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
        self._prepare_test_data(filename)
        with open(out_file, 'w') as fout:
            with self.graph.as_default():
                print('Sampling...')
                for (words_batch, syllable_labels_batch,
                     lengths_batch) in self.get_batches('test'):
                    feed_dict = {self.words: words_batch,
                                 self.syllable_labels: syllable_labels_batch,
                                 self.seq_lengths: lengths_batch}
                    (pred, indices, probs,
                     nums_of_syllables) = session.run([self.prediction,
                                                       self.separation_indices,
                                                       self.sliced_probs,
                                                       self.num_syllables],
                                                      feed_dict=feed_dict)
                    for idx, (k, word_probs, length) in enumerate(zip(nums_of_syllables, probs,
                                                                      lengths_batch)):
                        indices = top_k_indices(word_probs[:length], k=k)
                        pred[idx][indices] = 1
                    prediction = self._decode_prediction(words_batch, pred,
                                                         lengths_batch)
                    for word, syllables in prediction:
                        # print(word, ' '.join(syllables))
                        fout.write(word + ' ' + ' '.join(syllables) + '\n')
