import tensorflow as tf
import Util as Data
import numpy as np
import Memory as MNMemory

util = Data.Util()

glove_dimensionality = 100
d = 400
vocab_size = util.vocab_size
largest_num_of_sentences, largest_num_of_words, largest_num_of_words_any_paragraph = util.count_words_paragraphs_in_squad()
largest_num_of_words_in_answer = util.get_largest_num_of_words_in_answer()


class Cell:
    def __init__(self, text):
        self.memory = MNMemory.Memory(tf.expand_dims(text, 1))
        self.blank_word = tf.constant(np.zeros((glove_dimensionality, 1)), dtype=tf.float32)
        self.B_internal = tf.Variable(tf.random_normal([d, glove_dimensionality], stddev=0.1), name="B_internal")
        self.B_external = tf.Variable(tf.random_normal([d, glove_dimensionality], stddev=0.1), name="B_external")
        self.W = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=0.1), name="W")
        self._a = tf.Variable(tf.zeros([glove_dimensionality, 1]), name="a", trainable=False)
        self._o = tf.Variable(tf.zeros([1, d]), trainable=False)
        self.u = tf.Variable(tf.zeros([d, 1]), trainable=False)
        self.p = tf.Variable(tf.zeros([1, largest_num_of_words_any_paragraph]), trainable=False)

    def connect(self, previous_layer):
        with tf.name_scope('cell'):
            self.u = tf.matmul(self.B_internal, self._a) + tf.matmul(self.B_external,
                                                                     previous_layer.a)
            self.p = tf.nn.softmax(tf.matmul(tf.transpose(self.u), tf.transpose(self.memory.m)))
            self._o = tf.nn.tanh(tf.matmul(self.p, self.memory.c))
            self._a = tf.matmul(self.W, tf.add(tf.transpose(self._o), self.u))

    def connect_decode(self, previous_layer, previous_step):
        with tf.name_scope('cell'):
            self.u = tf.matmul(self.B_internal, previous_step._a) + tf.matmul(self.B_external,
                                                                              previous_layer.a)
            self.p = tf.nn.softmax(tf.matmul(tf.transpose(self.u), tf.transpose(self.memory.m)))
            self._o = tf.nn.tanh(tf.matmul(self.p, self.memory.c))
            self._a = tf.matmul(self.W, tf.add(tf.transpose(self._o), self.u))

    def connect_encode(self, previous_layer, q):
        with tf.name_scope('cell'):
            self.u = tf.cond(tf.reduce_all(tf.equal(q, self.blank_word)), lambda: self.u,
                             lambda: tf.matmul(self.B_internal, self._a) + tf.matmul(self.B_external,
                                                                                     previous_layer.a))
            self.p = tf.cond(tf.reduce_all(tf.equal(q, self.blank_word)), lambda: self.p,
                             lambda: tf.nn.softmax(
                                 tf.matmul(tf.transpose(self.u), tf.transpose(self.memory.m))))
            self._o = tf.cond(tf.reduce_all(tf.equal(q, self.blank_word)), lambda: self._o,
                              lambda: tf.nn.tanh(tf.matmul(self.p, self.memory.c)))
            self._a = tf.cond(tf.reduce_all(tf.equal(q, self.blank_word)), lambda: self._a,
                              lambda: tf.matmul(self.W,
                                                tf.add(tf.transpose(self._o), self.u)))

    @property
    def a(self):
        return self._a

    @property
    def o(self):
        return self._o


class InitCell:
    def __init__(self, text):
        self.memory = MNMemory.Memory(tf.expand_dims(text, 1))
        self.blank_word = tf.constant(np.zeros((glove_dimensionality, 1)), dtype=tf.float32)
        self.B_internal = tf.Variable(tf.random_normal([d, glove_dimensionality], stddev=0.1), name="B_internal")
        self.W = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=0.1), name="W")
        self._a = tf.Variable(tf.zeros([glove_dimensionality, 1]), name="a", trainable=False)
        self._o = tf.Variable(tf.zeros([1, d]), trainable=False)
        self.u = tf.Variable(tf.zeros([d, 1]), trainable=False)
        self.p = tf.Variable(tf.zeros([1, largest_num_of_words_any_paragraph]), trainable=False)

    def connect(self, q):
        with tf.name_scope('cell'):
            self.q = q
            self.u = tf.cond(tf.reduce_all(tf.equal(q, self.blank_word)), lambda: self.u,
                             lambda: tf.matmul(self.B_internal,
                                               tf.transpose(tf.expand_dims(self.q, 0))))
            self.p = tf.cond(tf.reduce_all(tf.equal(q, self.blank_word)), lambda: self.p,
                             lambda: tf.nn.softmax(
                                 tf.matmul(tf.transpose(self.u), tf.transpose(self.memory.m))))
            self._o = tf.cond(tf.reduce_all(tf.equal(q, self.blank_word)), lambda: self._o,
                              lambda: tf.nn.tanh(tf.matmul(self.p, self.memory.c)))
            self._a = tf.cond(tf.reduce_all(tf.equal(q, self.blank_word)), lambda: self._a,
                              lambda: tf.matmul(self.W,
                                                tf.add(tf.transpose(self._o), self.u)))

    @property
    def a(self):
        return self._a

    @property
    def o(self):
        return self._o


class DynamicCell:
    def __init__(self, text):
        self.X = tf.Variable(tf.random_normal([vocab_size, d], stddev=0.1), name="X")
        self.enc_cell1 = InitCell(text)
        self.enc_cell2 = Cell(text)
        self.enc_cell3 = Cell(text)
        self.enc_cell4 = Cell(text)
        self.enc_cell5 = Cell(text)
        self.dec_cell1 = InitCell(text)
        self.dec_cell2 = Cell(text)
        self.dec_cell3 = Cell(text)
        self.dec_cell4 = Cell(text)
        self.dec_cell5 = Cell(text)

    def encode(self, inp):
        with tf.name_scope('encode'):
            question_words = tf.unstack(inp)
            i=0
            for word in question_words:
                with tf.name_scope('encode_iteration'):
                    self.enc_cell1.connect(word)
                    self.enc_cell2.connect_encode(self.enc_cell1, word)
                    self.enc_cell3.connect_encode(self.enc_cell2, word)
                    self.enc_cell4.connect_encode(self.enc_cell3, word)
                    self.enc_cell5.connect_encode(self.enc_cell4, word)
                    if i==0:
                        state=self.enc_cell5.a
                    else:
                        state+=self.enc_cell5.a
                    i+=1
        return state

    def decode(self, answer, hidden_state):
        with tf.name_scope('decode'):
            self.dec_cell1 = self.enc_cell1
            self.dec_cell2 = self.enc_cell2
            self.dec_cell3 = self.enc_cell3
            self.dec_cell4 = self.enc_cell4
            self.dec_cell5 = self.enc_cell5
            s = np.zeros((1, vocab_size), dtype='float32')
            s[0][vocab_size - 1] = 1
            s = tf.constant(s)
            answer_words = tf.unstack(answer, axis=0)
            i = 0
            for word in answer_words:
                with tf.name_scope('decode_iteration'):
                    if i == 0:
                        self.dec_cell1.connect(tf.squeeze(hidden_state, 1))
                        self.dec_cell2.connect_decode(self.dec_cell1, self.enc_cell2)
                        self.dec_cell3.connect_decode(self.dec_cell2, self.enc_cell3)
                        self.dec_cell4.connect_decode(self.dec_cell3, self.enc_cell4)
                        self.dec_cell5.connect_decode(self.dec_cell4, self.enc_cell5)
                    else:
                        self.dec_cell2.connect(self.dec_cell1)
                        self.dec_cell3.connect(self.dec_cell2)
                        self.dec_cell4.connect(self.dec_cell3)
                        self.dec_cell5.connect(self.dec_cell4)
                    self.dec_cell1.connect(tf.squeeze(self.dec_cell5.a, 1))
                    if i==0:
                        answer_word = tf.transpose(tf.matmul(self.X, tf.transpose(self.enc_cell5.o)))
                    answer_word = tf.cond(tf.reduce_all(tf.equal(s, answer_word)), lambda : s,
                                          lambda : tf.transpose(tf.matmul(self.X, tf.transpose(self.enc_cell5.o))))
                    word_softmax = tf.nn.softmax(logits=answer_word)
                    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=answer_word, labels=word)
                    if i == 0:
                        self.predicted_answer = xentropy
                        self.answer_softmax = word_softmax
                    else:
                        self.predicted_answer = tf.concat([self.predicted_answer, xentropy], 0)
                        self.answer_softmax = tf.concat([self.answer_softmax, word_softmax], 0)
                    i += 1
            return self.predicted_answer, self.answer_softmax