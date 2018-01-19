import tensorflow as tf
import numpy as np
import Util as Data

util = Data.Util()

glove_dimensionality = util.glove_dimensionality
d = 400
_, _, largest_num_of_words_any_paragraph = util.count_words_paragraphs_in_squad()

class Memory:
    A = tf.Variable(tf.random_normal([largest_num_of_words_any_paragraph, glove_dimensionality, d], stddev=0.1), name="A")
    C = tf.Variable(tf.random_normal([largest_num_of_words_any_paragraph, glove_dimensionality, d], stddev=0.1), name="C")
    m = tf.Variable(tf.zeros([1, d]), trainable=False, name="m")
    c = tf.Variable(tf.zeros([1, d]), trainable=False)

    def __init__(self, text):
        self.m = tf.squeeze(tf.matmul(text, self.A), 1)
        self.c = tf.squeeze(tf.matmul(text, self.C), 1)
