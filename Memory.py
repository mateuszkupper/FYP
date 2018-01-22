import tensorflow as tf
import numpy as np
import main
Config = main.Config()
glove_dimensionality = Config.glove_dimensionality
d = Config.d

class Memory:
    def __init__(self, text, util):
        self.A = tf.Variable(tf.random_normal([util.largest_num_of_words_any_paragraph, util.glove_dimensionality, d], stddev=0.1),
                        name="A")
        self.C = tf.Variable(tf.random_normal([util.largest_num_of_words_any_paragraph, util.glove_dimensionality, d], stddev=0.1),
                        name="C")

        self.m = tf.squeeze(tf.matmul(text, self.A), 1)
        self.c = tf.squeeze(tf.matmul(text, self.C), 1)
