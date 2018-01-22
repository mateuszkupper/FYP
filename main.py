import math

class Config:
    def __init__(self):
        self.vocab_size = 8000
        self.glove_dimensionality = 200
        self.d = 500
        self.num_of_epochs = 10
        self.num_of_batches = 1000
        self.l_rate=0.001
        self.total_examples = 20000
        self.examples_per_batch = self.total_examples/self.num_of_batches
        self.clip_norm = 5.0
        self.special_chars = ["'", "/", ")", "(", "/", "'", "[", "{", "]", "}", "#", "$", "%",
                              "^", "&", "*", "-", "_", "+", "=", ".", "\"", ",", ":", ";"]
        self.num_of_paragraphs = self.total_examples