import math

class Config:
    def __init__(self):
        self.vocab_size = 20000
        self.glove_dimensionality = 100
        self.d = 400
        self.num_of_epochs = 10
        self.num_of_batches = 100
        self.l_rate=0.001
        self.total_examples = 1000
        self.examples_per_batch = self.total_examples/self.num_of_batches
        self.num_of_epochs = 10
        self.clip_norm = 5.0
        self.special_chars = ["'", "/", ")", "(", "/", "'", "[", "{", "]", "}", "#", "$", "%",
                              "^", "&", "*", "-", "_", "+", "=", ".", "\"", ",", ":", ";"]
        self.num_of_questions = 1000
        self.num_of_paragraphs = int(math.floor(self.num_of_questions / 4))