import numpy as np
import json
import re
import math


class Util:
    def __init__(self):
        self.vocab_size = 2000
        self.largest_num_of_sentences = 0
        self.largest_num_of_words = 0
        self.special_chars = ["'", "/", ")", "(", "/", "'", "[", "{", "]", "}", "#", "$", "%", "^", "&", "*", "-", "_", "+", "=",
                         ".", "\"", ",", ":", ";"]
        self.num_of_questions = 2000
        self.num_of_paragraphs = int(math.floor(self.num_of_questions / 4))
        self.glove_dimensionality = 100
        self.glove_lookup = self.initialise_glove_embeddings()
        self.glove_lookup_dict = {}

        for entry in self.glove_lookup:
            index = entry[0]
            vector = entry[1]
            self.glove_lookup_dict[index] = vector

        self.glove_lookup_dict_reversed = {}
        self.questions_list, self.paragraphs_list, self.answers_list, self.paragraph_question_mapping = self.read_squad()
        self.largest_num_of_sentences, self.largest_num_of_words, self.largest_num_of_words_any_paragraph = self.count_words_paragraphs_in_squad()

    def initialise_glove_embeddings(self):
        glove_path = 'glove.6B.100d.txt'
        glove_lookup = np.zeros(self.vocab_size, dtype='(100)string, (1,' + str(self.glove_dimensionality) + ')float')
        embedding_text = np.genfromtxt(glove_path, delimiter='\n', dtype='string')
        j = 0
        for word_embedding_line in embedding_text:
            embeddings_for_word = word_embedding_line.split(' ')
            i = 0
            embeddings_array = np.zeros(self.glove_dimensionality, dtype='float64')
            for single_embedding_dimension in embeddings_for_word:
                if i > 0:
                    embeddings_array[i - 1] = float(single_embedding_dimension)
                i = i + 1
            glove_lookup_entry = (embeddings_for_word[0], embeddings_array.squeeze())
            glove_lookup[j] = glove_lookup_entry
            j = j + 1
            if j == self.vocab_size:
                break
        unk_array = np.full(self.glove_dimensionality, 0.1)
        glove_lookup_unk = ("<unk>", unk_array.squeeze())
        glove_lookup[j-2] = glove_lookup_unk
        return glove_lookup

    def get_glove_embedding(self, word):
        if word in self.glove_lookup_dict:
            return self.glove_lookup_dict[word]
        else:
            return self.glove_lookup_dict["<unk>"]

    def get_one_hot_encoded_from_glove(self, word):
        one_hot_encoded = [0 for p in range(len(self.glove_lookup))]
        dimension = 0
        if word not in self.glove_lookup_dict:
            word = "<unk>"
        for word_embedding in self.glove_lookup:
            if word_embedding[0] == word:
                one_hot_encoded[dimension] = 1
                break
            dimension = dimension + 1
        return one_hot_encoded


    def get_word_from_one_hot_encoded(self, index):
        if index == len(self.glove_lookup) - 1:
            return "<stop>"
        else:
            return self.glove_lookup[index][0]


    def parse_squad(self):
        with open('train-v1.1.json', 'r') as squad_file:
            squad_string = squad_file.read()
            parsed_squad = json.loads(squad_string)
            return parsed_squad["data"]


    def count_squad(self):
        data = self.parse_squad()
        number_of_questions = 0
        number_of_answers = 0
        number_of_paragraphs = 0
        for text in data:
            paragraphs = text["paragraphs"]
            for paragraph in paragraphs:
                number_of_paragraphs = number_of_paragraphs + 1
                context = paragraph["context"]
                qas = paragraph["qas"]
                for qa in qas:
                    number_of_questions = number_of_questions + 1
                    question = qa["question"]
                    answers = qa["answers"]
                    for answer in answers:
                        number_of_answers = number_of_answers + 1
                        answer_text = answer["text"]
        return number_of_answers, number_of_questions, number_of_paragraphs


    def read_squad(self):
        data = self.parse_squad()
        number_of_answers, number_of_questions, number_of_paragraphs = self.count_squad()
        questions_list = ['x' for i in range(number_of_questions)]
        answers_list = ['x' for i in range(number_of_answers)]
        paragraphs_list = ['x' for i in range(number_of_paragraphs)]
        paragraph_question_mapping = [0 for i in range(number_of_questions)]
        paragraph_num = 0
        answer_num = 0
        question_num = 0
        for text in data:
            paragraphs = text["paragraphs"]
            for paragraph in paragraphs:
                context = paragraph["context"]
                paragraphs_list[paragraph_num] = context
                qas = paragraph["qas"]
                for qa in qas:
                    question = qa["question"]
                    questions_list[question_num] = question
                    answers = qa["answers"]
                    for answer in answers:
                        answer_text = answer["text"]
                        answers_list[answer_num] = answer_text
                        paragraph_question_mapping[answer_num] = paragraph_num
                        answer_num = answer_num + 1
                    question_num = question_num + 1
                paragraph_num = paragraph_num + 1
        return questions_list, paragraphs_list, answers_list, paragraph_question_mapping

    def count_words_paragraphs_in_squad(self):
        largest_num_of_sentences = 0
        largest_num_of_words = 0
        largest_num_of_words_any_paragraph = 0
        paragraphs = self.paragraphs_list[:self.num_of_paragraphs]
        for paragraph in paragraphs:
            sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)
            num_of_words = 0
            for sentence in sentences:
                words = sentence.split(' ')
                num_of_special_chars = 0
                for word in words:
                    characters = list(word)
                    if len(characters) > 0:
                        if characters[0] in self.special_chars:
                            num_of_special_chars = num_of_special_chars + 1
                        if characters[len(characters) - 1] in self.special_chars:
                            num_of_special_chars = num_of_special_chars + 1
                        if "'" in word and characters[0] not in "'" and characters[len(characters) - 1] not in "'":
                            num_of_special_chars = num_of_special_chars + 1
                num_of_words = num_of_words + num_of_special_chars + len(words)
                if len(words) + num_of_special_chars > largest_num_of_words:
                    largest_num_of_words = len(words) + num_of_special_chars
            if num_of_words > largest_num_of_words_any_paragraph:
                largest_num_of_words_any_paragraph = num_of_words
            if len(sentences) > largest_num_of_sentences:
                largest_num_of_sentences = len(sentences)
        return largest_num_of_sentences, largest_num_of_words, largest_num_of_words_any_paragraph

    def vectorise_paragraphs(self):
        paragraphs = self.paragraphs_list[:self.num_of_paragraphs]
        paragraphs_sentences = np.zeros((len(paragraphs), self.largest_num_of_words_any_paragraph, self.glove_dimensionality))
        i = 0
        for paragraph in paragraphs:
            sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)
            v = 0
            for sentence in sentences:
                words = sentence.split(' ')
                for word in words:
                    characters = list(word)
                    if len(characters) > 0:
                        if characters[0] in self.special_chars:
                            glove_embedding = self.get_glove_embedding(characters[0])
                            paragraphs_sentences[i][v] = glove_embedding
                            v = v + 1
                            word = word[1:]
                        if characters[len(characters) - 1] in self.special_chars:
                            word = word[:-1]
                        word = word.lower()
                        if "'" in word and characters[0] not in "'" and characters[len(characters) - 1] not in "'":
                            apostrophe_word = word.split("'")
                            glove_embedding = self.get_glove_embedding(apostrophe_word[0])
                            paragraphs_sentences[i][v] = glove_embedding
                            v = v + 1
                            glove_embedding = self.get_glove_embedding("'" + apostrophe_word[1])
                            paragraphs_sentences[i][v] = glove_embedding
                            v = v + 1
                        else:
                            glove_embedding = self.get_glove_embedding(word)
                            paragraphs_sentences[i][v] = glove_embedding
                            v = v + 1
                        if characters[len(characters) - 1] in self.special_chars:
                            glove_embedding = self.get_glove_embedding(characters[len(characters) - 1])
                            paragraphs_sentences[i][v] = glove_embedding
                v = v + 1
            i = i + 1
        return paragraphs_sentences

    def vectorise_questions(self, start, stop):
        questions = self.questions_list[start:stop]
        questions_words = np.zeros((len(questions), self.get_largest_num_of_words_in_question(), self.glove_dimensionality))
        j = 0
        for question in questions:
            words = question.split(' ')
            v = 0
            for word in words:
                characters = list(word)
                if len(characters) > 0:
                    if characters[0] in self.special_chars:
                        glove_embedding = self.get_glove_embedding(characters[0])
                        questions_words[j][v] = glove_embedding
                        v = v + 1
                        word = word[1:]
                    if characters[len(characters) - 1] in self.special_chars:
                        word = word[:-1]
                    word = word.lower()
                    if "'" in word and characters[0] not in "'" and characters[len(characters) - 1] not in "'":
                        apostrophe_word = word.split("'")
                        glove_embedding = self.get_glove_embedding(apostrophe_word[0])
                        questions_words[j][v] = glove_embedding
                        v = v + 1
                        glove_embedding = self.get_glove_embedding("'" + apostrophe_word[1])
                        questions_words[j][v] = glove_embedding
                        v = v + 1
                    else:
                        glove_embedding = self.get_glove_embedding(word)
                        questions_words[j][v] = glove_embedding
                        v = v + 1
                    if characters[len(characters) - 1] in self.special_chars:
                        glove_embedding = self.get_glove_embedding(characters[len(characters) - 1])
                        questions_words[j][v] = glove_embedding
                        v = v + 1
            j = j + 1
        return questions_words


    def get_largest_num_of_words_in_answer(self):
        answers = self.answers_list[:self.num_of_questions]
        self.largest_num_of_words = 0
        for answer in answers:
            words = answer.split(' ')
            v = 0;
            num_of_special_chars = 0
            for word in words:
                characters = list(word)
                if len(characters) > 0:
                    if characters[0] in self.special_chars:
                        num_of_special_chars = num_of_special_chars + 1
                    if characters[len(characters) - 1] in self.special_chars:
                        num_of_special_chars = num_of_special_chars + 1
                    if "'" in word and characters[0] not in "'" and characters[len(characters) - 1] not in "'":
                        num_of_special_chars = num_of_special_chars + 1
            if len(words) + num_of_special_chars > self.largest_num_of_words:
                self.largest_num_of_words = len(words) + num_of_special_chars
        return self.largest_num_of_words

    def get_largest_num_of_words_in_question(self):
        questions = self.questions_list[:self.num_of_questions]
        self.largest_num_of_words = 0
        for question in questions:
            words = question.split(' ')
            v = 0;
            num_of_special_chars = 0
            for word in words:
                characters = list(word)
                if len(characters) > 0:
                    if characters[0] in self.special_chars:
                        num_of_special_chars = num_of_special_chars + 1
                    if characters[len(characters) - 1] in self.special_chars:
                        num_of_special_chars = num_of_special_chars + 1
                    if "'" in word and characters[0] not in "'" and characters[len(characters) - 1] not in "'":
                        num_of_special_chars = num_of_special_chars + 1
            if len(words) + num_of_special_chars > self.largest_num_of_words:
                self.largest_num_of_words = len(words) + num_of_special_chars
        return self.largest_num_of_words

    def vectorise_answers(self, start, stop):
        largest_num_of_words_in_answer = self.get_largest_num_of_words_in_answer()
        answers = self.answers_list[start:stop]
        answers_words = np.zeros((len(answers), largest_num_of_words_in_answer, self.vocab_size))
        j = 0
        answer_num = 0
        for answer in answers_words:
            entry_num = 0
            for entry in answer:
                answers_words[answer_num][entry_num][self.vocab_size-1] = 1
                entry_num = entry_num + 1
            answer_num = answer_num + 1
        for answer in answers:
            words = answer.split(' ')
            v = 0
            try:
                for x in range(largest_num_of_words_in_answer):
                    for word in words:
                        characters = list(word)
                        if len(characters) > 0:
                            if characters[0] in self.special_chars:
                                glove_embedding = self.get_one_hot_encoded_from_glove(characters[0])
                                answers_words[j][v] = glove_embedding
                                v = v + 1
                                word = word[1:]
                            if characters[len(characters) - 1] in self.special_chars:
                                word = word[:-1]
                            word = word.lower()
                            if "'" in word and characters[0] not in "'" and characters[len(characters) - 1] not in "'":
                                apostrophe_word = word.split("'")
                                glove_embedding = self.get_one_hot_encoded_from_glove(apostrophe_word[0])
                                answers_words[j][v] = glove_embedding
                                v = v + 1
                                glove_embedding = self.get_one_hot_encoded_from_glove("'" + apostrophe_word[1])
                                answers_words[j][v] = glove_embedding
                                v = v + 1
                            else:
                                glove_embedding = self.get_one_hot_encoded_from_glove(word)
                                answers_words[j][v] = glove_embedding
                                v = v + 1
                            if characters[len(characters) - 1] in self.special_chars:
                                glove_embedding = self.get_one_hot_encoded_from_glove(characters[len(characters) - 1])
                                answers_words[j][v] = glove_embedding
                                v = v + 1
                    if x==0:
                        stop_token = np.zeros(self.vocab_size, dtype='float32')
                        stop_token[self.vocab_size - 1] = 1
                        answers_words[j][v] = stop_token
                        v = v + 1
            except:
                pass
            j = j + 1
        return answers_words

    def vectorise_squad(self, start, stop):
        return self.vectorise_answers(start, stop), self.vectorise_questions(start, stop), self.paragraph_question_mapping[start:stop]


    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    def return_validatation_set(self):
        data = self.parse_squad()
        number_of_answers, number_of_questions, number_of_paragraphs = self.count_squad()
        questions_list = ['x' for i in range(number_of_questions)]
        answers_list = ['x' for i in range(number_of_answers)]
        paragraphs_list = ['x' for i in range(number_of_paragraphs)]
        paragraph_question_mapping = [0 for i in range(number_of_questions)]
        paragraph_num = 0
        answer_num = 0
        question_num = 0
        for text in data:
            paragraphs = text["paragraphs"]
            for paragraph in paragraphs:
                context = paragraph["context"]
                paragraphs_list[paragraph_num] = context
                qas = paragraph["qas"]
                for qa in qas:
                    question = qa["question"]
                    questions_list[question_num] = question
                    answers = qa["answers"]
                    for answer in answers:
                        answer_text = answer["text"]
                        answers_list[answer_num] = answer_text
                        paragraph_question_mapping[answer_num] = paragraph_num
                        answer_num = answer_num + 1
                    question_num = question_num + 1
                paragraph_num = paragraph_num + 1

        i = 77000
        new_paragraph_question_mapping = [0 for d in range(3001)]
        j = 0
        a = 0
        for mapping in paragraph_question_mapping[77000:80000]:
            # print (i, a, paragraph_question_mapping[i], new_paragraph_question_mapping[a], j)
            i = i + 1
            a = a + 1
            if paragraph_question_mapping[i - 1] == paragraph_question_mapping[i]:
                new_paragraph_question_mapping[a] = new_paragraph_question_mapping[a - 1]
            else:
                j = j + 1
                new_paragraph_question_mapping[a] = j
        return questions_list[77000:80000], paragraphs_list[16686:17275], answers_list[
                                                                          77000:80000], new_paragraph_question_mapping


    # for paragraph_new, paragraph in zip(paragraphs_listasascsa, paragraphs_list):

    # _, _, _, c, g = return_validatation_set()

    # for u, a in zip(c, g):
    #	print (u, a)

    def vectorise_validation_paragraphs(self):
        self.largest_num_of_sentences, self.largest_num_of_words, words = self.count_words_paragraphs_in_squad()
        questions, paragraphs, answers, paragraph_question_mapping = self.return_validatation_set()
        paragraphs_sentences = np.zeros((len(paragraphs), self.largest_num_of_sentences, self.largest_num_of_words, self.glove_dimensionality))
        i = 0
        for paragraph in paragraphs:
            if i >= len(paragraphs) - 1:
                break
            sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)
            j = 0
            for sentence in sentences:
                if j >= self.largest_num_of_sentences - 1:
                    break
                words = sentence.split(' ')
                v = 0;
                for word in words:
                    if v >= self.largest_num_of_words - 1:
                        break
                    characters = list(word)
                    if len(characters) > 0:
                        if characters[0] in self.special_chars:
                            glove_embedding = self.get_glove_embedding(characters[0])
                            if v >= self.largest_num_of_words - 1:
                                break
                            paragraphs_sentences[i][j][v] = glove_embedding
                            v = v + 1
                            word = word[1:]
                        if characters[len(characters) - 1] in self.special_chars:
                            word = word[:-1]
                        word = word.lower()
                        if "'" in word and characters[0] not in "'" and characters[len(characters) - 1] not in "'":
                            apostrophe_word = word.split("'")
                            glove_embedding = self.get_glove_embedding(apostrophe_word[0])
                            if v >= self.largest_num_of_words - 1:
                                break
                            paragraphs_sentences[i][j][v] = glove_embedding
                            v = v + 1
                            glove_embedding = self.get_glove_embedding("'" + apostrophe_word[1])
                            if v >= self.largest_num_of_words - 1:
                                break
                            paragraphs_sentences[i][j][v] = glove_embedding
                            v = v + 1
                        else:
                            glove_embedding = self.get_glove_embedding(word)
                            if v >= self.largest_num_of_words - 1:
                                break
                            paragraphs_sentences[i][j][v] = glove_embedding
                            v = v + 1
                        if characters[len(characters) - 1] in self.special_chars:
                            glove_embedding = self.get_glove_embedding(characters[len(characters) - 1])
                            if v >= self.largest_num_of_words - 1:
                                break
                            paragraphs_sentences[i][j][v] = glove_embedding
                            v = v + 1
                j = j + 1
            i = i + 1
        return paragraphs_sentences


    def vectorise_validation_questions(self):
        self.largest_num_of_sentences, self.largest_num_of_words, words = self.count_words_paragraphs_in_squad()
        questions, paragraphs, answers, paragraph_question_mapping = self.return_validatation_set()
        questions_words = np.zeros((len(questions), self.largest_num_of_words, self.glove_dimensionality))
        j = 0
        for question in questions:
            if j >= len(questions) - 1:
                break
            words = question.split(' ')
            v = 0;
            for word in words:
                if v >= self.largest_num_of_words - 1:
                    break
                characters = list(word)
                if len(characters) > 0:
                    if characters[0] in self.special_chars:
                        glove_embedding = self.get_glove_embedding(characters[0])
                        questions_words[j][v] = glove_embedding
                        v = v + 1
                        word = word[1:]
                    if characters[len(characters) - 1] in self.special_chars:
                        word = word[:-1]
                    word = word.lower()
                    if "'" in word and characters[0] not in "'" and characters[len(characters) - 1] not in "'":
                        apostrophe_word = word.split("'")
                        glove_embedding = self.get_glove_embedding(apostrophe_word[0])
                        questions_words[j][v] = glove_embedding
                        v = v + 1
                        glove_embedding = self.get_glove_embedding("'" + apostrophe_word[1])
                        questions_words[j][v] = glove_embedding
                        v = v + 1
                    else:
                        glove_embedding = self.get_glove_embedding(word)
                        questions_words[j][v] = glove_embedding
                        v = v + 1
                    if characters[len(characters) - 1] in self.special_chars:
                        glove_embedding = self.get_glove_embedding(characters[len(characters) - 1])
                        questions_words[j][v] = glove_embedding
                        v = v + 1
            j = j + 1
        return questions_words


    def vectorise_validation_answers(self):
        questions, paragraphs, answers, paragraph_question_mapping = self.return_validatation_set()
        _, all_paragraphs, _, all_paragraph_question_mapping = self.read_squad()
        self.largest_num_of_sentences, self.largest_num_of_words, self.largest_num_of_words_any_paragraph = self.count_words_paragraphs_in_squad()
        self.largest_num_of_words_in_answer = self.get_largest_num_of_words_in_answer()
        answers_words = np.zeros((len(answers), self.largest_num_of_words_in_answer, self.largest_num_of_words_any_paragraph + 2))
        answer_num = 0
        for answer in answers_words:
            entry_num = 0
            for entry in answer:
                answers_words[answer_num][entry_num][self.largest_num_of_words_any_paragraph] = 1
                entry_num = entry_num + 1
            answer_num = answer_num + 1
        j = 0
        for answer in answers:
            if j >= len(answers) - 1:
                break
            answer_lookup_dict = self.get_answer_dictionary(j, paragraphs, paragraph_question_mapping,
                                                       self.largest_num_of_words_any_paragraph)
            words = answer.split(' ')
            v = 0;
            for word in words:
                if v >= self.largest_num_of_words_in_answer - 1:
                    break
                characters = list(word)
                if len(characters) > 0:
                    if characters[0] in self.special_chars:
                        try:
                            glove_embedding = answer_lookup_dict[characters[0]]
                        except Exception, e:
                            glove_embedding = answer_lookup_dict['unk']
                        if v >= self.largest_num_of_words_in_answer - 1:
                            break
                        answers_words[j][v] = glove_embedding
                        v = v + 1
                        word = word[1:]
                    if characters[len(characters) - 1] in self.special_chars:
                        word = word[:-1]
                    word = word.lower()
                    if "'" in word and characters[0] not in "'" and characters[len(characters) - 1] not in "'":
                        apostrophe_word = word.split("'")
                        try:
                            glove_embedding = answer_lookup_dict[apostrophe_word[0]]
                        except Exception, e:
                            glove_embedding = answer_lookup_dict['unk']
                        if v >= self.largest_num_of_words_in_answer - 1:
                            break
                        answers_words[j][v] = glove_embedding
                        v = v + 1
                        try:
                            glove_embedding = answer_lookup_dict["'" + apostrophe_word[1]]
                        except Exception, e:
                            glove_embedding = answer_lookup_dict['unk']
                        if v >= self.largest_num_of_words_in_answer - 1:
                            break
                        answers_words[j][v] = glove_embedding
                        v = v + 1
                    else:
                        try:
                            glove_embedding = answer_lookup_dict[word]
                        except Exception, e:
                            glove_embedding = answer_lookup_dict['unk']
                        if v >= self.largest_num_of_words_in_answer - 1:
                            break
                        answers_words[j][v] = glove_embedding
                        v = v + 1
                    if characters[len(characters) - 1] in self.special_chars:
                        try:
                            glove_embedding = answer_lookup_dict[characters[len(characters) - 1]]
                        except Exception, e:
                            glove_embedding = answer_lookup_dict['unk']
                        if v >= self.largest_num_of_words_in_answer - 1:
                            break
                        answers_words[j][v] = glove_embedding
                        v = v + 1
            j = j + 1
        return answers_words


    def vectorise_validation_squad(self):
        a, b, c, paragraph_question_mapping = self.return_validatation_set()
        return self.vectorise_validation_answers(), self.vectorise_validation_paragraphs(), self.vectorise_validation_questions(), paragraph_question_mapping

    # _,_,_,c = read_squad()
    # print c[77000]
    # print c[80000]
    # gcloud ml-engine jobs submit training k-v --module-name trainer.main --package-path train/trainer --staging-bucket gs://fyp_neural --scale-tier BASIC --region europe-west1
