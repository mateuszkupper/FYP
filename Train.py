import tensorflow as tf
import numpy as np
import Util as Data
import re
import string
import RMN as RMN
import Memory as MNMemory

class Train:
    def __init__(self):
        self.util = Data.Util()
        self.unk_answer = self.util.get_one_hot_encoded_from_glove("<unk>")
        self.d = 400
        self.num_of_batches = 10
        self.l_rate=0.001
        self.total_examples = 2000
        self.examples_per_batch = self.total_examples/self.num_of_batches
        self.num_of_epochs = 200
        _, _, self.largest_num_of_words_any_paragraph = self.util.count_words_paragraphs_in_squad()
        self.largest_num_of_words_in_answer = self.util.get_largest_num_of_words_in_answer()
        self.largest_num_of_words_in_question = self.util.get_largest_num_of_words_in_question()
        self.clip_norm = 5.0
        self.question = tf.placeholder(tf.float32, shape=(self.largest_num_of_words_in_question, self.util.glove_dimensionality), name="question")
        self.text = tf.placeholder(tf.float32, shape=(self.largest_num_of_words_any_paragraph, self.util.glove_dimensionality), name="text")
        self.answer = tf.placeholder(tf.float32, shape=(self.largest_num_of_words_in_answer, self.util.vocab_size))

    def train(self):
        #define model
        with tf.name_scope('train'):
            with tf.name_scope('model'):
                with tf.name_scope('encoder-decoder'):
                    dynamic_net = RMN.DynamicCell(self.text)
                    hidden_state = dynamic_net.encode(self.question)
                    predicted_answer, answer_softmax = dynamic_net.decode(self.answer, hidden_state)
            loss = tf.reduce_mean(predicted_answer)
            tf.summary.scalar("loss", loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.l_rate)

        #https://stackoverflow.com/questions/36498127/how-to-effectively-apply-gradient-clipping-in-tensor-flow
        #clip gradients
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        train = optimizer.apply_gradients(zip(gradients, variables))

        #log loss
        tf.summary.scalar("loss", loss)
        init = tf.global_variables_initializer()
        #merge all summaries
        merged_summary_op = tf.summary.merge_all()

        with tf.Session() as sess:
            init.run()
            #https://stackoverflow.com/questions/41085549/there-is-no-graph-with-tensorboard
            #file_writer = tf.summary.FileWriter('tflogs', sess.graph)
            paragraphs = self.util.vectorise_paragraphs()
            for epoch in range(self.num_of_epochs):
                for batch in range(self.num_of_batches):
                    answer_num = 0
                    answers, questions, paragraph_question_mapping = self.util.vectorise_squad\
                        (batch*self.examples_per_batch, (batch+1)*self.examples_per_batch)
                    for question_example in questions:
                        question_example = np.flip(question_example, 0)
                        print("Answer: ", answer_num, "Iter: ", epoch)
                        answer_example = answers[answer_num]

                        #get rid of answers with <unk>s
                        a_list = [list(item) for item in answer_example]
                        if self.unk_answer in a_list:
                            answer_num += 1
                            #skip example when <unk> found
                            continue

                        paragraph_example = paragraphs[paragraph_question_mapping[answer_num]]
                        sess.run(train, feed_dict={self.question: question_example, self.answer: answer_example, self.text: paragraph_example})
                        summary_str = sess.run(merged_summary_op, feed_dict={self.question: question_example, self.answer: answer_example, self.text: paragraph_example})
                        #file_writer.add_summary(summary_str, answer_num*k + answer_num)
                        acc_train = loss.eval(feed_dict={self.question: question_example, self.answer: answer_example, self.text: paragraph_example})
                        print(acc_train)
                        #if answer_num % 100 == 0:
                        #print(i, answer_num, " Train accuracy: ", acc_train)
                        feed_dict = {self.question: question_example, self.text: paragraph_example}
                        classification = sess.run(answer_softmax, feed_dict)
                        #if answer_num % 1000 == 0:
                         #   builder = tf.saved_model.builder.SavedModelBuilder("model" + str(i))
                          #  builder.add_meta_graph_and_variables(sess, ["tag"], signature_def_map={
                           #     "model": tf.saved_model.signature_def_utils.predict_signature_def(
                            #        inputs={"question": question, "text": text},
                             #       outputs={"answer": answer_softmax})
                           # })
                           # builder.save()
                        vectors = [0 for w in range(self.largest_num_of_words_in_answer)]
                        i = 0
                        for word in classification:
                            j = 0
                            emb_max = 0
                            for emb in word:
                                if emb > emb_max:
                                    emb_max = emb
                                    vector = j
                                j = j + 1
                            vectors[i] = vector
                            i = i + 1
                        print " "
                        print vectors
                        print "Answer: "
                        answerq = ""
                        for vector in vectors:
                            if vector ==self.util.vocab_size-1:
                                break
                            else:
                                answerq = answerq + " " + self.util.get_word_from_one_hot_encoded(vector)
                        answerq = answerq + "."
                        print answerq[1:].capitalize()

                        vectors = [0 for w in range(self.largest_num_of_words_in_answer)]
                        i = 0
                        for word in answer_example:
                            j = 0
                            for emb in word:
                                if emb == 1:
                                    vector = j
                                    break
                                j = j + 1
                            vectors[i] = vector
                            i = i + 1
                        print " "
                        print vectors
                        print "Actual answer: "
                        answerq = ""
                        for vector in vectors:
                            if vector == self.util.vocab_size-1:
                                break
                            else:
                                answerq = answerq + " " + self.util.get_word_from_one_hot_encoded(vector)
                        answerq = answerq + "."
                        print answerq[1:].capitalize()
                        answer_num += 1
            #file_writer.close()
Train().train()