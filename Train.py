import tensorflow as tf
import numpy as np
import Util as Data
import re
import string
import RMN as RMN
import Memory as MNMemory
import main

class Train:
    def __init__(self):
        self.Config = main.Config()
        self.d = self.Config.d
        self.num_of_batches = self.Config.num_of_batches
        self.l_rate=self.Config.l_rate
        self.total_examples = self.Config.total_examples
        self.examples_per_batch = self.total_examples/self.num_of_batches
        self.clip_norm = self.Config.clip_norm
        self.num_of_epochs = self.Config.num_of_epochs

        self.util = Data.Util()
        self.unk_answer = self.util.get_one_hot_encoded_from_glove("<unk>")
        self.largest_num_of_words_any_paragraph = self.util.largest_num_of_words_any_paragraph
        self.largest_num_of_words_in_answer = self.util.get_largest_num_of_words_in_answer()
        self.largest_num_of_words_in_question = self.util.get_largest_num_of_words_in_question()

        self.question = tf.placeholder(tf.float32, shape=(self.largest_num_of_words_in_question, self.util.glove_dimensionality), name="question")
        self.text = tf.placeholder(tf.float32, shape=(self.largest_num_of_words_any_paragraph, self.util.glove_dimensionality), name="text")
        self.answer = tf.placeholder(tf.float32, shape=(self.largest_num_of_words_in_answer, self.util.vocab_size))

    def train(self):
        #define model
        with tf.name_scope('train'):
            with tf.name_scope('model'):
                with tf.name_scope('encoder-decoder'):
                    dynamic_net = RMN.DynamicCell(self.text, self.util)
                    hidden_state = dynamic_net.encode(self.question)
                    predicted_answer, answer_softmax = dynamic_net.decode(self.answer, hidden_state)
                    answer_softmax_save = tf.identity(answer_softmax, name="answer")
            loss = tf.reduce_mean(predicted_answer)
            tf.summary.scalar("loss", loss)
        print answer_softmax
        print answer_softmax_save
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
            file_writer = tf.summary.FileWriter('tflogs', sess.graph)
            for epoch in range(self.num_of_epochs):
                example_num = 0
                accuracy = 0
                for batch in range(self.num_of_batches):
                    answer_num = 0
                    answers, questions, paragraphs = self.util.vectorise_squad\
                        (batch*self.examples_per_batch, (batch+1)*self.examples_per_batch)
                    for question_example in questions:
                        question_example = np.flip(question_example, 0)
                        answer_example = answers[answer_num]

                        #get rid of answers with <unk>s - out of vocab
                        a_list = [list(item) for item in answer_example]
                        if self.unk_answer in a_list:
                            answer_num += 1
                            #skip example when <unk> found
                            continue

                        paragraph_example = paragraphs[answer_num]

                        #run training
                        sess.run(train, feed_dict={self.question: question_example, self.answer: answer_example, self.text: paragraph_example})

                        #log loss
                        summary_str = sess.run(merged_summary_op, feed_dict={self.question: question_example, self.answer:
                            answer_example, self.text: paragraph_example})
                        file_writer.add_summary(summary_str, answer_num*batch*epoch + answer_num)

                        #evaluate loss
                        if example_num % 100 == 0:
                            acc_train = loss.eval(
                                feed_dict={self.question: question_example, self.answer: answer_example,
                                           self.text: paragraph_example})
                            print(epoch, example_num, "Loss: ", acc_train)

                        #measure accuracy
                        feed_dict = {self.question: question_example, self.text: paragraph_example}
                        classification = sess.run(answer_softmax, feed_dict)
                        words = self.util.get_words(classification)
                        actual_answer = self.util.get_words(answer_example)
                        #https://stackoverflow.com/questions/2864842/common-elements-comparison-between-2-lists
                        common_words = list(set(words).intersection(actual_answer))
                        print actual_answer
                        print words
                        num_of_correct_words = len(common_words)
                        #accuracy = (accuracy*(example_num+1) + (num_of_correct_words/self.largest_num_of_words_in_answer)*100)/(example_num+1)
                        #if example_num % 10 == 0:
                        #    print ("Train accuracy: ", accuracy)
                        if len(actual_answer) > 0:
                            accuracy = (float(num_of_correct_words) / float(len(actual_answer))) * 100.0
                        print ("Train accuracy: ", accuracy)

                        #save model
                        if example_num % 1000 == 0:
                            builder = tf.saved_model.builder.SavedModelBuilder("model" + str(example_num) + str(epoch))
                            builder.add_meta_graph_and_variables(sess, ["tag"], signature_def_map={
                             "model": tf.saved_model.signature_def_utils.predict_signature_def(
                                inputs={"question": self.question, "text": self.text},
                                outputs={"answer": answer_softmax_save})
                         })
                        builder.save()
                        answer_num += 1
                        example_num += 1
            file_writer.close()

Train().train()
