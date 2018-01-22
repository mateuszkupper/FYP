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

                #check test accuracy
                answers_test, questions_test, paragraphs_test =  self.util.vectorise_squad(30000,30010)
                test_example_num = 0
                accuracy_list = []
                #compute accuracy for single examples
                for answer_test in answers_test:
                    question_test = questions_test[test_example_num]
                    paragraph_test = paragraphs_test[test_example_num]
                    feed_dict = {self.question: question_test, self.text: paragraph_test}
                    classification = sess.run(answer_softmax, feed_dict)
                    class_words = self.util.get_words(classification)
                    actual_words = self.util.get_words(answer_test)
                    common_words = list(set(class_words).intersection(actual_words))
                    if len(actual_words) > 0:
                        example_accuracy = float(len(common_words))/float(len(actual_words))
                        accuracy_list.append(example_accuracy)
                    test_example_num+=1
                #sum and average
                test_accuracy = sum(accuracy_list) / float(len(accuracy_list))
                print("Accuracy: ", test_accuracy)
            file_writer.close()

Train().train()
