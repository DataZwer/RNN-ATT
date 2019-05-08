import tensorflow as tf
from data_helper.sen_id_label import prepocess
from data_helper.embedding import load_word_embedding
from bean.file_path import *
from data_batches.batches import dataset_iterator
import numpy as np


class gru_model(object):
    def __init__(self, config_lstm, sess):
        self.batch_size = config_lstm['batch_size']
        self.lstm_units = config_lstm['lstm_units']
        self.num_classes = config_lstm['num_classes']
        self.n_epochs = config_lstm['n_epochs']
        self.embedding_size = config_lstm['embedding_size']
        self.l2_reg_lambda = config_lstm['l2_reg_lambda']

        self.l2_loss = tf.constant(0.0)
        self.sess = sess
        self.x_train, self.y_train, self.word_index, self.x_dev, self.y_dev, self.max_len = \
            prepocess(mr_pos_path, mr_neg_path)
        self.x_inputs = tf.placeholder(tf.int32, [None, self.max_len], name='x_inputs')
        self.y_inputs = tf.placeholder(tf.float32, [None, self.num_classes], name='y_inputs')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    def embedding_layer(self):
        # embedding layer
        with tf.name_scope('embedding'):
            glove_embedding = load_word_embedding(
                word_index=self.word_index,
                file='',
                trimmed_filename=glove_embedding_save,
                load=True,
                dim=300
            )
            glove_w2v = tf.Variable(glove_embedding, dtype=tf.float32, name='glove_w2v')
            sen_inputs_glove = tf.nn.embedding_lookup(glove_w2v, self.x_inputs)
        return sen_inputs_glove

    def lstm(self):
        sen_inputs_glove = self.embedding_layer()
        gru_cell = tf.contrib.rnn.GRUCell(self.lstm_units)
        gru_cell = tf.contrib.rnn.DropoutWrapper(cell=gru_cell, output_keep_prob=0.75)

        value, _ = tf.nn.dynamic_rnn(
            gru_cell,
            sen_inputs_glove,  # embedding
            dtype=tf.float32
        )
        return value

    def softmax_output(self):
        value = self.lstm()
        weight = tf.Variable(tf.truncated_normal([self.lstm_units, self.num_classes]))
        bias = tf.Variable(tf.constant(0.1, shape=[self.num_classes]))

        value = tf.transpose(value, [1, 0, 2])  # [max_len, b_s, lstm_units]
        last = tf.gather(value, int(value.get_shape()[0]) - 1)  # [, lstm_units]
        prediction = (tf.matmul(last, weight) + bias)

        return prediction

    def opt_op(self):
        prediction = self.softmax_output()
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=prediction,
                labels=self.y_inputs
            )
        )
        correct_predictions = tf.equal(
            tf.argmax(prediction, 1),
            tf.argmax(self.y_inputs, 1)
        )
        accuracy = tf.reduce_mean(
            tf.cast(correct_predictions, 'float'),
            name='accuracy'
        )
        optim = tf.train.AdamOptimizer(learning_rate=0.001)  # Adam优化器
        train_op = optim.minimize(loss)  # 使用优化器最小化

        return loss, train_op, accuracy, prediction

    def train(self):
        loss_op, train_op, accuracy_op, prediction = self.opt_op()
        init = tf.global_variables_initializer()
        # saver = tf.train.Saver() 模型还未生成
        self.sess.run(init)
        saver = tf.train.Saver()
        # 划分数据batch
        iterations, next_iterator = dataset_iterator(self.x_train, self.y_train, len(self.x_train))
        max_test_acc = 0
        for epoch in range(self.n_epochs):
            for iter in range(iterations):
                x_train_batch, labels_train_batch = self.sess.run(next_iterator)
                if len(x_train_batch) < self.batch_size:
                    continue
                f_dict = {
                    self.x_inputs: x_train_batch,
                    self.y_inputs: labels_train_batch,
                    self.dropout_keep_prob: 0.5
                }
                self.sess.run([train_op, loss_op, accuracy_op], feed_dict=f_dict)

            # test
            iterations_test, next_iterator_test = dataset_iterator(self.x_dev, self.y_dev, len(self.x_dev))
            test_acc = self.test(iterations_test, next_iterator_test, epoch, loss_op, accuracy_op)

            # 保证最大准确率是最后一个保存的ckpt
            if test_acc > max_test_acc:
                saver.save(self.sess, r'D:\NLP程序相关\RNN-ATT\model_saver\gru\gru', global_step=epoch+1)
                max_test_acc = test_acc

    def test(self, iterations_test, next_iterator_test, epoch, loss_op, accuracy_op):
        test_loss = 0
        test_acc = 0
        count = 0

        for iter in range(iterations_test):
            x_test_batch, labels_test_batch = self.sess.run(next_iterator_test)

            if len(x_test_batch) < self.batch_size:
                continue

            f_dict = {
                self.x_inputs: x_test_batch,
                self.y_inputs: labels_test_batch,
                self.dropout_keep_prob: 1.0
            }

            count = count + 1
            loss, acc = self.sess.run([loss_op, accuracy_op], feed_dict=f_dict)
            test_loss = test_loss + loss
            test_acc = test_acc+acc

        test_loss = test_loss / count
        test_acc = test_acc / count
        print("-----%dth epoch, test loss: %f,  test acc: %f-----" % (epoch, test_loss, test_acc))
        return test_acc

    def sample_test(self):
        def load_word_id(text, word_index, max_len):
            word_id = []
            text_split = text.split(" ")
            for t_s in text_split:
                if t_s in word_index.keys():
                    word_id.append(word_index[t_s])
                else:
                    word_id.append(0)
            word_id = word_id + [0] * (max_len - len(word_id))
            return word_id

        def data_prepare(sess):
            x_samples = []
            y_samples = []
            iterations_test, next_iterator_test = dataset_iterator(self.x_dev, self.y_dev, len(self.x_dev), batch_size=1)
            for iter in range(iterations_test):
                x_test_batch, labels_test_batch = sess.run(next_iterator_test)
                x_samples.append(x_test_batch)
                y_samples.append(labels_test_batch)
            return x_samples, y_samples

        text = "this is a good movie ! "
        text_word_id = load_word_id(text, self.word_index, self.max_len)

        # 创建和原来一样的网络
        _, _, accuracy_op, prediction = self.opt_op()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # 导入原来的计算图(根本加载不到训练好的模型，等于随机瞎猜)
        # new_saver = tf.train.import_meta_graph(r'D:\NLP程序相关\RNN-ATT\model_saver\gru\gru-12.meta')
        # new_saver.restore(self.sess, tf.train.latest_checkpoint(r'D:\NLP程序相关\RNN-ATT\model_saver\gru'))
        saver = tf.train.Saver()
        model_file = tf.train.latest_checkpoint(r'D:\NLP程序相关\RNN-ATT\model_saver\gru')
        saver.restore(self.sess, model_file)
        # x_samples, y_samples = data_prepare(self.sess)

        f_dict = {
            self.x_inputs: [text_word_id],
            self.y_inputs: [[1, 0]],
            self.dropout_keep_prob: 1.0
        }

        acc, pre = self.sess.run([accuracy_op, prediction], feed_dict=f_dict)
        softmax_pre = np.exp(pre[0]) / np.sum(np.exp(pre[0]))

        print("输入句子为：" + text)
        print("判断为：")
        if acc == 1:
            print("正向情感")
        else:
            print("负向情感")
        print("正向情感的概率：%f, 负向情感的概率：%f" % (softmax_pre[0], softmax_pre[1]))
