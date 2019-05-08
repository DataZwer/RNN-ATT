import tensorflow as tf
from data_helper.sen_id_label import prepocess
from bean.file_path import *
import numpy as np
from data_batches.batches import dataset_iterator

x_train, y_train, word_index, x_dev, y_dev, max_len = prepocess(mr_pos_path, mr_neg_path)


# def data_prepare(sess):
#     x_samples = []
#     y_samples = []
#     iterations_test, next_iterator_test = dataset_iterator(x_dev, y_dev, len(x_dev), batch_size=1)
#     x_test_batch, labels_test_batch = sess.run(next_iterator_test)
#     x_samples.append(x_test_batch)
#     y_samples.append(labels_test_batch)
#     return x_samples, y_samples

#
# def load_word_id(text, word_index, max_len):
#     word_id = []
#     text_split = text.split(" ")
#     for t_s in text_split:
#         if t_s in word_index.keys():
#             word_id.append(word_index[t_s])
#         else:
#             word_id.append(0)
#     word_id = word_id+[0]*(max_len-len(word_id))
#     return word_id




# trimmed_filename = r'D:\NLP程序相关\RNN-ATT\data\glove.840B.300d.mr.npz'
# text = "the movie is no good enough !"
# text_word_id = load_word_id(text, word_index, max_len)
# print("hehe")
# embedding_maxtrix = tf.Variable(load_word_embedding(trimmed_filename), name="embedding")
# text_embedding = tf.nn.embedding_lookup(embedding_maxtrix, text_word_id)
# init = tf.global_variables_initializer()

# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(text_embedding)[:4])
# with tf.Session() as sess:
#     v1 = tf.Variable()
#     x_samples, y_samples = data_prepare(sess)
#     new_saver = tf.train.import_meta_graph('D:\NLP程序相关\RNN-ATT\model_saver\gru_att\gru_att-6.meta')
#     new_saver.restore(sess, tf.train.latest_checkpoint('D:\NLP程序相关\RNN-ATT\model_saver\gru_att'))
#     graph = tf.get_default_graph()
#     tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
#     att_alpha = graph.get_tensor_by_name("att_alpha:0")
#     print(att_alpha.shape)
#     all_att = sess.run(att_alpha)
#     print(all_att[1])
#
#     embed = graph.get_tensor_by_name("embedding/glove_w2v:0")
#     print(sess.run(embed))






