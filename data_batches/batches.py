import math
import tensorflow as tf
from data_helper.sen_id_label import prepocess
from bean.file_path import *


def dataset_iterator(x_input, y_input, data_len, batch_size=64):
    train_nums = data_len
    iterations = math.ceil(train_nums / batch_size)  # 总共可以划分出来的batch数量

    # 使用from_tensor_slices将数据放入队列，使用batch和repeat划分数据批次，且让数据序列无限延续
    dataset = tf.data.Dataset.from_tensor_slices((x_input, y_input))
    dataset = dataset.batch(batch_size).repeat()

    # 使用生成器make_one_shot_iterator和get_next取数据
    iterator = dataset.make_one_shot_iterator()
    next_iterator = iterator.get_next()
    return iterations, next_iterator


# x_sample = []
# if __name__ == '__main__':
#     x_train, y_train, word_index, x_dev, y_dev, max_len = prepocess(mr_pos_path, mr_neg_path)
#     terations_test, next_iterator_test = dataset_iterator(x_dev, y_dev, len(x_dev), batch_size=1)
#     with tf.Session() as sess:
#         for epoch in range(1):
#             for iteration in range(terations_test):
#                 x_test_batch, y_test_batch = sess.run(next_iterator_test)
#                 x_sample.append(x_test_batch)
#                 print(x_test_batch)
#                 print(y_test_batch)
#                 break
#             break
# print(len(x_sample))
# print(x_sample)
