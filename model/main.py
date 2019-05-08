import tensorflow as tf
from bean.configuration import config_rnn
from model import lstm, lstm_att, gru, gru_att


def modle_train(dl_model, c):
    with tf.Session() as sess:
        model = dl_model(c, sess)
        model.train()


def model_test(dl_model, c):
    with tf.Session() as sess:
        model = dl_model(c, sess)
        model.sample_test()



# modle_train(gru_att.gru_att_model, config_rnn)
# modle_train(gru.gru_model, config_rnn)
modle_train(lstm.lstm_model, config_rnn)
# modle_train(lstm_att.lstm_att_model, config_rnn)


# model_test(gru_att.gru_att_model, config_rnn)

# mean_test_acc, max_test_acc = modle_train(lstm.lstm_model, config_lstm)

# mean_test_acc, max_test_acc = modle_train(lstm_att.lstm_att_model, config_rnn)
# mean_test_acc, max_test_acc = modle_train(gru_att.gru_model, config_lstm)
# mean_test_acc, max_test_acc = modle_train(gru_att.gru_att_model, config_rnn)
