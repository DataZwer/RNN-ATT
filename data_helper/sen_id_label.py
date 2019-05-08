import numpy as np
from tensorflow.contrib import learn
import re


def clean_str(string):
    """
    Clean the text data using the same code as the original paper
    from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    :param string: input string to process
    :return: a sentence with lower representation and delete spaces
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Load positive and negative sentences from the raw data files
    :param positive_data_file: pos
    :param negative_data_file: neg
    :return: split sentences and labels
    """
    # load file

    positive_examples = list(open(positive_data_file, 'r', encoding='utf-8', errors='ignore').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, 'r', encoding='utf-8', errors='ignore').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # clean sents
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # generate labels
    positive_labels = [[1, 0] for _ in positive_examples]
    negative_labels = [[0, 1] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def prepocess(mr_pos_path, mr_neg_path):
    """
    For load and process data
    :return:
    """
    print("Loading data...")
    x_text, y = load_data_and_labels(mr_pos_path, mr_neg_path)
    # bulid vocabulary
    max_document_length = max(len(x.split(' ')) for x in x_text)
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # shuffle
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # split train/test dataset
    dev_sample_index = -1 * int(.1 * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    del x, y, x_shuffled, y_shuffled

    print('Vocabulary Size: {:d}'.format(len(vocab_processor.vocabulary_)))
    print('Train/Dev split: {:d}/{:d}'.format(len(y_train), len(y_dev)))

    return x_train, y_train, vocab_processor.vocabulary_._mapping, x_dev, y_dev, max_document_length
