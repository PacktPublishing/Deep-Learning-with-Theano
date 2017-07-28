from __future__ import division
from __future__ import print_function

import argparse

import re
from nltk.tokenize import TweetTokenizer
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Bidirectional, Dense, Activation

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical


def read_data(file_name):
    tweets = []
    labels = []

    polarity2idx = {'positive': 0, 'negative': 1, 'neutral': 2}

    with open(file_name) as fin:
        for line in fin:
            _, _, _, _, polarity, tweet = line.strip().split("\t")
            tweet = process(tweet)
            cls = polarity2idx[polarity]  # transform the polarity to int value {0, 1, or 2}
            tweets.append(tweet)
            labels.append(cls)

    return tweets, labels


def process(tweet):
    tknz = TweetTokenizer()

    tokens = tknz.tokenize(tweet)

    tweet = " ".join(tokens)  # we need to ensure that all the tokens are separated using the space

    tweet = tweet.lower()

    tweet = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', '<url>', tweet)  # URLs
    tweet = re.sub(r'(?:@[\w_]+)', '<user>', tweet)  # user-mentions
    tweet = re.sub(r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)', '<hashtag>', tweet)  # hashtags
    tweet = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', '<number>', tweet)  # numbers

    return tweet.split(" ")  # return a list of tokens


def get_vocabulary(data):
    max_len = 0

    index = 0

    word2idx = {'<unknown>': index}  # dictionary to map each word to an index

    for tweet in data:

        max_len = max(max_len, len(tweet))

        for word in tweet:

            if word not in word2idx:
                index += 1
                word2idx[word] = index

    return word2idx, max_len


def transfer(data, word2idx):
    transfer_data = []

    for tweet in data:

        tweet2vec = []

        for word in tweet:

            if word in word2idx:  # the word exists in the vocabulary
                tweet2vec.append(word2idx[word])  # we map it to its index
            else:  # out of vocabulary
                tweet2vec.append(0)  # we map it to the unknown token

        transfer_data.append(tweet2vec)

    return transfer_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sentiment Analysis of Tweets using BI-LSTM')

    parser.add_argument('-t', dest="train_file", default="sem_eval2103.train", type=str)
    parser.add_argument('-d', dest="dev_file", default="sem_eval2103.dev", type=str)
    parser.add_argument('-v', dest="test_file", default="sem_eval2103.test", type=str)

    parser.add_argument('-D', dest="emb_size", default=100, type=int)
    parser.add_argument('-R', dest="rnn_size", default=64, type=int)
    parser.add_argument('-C', dest="nb_classes", default=3, type=int)

    parser.add_argument('-e', dest="nb_epochs", default=30, type=int)
    parser.add_argument('-b', dest="batch_size", default=10, type=int)

    args = parser.parse_args()

    # load the train and dev data
    train_tweets, y_train = read_data(args.train_file)
    dev_tweets, y_dev = read_data(args.dev_file)

    word2idx, max_len = get_vocabulary(train_tweets)
    vocab_size = len(word2idx)

    X_train = transfer(train_tweets, word2idx)

    del train_tweets  # for saving memory

    X_dev = transfer(dev_tweets, word2idx)

    del dev_tweets

    # pad the sequencess to ensure that we have a fixed length
    X_train = pad_sequences(X_train, maxlen=max_len, truncating='post')
    X_dev = pad_sequences(X_dev, maxlen=max_len, truncating='post')

    # convert the classes into a categorical

    y_train = to_categorical(y_train, args.nb_classes)
    y_dev = to_categorical(y_dev, args.nb_classes)

    print("Train size: ", X_train.shape)
    print("Dev size: ", X_dev.shape)

    # define the model

    model = Sequential()
    emb_layer = Embedding(vocab_size + 1, output_dim=args.emb_size, input_length=max_len)
    model.add(emb_layer)
    rnn_size = 64
    lstm = LSTM(args.rnn_size)
    bi_lstm = Bidirectional(lstm)
    model.add(bi_lstm)

    fc = Dense(args.nb_classes)
    classifier = Activation('softmax')
    model.add(fc)
    model.add(classifier)
    print(model.summary())

    # compile the model
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # train the model

    model.fit(x=X_train, y=y_train, batch_size=args.batch_size, epochs=args.nb_epochs, validation_data=[X_dev, y_dev])
    # evaluate the model
    test_tweets, y_test = read_data(args.test_file)

    X_test = transfer(test_tweets, word2idx)

    del test_tweets

    # pad the sequencess to ensure that we have a fixed length
    X_test = pad_sequences(X_test, maxlen=max_len, truncating='post')

    # convert the classes into a categorical

    y_test = to_categorical(y_test)

    print("Test size: ", X_test.shape)

    test_loss, test_acc = model.evaluate(X_test, y_test)

    print("Testing loss: {:.5}; Testing Accuracy: {:.2%}".format(test_loss, test_acc))

    # save the model

    model.save('bi_lstm_sentiment.h5')
