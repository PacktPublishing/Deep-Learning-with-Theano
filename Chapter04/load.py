### borrowed from Denny Britz with permission
### https://github.com/dennybritz/rnn-tutorial-rnnlm/blob/master/train-theano.py
from __future__ import print_function
import nltk
import itertools
import numpy as np

unknown_token = "UNKNOWN"
sentence_start_token = "START"
sentence_end_token = "END"

def parse_text(filename, vocabulary_size=9000, type="word"):
    with open(filename, 'rb') as f:
        txt = f.read()
        if type == "word":
            sentences = nltk.sent_tokenize(txt.decode('utf-8').lower().replace('\n', ' '))
            # sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
            tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
            word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
            print("Found %d unique words tokens." % len(word_freq.items()))
            vocab = word_freq.most_common(vocabulary_size-1)
            index = [sentence_start_token, sentence_end_token, unknown_token] + [x[0] for x in vocab]
            word_to_index = dict([(w,i) for i,w in enumerate(index)])
            print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))
            for i, sent in enumerate(tokenized_sentences):
                tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
            X_train = np.asarray([ [0]+[word_to_index[w] for w in sent] for sent in tokenized_sentences])
            y_train = np.asarray([ [word_to_index[w] for w in sent]+[1] for sent in tokenized_sentences])
            # X_train, y_train = [], []
            # for sent in tokenized_sentences:
            #     l = len(sent) - 1
            #     X_train.append(coo_matrix((np.ones( (l) ), ( range(l), [word_to_index[w] for w in sent[:-1]] )), shape=(l, vocabulary_size )).toarray())
            #     y_train.append( [word_to_index[w] for w in sent[1:] ] )
        else:
            sentences = nltk.sent_tokenize(txt.decode('utf-8').lower().replace('\n', ' '))
            index = ['^','$'] + list(set(txt))
            char_to_index = dict([(w,i) for i,w in enumerate(index)])
            X_train = np.asarray([ [0]+[ char_to_index[w] for w in sent]  for sent in sentences])
            y_train = np.asarray([ [ char_to_index[w] for w in sent]+[1] for sent in sentences])

    return X_train, y_train, index
