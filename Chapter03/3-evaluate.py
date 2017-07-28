from __future__ import division
from __future__ import print_function
import argparse
import theano
import theano.tensor as T
from utils import *
from model import CBOW
import math
import numpy as np
import six.moves.cPickle as pickle

def get_analogy_prediction_model(embeddings, emb_size, vocab_size):

    # The eval feeds three vectors of word ids for a, b, c, each of
    # which is of size bsz, where bsz is the number of analogies we want to
    # evaluate in one batch.
    analogy_a = T.ivector('analogy_a')  
    analogy_b = T.ivector('analogy_b')  
    analogy_c = T.ivector('analogy_c')  

    # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
    
    a_emb = embeddings[analogy_a]  # a's embs
    b_emb = embeddings[analogy_b]  # b's embs
    c_emb = embeddings[analogy_c]  # c's embs

    # We expect that d's embedding vectors on the unit hyper-sphere is
    # near: c_emb + (b_emb - a_emb), which has the shape [bsz, emb_size].
    target = c_emb + (b_emb - a_emb)

    # Compute cosine distance between each pair of target and vocab.
    # dist has shape [bsz, vocab_size].
    dist = T.dot(target, embeddings.T)

    # For each question (row in dist), find the top 4 words.
    pred_idx = T.argsort(dist, axis=1)[:, -4:]

    prediction_fn = theano.function([analogy_a, analogy_b, analogy_c], pred_idx)

    return prediction_fn

def read_analogies(fname, word2idx):
    """Reads through the analogy question file.
    Returns:
      questions: a [n, 4] numpy array containing the analogy question's
                 word ids.
      questions_skipped: questions skipped due to unknown words.
    """
    questions = []
    questions_skipped = 0
    with open(fname, "r") as analogy_f:
      for line in analogy_f:
        if line.startswith(":"):  # Skip comments.
          continue
        words = line.strip().lower().split(" ")
        ids = [word2idx.get(w.strip()) for w in words]
        if None in ids or len(ids) != 4:
          questions_skipped += 1
        else:
          questions.append(np.array(ids))
    print("Eval analogy file: ", fname)
    print("Questions: ", len(questions))
    print("Skipped: ", questions_skipped)

    return np.array(questions, dtype=np.int32)





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx2word_file', default='idx2word.pkl', type=str)
    parser.add_argument('--params_file', default='model-1-epoch', type=str)
    parser.add_argument('--emb_size', default=128, type=int)
    parser.add_argument('--eval_data', default='questions-words.txt', type=str)
    args = parser.parse_args()

    emb_size = args.emb_size
    with open(args.idx2word_file, 'rb') as f:
        idx2word = pickle.load(f)
    
    vocab_size = len(idx2word)

    word2idx = dict([(idx2word[idx], idx) for idx in idx2word])

    _, _, params = CBOW(vocab_size, emb_size)
    load_params(args.params_file, params)

    embeddings = params[0]
    norm = T.sqrt(T.sum(T.sqr(embeddings), axis=1, keepdims=True))
    normalized_embeddings = embeddings / norm

    predict = get_analogy_prediction_model(normalized_embeddings, emb_size, vocab_size)    

    """Evaluate analogy questions and reports accuracy."""

    # How many questions we get right at precision@1.
    correct = 0
    analogy_data = read_analogies(args.eval_data, word2idx)
    analogy_questions = analogy_data[:, :3]
    answers = analogy_data[:, 3]
    del analogy_data
    total = analogy_questions.shape[0]
    start = 0

    while start < total:
      limit = start + 200
      sub_questions = analogy_questions[start:limit, :]
      sub_answers = answers[start:limit]
      idx = predict(sub_questions[:,0], sub_questions[:,1], sub_questions[:,2])

      start = limit
      for question in xrange(sub_questions.shape[0]):
        for j in xrange(4):
          if idx[question, j] == sub_answers[question]:
            # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
            correct += 1
            break
          elif idx[question, j] in sub_questions[question]:
            # We need to skip words already in the question.
            continue
          else:
            # The correct label is not the precision@1
            break
    print()
    print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total,
                                              correct * 100.0 / total))
