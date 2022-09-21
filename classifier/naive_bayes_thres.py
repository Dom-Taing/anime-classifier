import pandas as pd
import numpy as np
import json
from os.path import exists
from collections import Counter
from collections import defaultdict
import random
from classifier import naive_bayes, eval, preproc
from classifier.naive_bayes import OFFSET

# This function takes in the data points and return a counter that's a bag of words for the whole corpus
# input:
# x_tr - a list of counters of the synopsis
# y_tr - a numpy array with row i being the genre associate with synopsis in x_tr[i], each column is a genre
#        there might be column with nan as entry, guaranteed each row to have at least 1 genre (it will always
#        be in the first column)
# return:
# result - a bag of words for the whole corpus
# total_number - the total number of word in the whole corpus
def total_word_count(x_tr, y_tr):
    count, words = naive_bayes.count_words(x_tr, y_tr)
    result = Counter()
    total_number = 0
    for genre in count:
        for word in count[genre]:
            result[word] += count[genre][word]
            total_number += count[genre][word]
    return result, total_number

# find the probabilities of the all the synopsis:
# input:
# x - a list of counter of words
# count - a counter of all the word in the training data
# total_number - the total number of word in the training data
# smoother - the smoothing value
# return 
# a list of probabilites
def find_sentence_probabilites(x, count, total_number, smoother):
    
    total_number += smoother * len(count)
    probabilites = []
    for counter in x:
        prob = 0
        for word in counter:
            if word not in count:
                continue
            prob += np.log((count[word] + smoother) / total_number) * counter[word]
        probabilites.append(prob)
    return probabilites

# predict the results of x
# input:
# x - a counter of words of the synopsis
# sent_prob - the probabilities of the sentence occuring in our corpus
# weights - weights we'll use for predicting
# genre_list - a set of genre that we need to evaluate
# threshold - a threshold in which only certain genres that achieve over it can pass
#
# output:
# a dict with each genre as the key and the score as it's value
# a list of genres we predict to be most likely
def threshold_predict(x, sent_prob, weights, genre_list, threshold):
    # score = defaultdict(float)
    # result = []
    # for genre in genre_list:
    #     score[genre] += weights[(OFFSET, genre)]
    #     for word in x:
    #         # unseen word would just be thrown away
    #         score[genre] += weights[(word, genre)] * x[word]
    # total_score = 0
    score = naive_bayes.predict(x, weights, genre_list, 0)[0]
    result = []
    for genre in score:
        # total_score += np.exp(score[genre] - sent_prob)
        if score[genre] - sent_prob >= np.log(threshold + 1e-10):
            result.append(genre)
    return score, result

# This function predict the label of all synopsis in x using the weights
# it give a score to every genre and the higher the score is more likely it's correct
# input:
# x - a list of counter of words of the synopsis
# sentence_prob - a list of probabilities of sentences the order matters
# weights - weights we'll use for predicting
# genre_list - a set of genre that we need to evaluate
# threshold - an int that's a threshold in which only certain genres that achieve over it can pass
#
# output:
# a list of predictions for our input
def threshold_predict_all(x, sentence_prob, weights, genre_list, threshold):
    result = []
    for i in range(len(x)):
        result.append(threshold_predict(x[i], sentence_prob[i], weights, genre_list, threshold)[1])
    result = np.array(result, dtype=object)
    return result

# hyper_parameter tuning by using random 
# x_tr - a list of counter that's the synopsis in the train dataset
# y_tr - a 2d numpy array representing the actual genres of the train dataset
# x_dev - a list of counter that's the synopsis in the dev dataset
# y_dev - a 2d numpy array representing the actual genres of the dev dataset
# smoothers - a list of smoothing value
# thresholds - a list of thresholds value
# num_tries - an int that's the number of loop we want to go over
# return 
# a tuple where the first element is smoothing value and the second is the threshold
def threshold_find_best_hyperparameter(x_tr, y_tr, x_dev, y_dev, smoothers, thresholds, num_tries):

    genre_list = naive_bayes.get_label_count(y_tr)[1]
    count, total_number = total_word_count(x_tr, y_tr)

    max_acc = 0
    best_hp = None

    y_dev = preproc.one_hot_encoding_label(y_dev, genre_list)
    for i in range(num_tries):
        smoothing = random.choice(smoothers)
        threshold = random.choice(thresholds)
        weights = naive_bayes.calculating_weights(x_tr, y_tr, smoothing)
        sentence_probabilites = find_sentence_probabilites(x_dev, count, total_number, smoothing)
        
        y_pred = threshold_predict_all(x_dev, sentence_probabilites, weights, genre_list, threshold)
        y_pred = preproc.one_hot_encoding_label(y_pred, genre_list)
        acc = eval.accuracy(y_pred, y_dev)
        if acc > max_acc:
            max_acc = acc
            best_hp = (smoothing, threshold)
    return best_hp