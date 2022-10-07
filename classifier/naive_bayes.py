import pandas as pd
import numpy as np
import json
from os.path import exists
from collections import Counter
from collections import defaultdict
from classifier import eval, preproc

OFFSET = '**OFFSET**'

def prune_vocab(total_counts, target_data, min_count):
    data_new = []
    for bag_of_word in target_data:
        bag_of_word_new = Counter()
        for word in bag_of_word:
            if total_counts[word] >= min_count:
                bag_of_word_new[word] = bag_of_word[word]
        data_new.append(bag_of_word_new)
    return data_new
    

# This function calculate the count of each word in the label,
# it also returns a set of all the word in the training data
# input:
# x_tr - a list of counters of the synopsis
# y_tr - a numpy array with row i being the genre associate with synopsis in x_tr[i], each column is a genre
#        there might be column with nan as entry, guaranteed each row to have at least 1 genre (it will always
#        be in the first column)
# return:
# count - a defaultdict of counters, with the genre as the key
# words - a set of word
def count_words(x_tr, y_tr):
    count = defaultdict(lambda: Counter())
    words = set()
    for i in range(len(x_tr)):
        for genre in y_tr[i]:
            if pd.isna(genre):
                break
            for word in x_tr[i]:
                count[genre][word] += 1
                words.add(word)
    return count, words

# This function count the number of times a label occur in the training data
# it also return a set of the genre in the training dataa
# input:
# y_tr - a numpy array with row i being the genre associate with synopsis in x_tr[i], each column is a genre
#        there might be column with nan as entry, guaranteed each row to have at least 1 genre (it will always
#        be in the first column)
# return:
# a Counter with the genre as key and the value is the number of it's occurence
# a set of all the genre
def get_label_count(y_tr):
    label_count = Counter()
    genre_list = set()
    for row in y_tr:
        for genre in row:
            if pd.isna(genre):
                break
            label_count[genre] += 1
            genre_list.add(genre)
    return label_count, genre_list

# This function calculate the weights use for prediction from the training data
# input:
# x_tr - a list of counters of the synopsis
# y_tr - a numpy array with row i being the genre associate with synopsis in x_tr[i], each column is a genre
#        there might be column with nan as entry, guaranteed each row to have at least 1 genre (it will always
#        be in the first column)
# return:
# weights - a defaultdict with (word, genre) as the key and the value is the weight use
#           for prediction
def calculating_weights(x_tr, y_tr, smoothing):
    weights = defaultdict(int)

    count, vocab = count_words(x_tr, y_tr)
    
    genre_count, genre_list = get_label_count(y_tr)
    
    total_genre_count = 0
    for genre in genre_count:
        total_genre_count += genre_count[genre]
    
    # This calculate the denominator of the likelihood of the naivebayes equation
    total = defaultdict(float)
    for genre in genre_list:
        for word in vocab:
            total[genre] += count[genre][word] + smoothing
        
    for genre in genre_list:
        for word in vocab:
            weights[(word, genre)] = np.log((count[genre][word] + smoothing) / total[genre])
        weights[(OFFSET, genre)] = np.log(genre_count[genre] / total_genre_count)

    return weights

# This function predict the label of x using the weights
# it give a score to every genre and the higher the score is more likely it's correct
# input:
# x - a counter of words of the synopsis
# weights - weights we'll use for predicting
# genre_list - a set of genre that we need to evaluate
# amount - the number of top level tags to output 
#
# output:
# a dict with each genre as the key and the score as it's value
# a list of genres we predict to be most likely
def predict(x, weights, genre_list, amount):
    score = defaultdict(float)
    for genre in genre_list:
        score[genre] += weights[(OFFSET, genre)]
        for word in x:
            # unseen word would just be thrown away
            score[genre] += weights[(word, genre)] * x[word]
            
    top = sorted(score.items(), key=lambda item: item[1], reverse=True)[:amount]
    return score, [key[0] for key in top]

# This function predict the label of all synopsis in x using the weights
# it give a score to every genre and the higher the score is more likely it's correct
# input:
# x - a list of counter of words of the synopsis
# weights - weights we'll use for predicting
# genre_list - a set of genre that we need to evaluate
# amount - a list of amount of top prediction we want for x[i] (same size as x)
#
# output:
# a list of predictions for our input
def predict_all(x, weights, genre_list, amount):
    result = []
    for i in range(len(x)):
        result.append(predict(x[i], weights, genre_list, amount[i])[1])
    
    result = np.array(result, dtype=object)
    return result

# This function create the amount list for the predict_all method
# input:
# y - a 2d numpy array representing the genres
# output:
# a list of int that represent the number of genres in that synopsis
# len of output is the same as the number of row in y
def get_amount_list(y):
    result = []
    for row in y:
        num_genres = 0
        for column in row:
            if pd.isna(column):
                break
            num_genres += 1
        result.append(num_genres)
    return result

# This function go through a list of smoothing value and choose the one with the
# highest accuracy on the dev dataset
# input:
# x_tr - a list of counter that's the synopsis in the train dataset
# y_tr - a 2d numpy array representing the actual genres of the train dataset
# x_dev - a list of counter that's the synopsis in the dev dataset
# y_dev - a 2d numpy array representing the actual genres of the dev dataset
# smoothers - a list of smoothing value
# output:
# a float that's the best smoothing value
def find_best_smoother(x_tr, y_tr, x_dev, y_dev, smoothers):
    genre_list = get_label_count(y_tr)[1]
    amount_list = get_amount_list(y_dev)
    max_acc = 0
    best_smoother = 0.1

    y_dev = preproc.one_hot_encoding_label(y_dev, genre_list)
    for smoothing in smoothers:
        weights = calculating_weights(x_tr, y_tr, smoothing)
        y_pred = predict_all(x_dev, weights, genre_list, amount_list)
#         y_pred = predict_all(x_dev, weights, genre_list, amount_list)
        y_pred = preproc.one_hot_encoding_label(y_pred, genre_list)
        acc = eval.accuracy(y_pred, y_dev)
        if acc > max_acc:
            max_acc = acc
            best_smoother = smoothing
    return best_smoother

# This function takes the weight and save it in a csv file using panda dataframe
# input:
# weights - the weight we got from calculating weight (key - (word, genre), value - score)
# filename - saving the weight to the file with the name filename
def save_weights(weights, filename="nb_weight"):
    file = filename
    if exists(file) == False:
        panda_dict = {'word': [], 'genre': [], 'score': []}
        for key in weights:
            panda_dict["word"].append(key[0])
            panda_dict["genre"].append(key[1])
            panda_dict["score"].append(weights[key])
        df = pd.DataFrame(data=panda_dict)
        df.to_csv(file, index=False)
        return df

# This function takes the weights that was save as a dataframe and turn it into a dict
# input:
# df_weights - our weights in dataframe form
# return:
# a dict of the weights
def read_weights(df_weights):
    weights = defaultdict(int)
    word = df_weights["word"].values
    genre = df_weights["genre"].values
    score = df_weights["score"].values
    for i in range(word.size):
        weights[(word[i], genre[i])] = score[i]
    return weights