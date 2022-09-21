import pandas as pd
import numpy as np
import json
from os.path import exists
from collections import Counter
from collections import defaultdict

# This function remove any non alphabetical or numerical character
# and convert character to lower case
# input:
# para - a string
# return:
# a string
def clean_para(para):
    ret_str = para.replace("\n\n[Written by MAL Rewrite]\n", '')
    ret_str = ''.join(char.lower() for char in ret_str if char.isalnum() or char == ' ')
    return ret_str

# This function take a string and convert it into a bag of words
# input:
# para - a string
# return:
# a counter (dict)
def bag_of_words(para):
    words = para.split()
    return Counter(words)

# This function take a string and convert it into a list of words
# input:
# para - a string
# return:
# a list of string
def sentence_to_list(para):
    words = para.split()
    return words

# This function takes the bag of words and remvoe the stop words
# input:
# syn - a bag of words (counter)
# return:
# a clean up bag of words
def remove_stop_words(syn):
    stop_word = {'the', 'a', 'he', 'she', 'it', 'i', 'we', 'you', 'they', 
                 'and', 'of', 'to', 'in', 'is', 'are', 'with', 'by', 'for', 
                 'on', 'that', 'an', 'his', 'their', 'this'}
    for word in stop_word:
        if word in syn:
            del syn[word]
    return syn

# This function clean the data set
# input:
# dataset - a pd dataframe
# list_function - a list of functiin (the return of ith function is the same type as argument of the i + 1 function)
# return:
# synopsis_list - a list of the return type of the last function in list_function
# genre_list - a numpy array with row i being the genre associate with synopsis in synopsis_list[i], each column is a genre
#              there might be column with nan as entry, guaranteed each row to have at least 1 genre (it will always
#              be in the first column)
def cleaning_data(dataset, list_functions):
    synopsis_list = dataset["synopsis"].values
    for func in list_functions:
        synopsis_list = [func(synopsis) for synopsis in synopsis_list]

    genres_list = dataset[dataset.columns[3:len(dataset.columns)]].to_numpy()
    return synopsis_list, genres_list

# This function takes in a list of counters and create a 2d numpy array (a list of vector)
# there might be multiple copy of the same vector due to synopsis having more than 1 genre
# input:
# bag_of_words - a list of counters
# num_genres - a list of the number of genre for each synopsis (list of int)
# vocab - a set of all the vocab in our training data
# return:
# a 2d numpy array
def make_numpy_bag_of_word(bag_of_words, num_genres, vocab):
    vocab = sorted(vocab)
    total_row = sum(num_genres)
    result = np.zeros((total_row, len(vocab)))
    
    vocab_dict = defaultdict(lambda: -1)
    for i in range(len(vocab)):
        vocab_dict[vocab[i]] = i
    
    index = 0
    for i in range(len(num_genres)):
        counter = bag_of_words[i]
        for j in range(num_genres[i]):
            for word in counter:
                word_index = vocab_dict[word]
                # handling case where the word does not exist in vocab
                if word_index == -1:
                    continue
                result[index][word_index] = counter[word]
            index += 1

    return result

# This function takes the 2d numpy array of labels and change it into a 2d array of 1 and 0
# the 1 means that that genre correspond to the synopsis
# input:
# y - a 2d numpy array
# genre_list - a set of all the genre
# output
# a 2d numpy array of 0 and 1
def one_hot_encoding_label(y, genre_list):
    genre_list = sorted(genre_list)
    result = np.zeros((y.shape[0], len(genre_list)))
    for row in range(y.shape[0]):
        for column in y[row]:
            if pd.isna(column):
                break
            if column not in genre_list:
                continue
            result[row][(genre_list.index(column))] = 1
    
    return result
