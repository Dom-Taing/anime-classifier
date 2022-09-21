from re import S
from classifier import naive_bayes_thres as nb_thres
from classifier import naive_bayes as nb
from classifier import preproc
from classifier import LSTM
import pandas as pd
import torch
import numpy as np
from torch.autograd import Variable

class Model():
    def __init__(self):
        self.genre_list = pd.read_csv('data/genre_list.csv')['genres'].values.tolist()

        pass
    def predict(self,text):
        pass
# save vocab, genre_list, count, total_number

class naive_bayes_model(Model):
    def __init__(self):
        super().__init__()
        weight = pd.read_csv('data/nb_weight.csv')
        self.weights = nb.read_weights(weight)
        self.thres = 0.5
        self.smoothing = 0.001
        self.func_list = [preproc.clean_para, preproc.bag_of_words, preproc.remove_stop_words]
        
        bag_of_word = pd.read_csv('data/count.csv')
        words = bag_of_word['words'].values
        count_num = bag_of_word['counts'].values
        self.count = {}
        self.total_number = 0
        for index in range(len(words)):
            self.count[words[index]] = count_num[index]
            self.total_number += count_num[index]
    def predict(self,text):
        x = text
        for func in self.func_list:
            x = func(x)
        sent_prob = nb_thres.find_sentence_probabilites([x], self.count, self.total_number, self.smoothing)[0]
        y_pred = nb_thres.threshold_predict(x, sent_prob, self.weights, self.genre_list, self.thres)[1]
        return y_pred

class lg_model(Model):
    def __init__(self):
        super().__init__()
        self.func_list = [preproc.clean_para, preproc.bag_of_words, preproc.remove_stop_words]
        self.threshold = np.log(0.2)

        self.vocab = set(pd.read_csv('data/vocab_no_stop_word.csv')['word'].values.tolist())
        self.model = model = torch.nn.Sequential(
            torch.nn.Linear(len(self.vocab), len(self.genre_list), bias=True),
        )
        self.model.add_module('softmax',torch.nn.LogSoftmax(dim=1))
        self.model = torch.load("../lg_best_params.params")

    def predict(self, text):
        x = text
        for func in self.func_list:
            x = func(x)
        result = []
        Y_hat = self.model.forward(x).data
        for row in range(Y_hat.size(dim=0)):
            if Y_hat[row] >= self.threshold:
                    result.append(self.genre_list[i])
        return result
        
class lstm_model(Model):
    def __init__(self):
        super().__init__()
        self.func_list = [preproc.clean_para, preproc.sentence_to_list]
        vocab = pd.read_csv('data/vocab_with_stop_word.csv')['word'].values
        vocab = set([str(word) for word in vocab])
        vocab.add('null')

        embedding = LSTM.load_glove_vectors(vocab)

        self.word_to_index = LSTM.word_to_ix(vocab)
        self.model = LSTM.BiLSTM(len(self.word_to_index), len(self.genre_list), 100, 128, embeddings=embedding)
        checkpoint = torch.load("lstm_complex_loss_weight.params")
        self.model.load_state_dict(checkpoint['state_dict'])

    def predict(self,text):
        input = text
        for func in self.func_list:
            input = func(input)

        input = LSTM.prepare_sequence(input, self.word_to_index)
        Y_hat = self.model(Variable(torch.Tensor(input)).long())
        
        result = []
        # compute dev accuracy
        for i in range(Y_hat.size(dim=0)):
            if Y_hat[i] >= 0.8:
                result.append(self.genre_list[i])
        return result

