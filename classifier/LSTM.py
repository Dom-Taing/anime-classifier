import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from collections import defaultdict
from classifier import eval

def word_to_ix(vocab):
    vocab = sorted(vocab)
    map = {'UNK': 0}
    index = 1
    for word in vocab:
        map[word] = index
        index += 1
    return map

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else to_ix['UNK'] for w in seq]
    return idxs

def calculate_simple_loss_weights(genre_count, genre_list):
    genre_list = sorted(genre_list)
    result = np.zeros(len(genre_list))
    for genre in genre_count:
        index = genre_list.index(genre)
        result[index] = 1 / genre_count[genre]
    return torch.from_numpy(result)

def calculate_complex_loss_weights(genre_count, genre_list, num_data):
    genre_list = sorted(genre_list)
    positive_weights = [None] * len(genre_list)
    negative_weights = [None] * len(genre_list)
    
    i = 0
    for label in genre_list:
        positive_weights[i] = num_data / (2 * genre_count[label])
        negative_weights[i] = num_data / (2 * (num_data - genre_count[label]))
        i += 1
    return torch.tensor(positive_weights), torch.tensor(negative_weights)

def BCEloss_with_weight(output, target, w_p, w_n):
    # a simple demonstration of how the loss function work without using numpy function
#     loss = []
#     for i in range(output.size(dim=0)):
#         first_term = w_p[genre_list[i]] * target[i] * torch.log(output[i] + 1e-10)
#         second_term = w_n[genre_list[i]] * (1 - target[i]) + torch.log(1 - output[i] + 1e-10)
#         loss.append(first_term + second_term)
    loss_func = torch.nn.BCELoss(reduction = "none")
    first_term = target * w_p
    second_term = (1 - target) * w_n
    
    loss = loss_func(output, target)
    loss = (first_term + second_term) * loss
    return torch.mean(loss)


def load_glove_vectors(vocab, glove_file="./data/glove.6B/glove.6B.100d.txt"):
    """Load the glove word vectors"""
    word_vectors = {}
    with open(glove_file) as f:
        for line in f:
            split = line.split()
            word_vectors[split[0]] = np.array([float(x) for x in split[1:]])
    word_vectors["UNK"] = np.random.uniform(-0.25, 0.25, 100)

    word_embeddings = [word_vectors["UNK"]]
    vocab = sorted(vocab)
    for word in vocab:
        if word in word_vectors:
            embed=word_vectors[word]
        else:
            embed=word_vectors["UNK"] # make sure the UNK is in the embed
        word_embeddings.append(embed)
    
    word_embeddings = np.array(word_embeddings)
    return word_embeddings

class BiLSTM(nn.Module):
    """
    Class for the BiLSTM model tagger
    """
    
    def __init__(self, vocab_size, genre_size, embedding_dim, hidden_dim, embeddings=None):
        super(BiLSTM, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.genre_size = genre_size
        
        # add 1 due to unknown word
        self.word_embeds = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        
        if embeddings is not None:
            self.word_embeds.weight.data.copy_(torch.from_numpy(embeddings))
        
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim)

        self.hidden2tag = nn.Linear(in_features=hidden_dim, out_features=genre_size, bias=True)

    def forward(self, sentence):
        """
        The function obtain the scores for each tag for each of the words in a sentence
        Input:
        sentence: a sequence of ids for each word in the sentence
        Make sure to reshape the embeddings of the words before sending them to the BiLSTM. 
        The axes semantics are: seq_len, mini_batch, embedding_dim
        Output: 
        returns lstm_feats: scores for each tag for each token in the sentence.
        """
        x = self.word_embeds(sentence)
        lstm_out, (ht, ct) = self.lstm(x)
        output = self.hidden2tag(ht[-1])
        sigmoid_layer = torch.nn.Sigmoid()
        probs = sigmoid_layer(output)
        return probs

def train_model(model, X_tr, Y_tr, w_p, w_n, X_dv=None, Y_dv = None, num_its=50, status_frequency=10,
               optim_args = {'lr':0.1},
               param_file = 'best.params'):
    
    #initialize optimizer
    optimizer = optim.SGD(model.parameters(), **optim_args)
    
    losses=[]
    accuracies=[]
    f_score_list = []
    
    for epoch in range(num_its):
        
        model.train()
        loss_value=0
        count1=0
        
        for X,Y in zip(X_tr,Y_tr):
            X_tr_var = Variable(torch.Tensor(X)).long()
            Y_tr_var = Variable(torch.from_numpy(Y))
            
            y_pred = model(X_tr_var)
            # set gradient to zero
            optimizer.zero_grad()
            
            output = BCEloss_with_weight(y_pred, Y_tr_var, w_p, w_n)
            
            output.backward()
            optimizer.step()
            loss_value += output.item()
            count1+=1
            
        losses.append(loss_value/count1)
        
        # write parameters if this is the best epoch yet
        acc=0        
        if X_dv is not None and Y_dv is not None:
            acc=0
            index=0
            y_pred = np.zeros((Y_dv.shape[0],Y_dv.shape[1]))
            for Xdv, Ydv in zip(X_dv, Y_dv):
                
                X_dv_var = Variable(torch.Tensor(Xdv)).long()
                # run forward on dev data
                Y_hat = model(X_dv_var)
                
                # compute dev accuracy
                for i in range(Y_hat.size(dim=0)):
                    if Y_hat[i] >= 0.5:
                        Y_hat[i] = 1
                    else:
                        Y_hat[i] = 0
                y_pred[index] = Y_hat.tolist()
                index += 1
                # save
            acc = eval.accuracy(y_pred, Y_dv)
            f_score = eval.f_score(y_pred, Y_dv)

            # we want the epoch with the highest f_score
            if len(f_score_list) == 0 or f_score > max(f_score_list):
                state = {'state_dict':model.state_dict(),
                         'epoch':len(accuracies)+1,
                         'accuracy':acc}
                torch.save(state,param_file)
            accuracies.append(acc)
            f_score_list.append(f_score)
        # print status message if desired
        if status_frequency > 0 and epoch % status_frequency == 0:
            print("Epoch "+str(epoch+1)+": Dev Accuracy: "+str(acc))
            print("Epoch "+str(epoch+1)+": Dev F_score: "+str(f_score))
    return model, losses, accuracies


