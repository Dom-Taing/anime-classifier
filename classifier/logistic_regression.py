import numpy as np
import pandas as pd
import torch
from torch import optim

from collections import Counter
from collections import defaultdict
from classifier import eval

# This function create a list of the number of genre for each synopsis
# input:
# y - a 2d numpy array of the genre
# output:
# a list of int
def make_num_genres(y):
    result = []
    for row in y:
        num_genre = 0
        for column in row:
            if pd.isna(column):
                break
            num_genre += 1
        result.append(num_genre)
    return result

# This function takes the 2d numpy array of labels and change it into a 1d array of int
# the int is the representation of the label (the index of the label when it's sorted)
# input:
# y - a 2d numpy array
# genre_list - a set of all the genre
# output
# a 1d numpy array of int
def make_numpy_label(y, genre_list):
    genre_list = sorted(genre_list)
    result = []
    for row in y:
        for column in row:
            if pd.isna(column):
                break
            result.append(genre_list.index(column))
    return np.array(result)

# This function trains the model
def train_model(loss, model, X_tr_var, Y_tr_var,
                num_its = 200,
                threshold = 0.1,
                X_dv_var = None,
                Y_dv_var = None,
                status_frequency=10,
                optim_args = {'lr':0.002,'momentum':0},
                param_file = 'best.params'):

    # initialize optimizer
    optimizer = optim.SGD(model.parameters(), **optim_args)

    losses = []
    accuracies = []

    for epoch in range(num_its):
        # set gradient to zero
        optimizer.zero_grad()
        # run model forward to produce loss
        output = loss.forward(model.forward(X_tr_var),Y_tr_var)
        # backpropagate and train
        output.backward()
        optimizer.step()

        losses.append(output.item())

        # write parameters if this is the best epoch yet
        if X_dv_var is not None:
            # run forward on dev data
            Y_hat = model.forward(X_dv_var).data
            for row in range(Y_hat.size()[0]):
                for column in range(Y_hat.size()[1]):
                    if Y_hat[row][column] >= threshold:
                        Y_hat[row][column] = 1
                    else:
                        Y_hat[row][column] = 0
            # compute dev accuracy
            acc = eval.accuracy(Y_hat.data.numpy().astype(int), Y_dv_var.data.numpy())
            f_score = eval.f_score(Y_hat.data.numpy().astype(int), Y_dv_var.data.numpy())
            # save
            if len(accuracies) == 0 or acc > max(accuracies):
                state = {'state_dict':model.state_dict(),
                         'epoch':len(accuracies)+1,
                         'accuracy':acc}
                torch.save(state,param_file)
            accuracies.append(acc)

        # print status message if desired
        if status_frequency > 0 and epoch % status_frequency == 0:
            print("Epoch "+str(epoch+1)+": Dev Accuracy: "+str(acc))
            print("Epoch "+str(epoch+1)+": Dev F_score: "+str(f_score))

    # load parameters of best model
    checkpoint = torch.load(param_file)
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, losses, accuracies


def train_1_label_model(loss, model, X_tr_var, Y_tr_var,
                num_its = 200,
                X_dv_var = None,
                Y_dv_var = None,
                status_frequency=10,
                optim_args = {'lr':0.002,'momentum':0},
                param_file = 'best.params'):

    # initialize optimizer
    optimizer = optim.SGD(model.parameters(), **optim_args)

    losses = []
    accuracies = []

    for epoch in range(num_its):
        # set gradient to zero
        optimizer.zero_grad()
        # run model forward to produce loss
        output = loss.forward(model.forward(X_tr_var),Y_tr_var)
        # backpropagate and train
        output.backward()
        optimizer.step()

        losses.append(output.item())

        # write parameters if this is the best epoch yet
        if X_dv_var is not None:
            # run forward on dev data
            _, Y_hat = model.forward(X_dv_var).max(dim=1)
            # compute dev accuracy
            acc = eval.accu(Y_hat.data.numpy(),Y_dv_var.data.numpy())
            # save
            if len(accuracies) == 0 or acc > max(accuracies):
                state = {'state_dict':model.state_dict(),
                         'epoch':len(accuracies)+1,
                         'accuracy':acc}
                torch.save(state,param_file)
            accuracies.append(acc)

        # print status message if desired
        if status_frequency > 0 and epoch % status_frequency == 0:
            print("Epoch "+str(epoch+1)+": Dev Accuracy: "+str(acc))

    # load parameters of best model
    checkpoint = torch.load(param_file)
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, losses, accuracies

