from flask import Flask
from flask import render_template
from flask import request
import torch
import pandas as pd
from torch.autograd import Variable
import numpy as np
from classifier import model

# train_data = pd.read_csv('data/train_data.csv')

# func_list = [preproc.clean_para, preproc.sentence_to_list]
# x_tr, y_tr = preproc.cleaning_data(train_data, func_list)
# vocab = nb.count_words(x_tr, y_tr)[1]
# genre_count, genre_list = nb.get_label_count(y_tr)
# genre_list = sorted(genre_list)
# embedding = LSTM.load_glove_vectors(vocab)
# word_to_index = LSTM.word_to_ix(vocab)

# model = LSTM.BiLSTM(len(word_to_index), len(genre_list), 100, 128, embeddings=embedding)
# checkpoint = torch.load("lstm_complex_loss_weight.params")
# model.load_state_dict(checkpoint['state_dict'])

my_model = model.lstm_model()
# weights = nb.read_weights("nb_weight.csv")

# print a nice greeting.
def say_hello(username = "World"):
    return '<p>Hello %s!</p>\n' % username

# a function that uses the model to predict the genre
def predict(input):
    # input = preproc.clean_para(input)
    # input = preproc.sentence_to_list(input)
    # input = LSTM.prepare_sequence(input, word_to_index)
    # Y_hat = model(Variable(torch.Tensor(input)).long())
    
    # result = []
    # # compute dev accuracy
    # for i in range(Y_hat.size(dim=0)):
    #     if Y_hat[i] >= 0.5:
    #         result.append(genre_list[i])
    
    # result_str = ""
    # for genre in result:
    #     result_str += genre + " "
    # return result_str
    return my_model.predict(input)

# def predict(input):
#     input = preproc.clean_para(input)
#     input = preproc.bag_of_words(input)
#     input = preproc.remove_stop_words(input)
#     output = nb_thres.
    
#     result = []
#     # compute dev accuracy
#     for i in range(Y_hat.size(dim=0)):
#         if Y_hat[i] >= 0.5:
#             result.append(genre_list[i])
    
#     result_str = ""
#     for genre in result:
#         result_str += genre + " "
#     return result_str

# some bits of text for the page.
header_text = '''
    <html>\n<head> <title>EB Flask Test</title> </head>\n<body>'''
instructions = '''
    <p><em>Hint</em>: This is a RESTful web service! Append a username
    to the URL (for example: <code>/Thelonious</code>) to say hello to
    someone specific.</p>\n'''
home_link = '<p><a href="/">Back</a></p>\n'
footer_text = '</body>\n</html>'

# EB looks for an 'application' callable by default.
application = Flask(__name__)

@application.route('/', methods=['GET', 'POST'])
def synopsis_request():
    if request.method == 'POST':
        genres = predict(request.form["synopsis"])
    else:
        genres = []
    return render_template('index.html', genres=genres)

# @application.route('/home/')
# def home_page():
#     return render_template('home.html')

# @application.route('/')
# def home():
#     return header_text + say_hello() + instructions + footer_text

# @application.route('/<text>')
# def request(text):
#     return header_text + predict(text) + home_link + footer_text

# # add a rule for the index page.
# application.add_url_rule('/', 'index', (lambda: header_text +
#     say_hello() + instructions + footer_text))

# # add a rule when the page is accessed with a name appended to the site
# # URL.
# application.add_url_rule('/<text>', 'hello', (lambda text:
#     header_text + predict(text) + home_link + footer_text))

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()