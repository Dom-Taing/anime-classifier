from flask import Flask
from flask import render_template
from flask import request
import torch
import pandas as pd
from torch.autograd import Variable
import numpy as np
from classifier import model

my_model = model.naive_bayes_model("data/")

# a function that uses the model to predict the genre
def predict(input):
    return my_model.predict(input)

# EB looks for an 'application' callable by default.
application = Flask(__name__)

@application.route('/', methods=['GET', 'POST'])
def synopsis_request():
    if request.method == 'POST':
        genres = predict(request.form["synopsis"])
    else:
        genres = []
    return render_template('index.html', genres=genres)

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()