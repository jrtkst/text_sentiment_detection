# Packages
import pandas as pd
from datetime import datetime         
import re
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
import torch
import torch.nn as nn
import pickle

from flask import Flask, request, render_template, url_for


# Neural Network Class
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.LSTM = nn.LSTM(vocab_size*embedding_dim, hidden_size)
        self.FC = nn.Linear(hidden_size, 2)
        self.Softmax = nn.Softmax(dim=1)
        self.h = torch.zeros(1, self.hidden_size, requires_grad=True)
        self.c = torch.zeros(1, self.hidden_size, requires_grad=True)
    def forward(self, x):
        # Forward propagation
        h0, c0 = self.h, self.c
        x = self.embedding(x)
        x = x.flatten(start_dim=1)
        x, (hn, cn) = self.LSTM(x, (h0.detach(), c0.detach()))
        self.h, self.c = hn, cn
        # Output reshaping
        x = x.reshape(-1, self.hidden_size)
        x = self.FC(x)
        x = self.Softmax(x)
        return x
embedding_dim = 5
hidden_size = 256
vocab_size = 13042
LSTM_net = LSTM(vocab_size, embedding_dim, hidden_size)

# Word Processor
def preprocessing(text):
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    rev = re.sub("@[A-Za-z0-9_]+", " ", text)
    rev = re.sub("[^a-zA-Z]", " ", rev)
    rev = rev.lower()
    rev = rev.split()
    rev = [ps.stem(word) for word in rev if not word in set(all_stopwords) if len(word) > 2]
    rev = ' '.join(rev)
    return rev

# Load model
cv = pickle.load(open('PATH_TO cv_model.sav', 'rb'))
model_load = torch.load('PATH_TO LSTM_sentiment_model.pth')
LSTM_net.load_state_dict(model_load['model_state_dict'])

# Inference
def inference(text_input):
    tic = datetime.now()
    pre_text = preprocessing(text_input)
    pre_text = cv.transform(pd.Series(pre_text)).toarray()
    out = LSTM_net.forward(torch.LongTensor(pre_text))
    cls = torch.argmax(out)
    toc = datetime.now()
    if cls.item() == 0:
        return ("Negative", out.max().item(), (toc - tic).total_seconds())
    else:
        return ("Positive/Neutral", out.max().item(), (toc - tic).total_seconds())

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def infer():
    output = False
    if request.method == "POST":
        raw_input = request.form.get("text_input")
        output = inference(raw_input)
        return render_template('index.html', raw_input = raw_input, result = output, state = True)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host = "127.0.0.1", port = 5000, debug = True)
