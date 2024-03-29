import numpy as np
import pandas as pd
from pandas import read_csv
import sys
import io
import random

# NLP and text
import html2text
from html2text import html2text
import re
import string
import nltk
from nltk.data import find
import gensim
from gensim.models import Word2Vec
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Machine learning
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, CuDNNLSTM
from keras.optimizers import RMSprop , Adam
from keras.utils.data_utils import get_file


# load ascii text and covert to lowercase
filename = "dataset/all_djt_tweets.csv"
df = pd.read_csv(filename, header=0)
df['text']

df = df.astype({'text': 'str'})
df['text'][0]

"""Since this data is generated from Tweets, we would want to get rid of things like urls, special characters

Src: https://www.kaggle.com/davidg089/all-djtrum-tweets
"""

# cleanup
alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
def cleanup(sentence):

    if(isinstance(sentence, float)):
      return ''

    output = html2text(sentence) 
    # remove retweets and mentions
    output = re.sub("^RT @.*", "", output)

    output = re.sub("^@.*", "", output)
    
    output = output.lower()
    # remove hashtags
    output = re.sub("#\w+$", "", output)
    # remove urls
    urlregex = "https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,}"
    output = re.sub(urlregex, "", output)
    
    # remove special characters 
    output = re.sub("[…“\"”&+,:;=?#$|<>.^*()!–_]", "", output)

    output = re.sub("-", " ", output)
    output = ''.join(filter(lambda x: x in alphabet, output))
    
    # remove twitter handles 
    output = re.sub("@[a-zA-Z0-9]+", " <@twitter_handle> ", output)
    
    # remove numbers and percentages
    output = re.sub("(\d+%)\s+|\s+(\d+%)", " <percentage> ", output)
    output = re.sub("(\d+)\s+|\s+(\d+)", " <number> ", output)

    # remove extra spaces
    output = re.sub("\s+", " ", output).strip()
    # if(output == 'nan'):
    #   print(sentence)
    return  output + ' <eot>' if len(output) > 0 else ''

df['text_clean'] = df['text'].apply(cleanup)
tweets = df['text_clean']

all_text = '.\n'.join(tweets.tolist())
n_first_charactrers = 1000000
text = all_text[0:n_first_charactrers]

text

# https://keras.io/examples/lstm_text_generation/
chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# build the model: a single LSTM
print('Build model...')
model = Sequential()
# use CuDNNLSTM  if you're on GPU
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(Dense(len(chars), activation='softmax'))

#optimizer = RMSprop(lr=0.001, decay=1e-5)
optimizer = Adam(lr=0.001, decay=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

history = model.fit(x, y,
          batch_size=128,
          epochs=120,
          callbacks=[print_callback])

print(history.history.keys())

from IPython.display import SVG
from keras.utils import model_to_dot

SVG(model_to_dot(model, dpi=65).create(prog='dot', format='svg'))

import matplotlib.pyplot as plt

# # Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

