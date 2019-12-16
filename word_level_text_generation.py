## please install html2text
#  pip install html2text #ONLY if html2text not installed

import numpy as np
import pandas as pd
import sys
import io
import random

# NLP and text
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
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file

nltk.download('punkt')


# load ascii text and covert to lowercase
filename = "dataset/all_djt_tweets.csv"
df = pd.read_csv(filename, header=0)
df['text']


"""Since this data is generated from YouTube subtitles, we would want to get rid of things like special characters

Src: https://www.kaggle.com/binksbiz/sentiment-analysis
"""



df.loc[df['text'] == None]

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


df['clean_tweets'] = df['text'].apply(cleanup)

len(df['clean_tweets'])

clean_tweets = df.loc[df['clean_tweets'] != '']

clean_tweets = clean_tweets.astype(str)['clean_tweets']

len(clean_tweets)

tweets = clean_tweets.astype(str).tolist()

all_tweets = ' '.join(tweets)
corpus = all_tweets
len(all_tweets)

# nltk.download('word2vec_sample')
# word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
# embedding = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)

# AVERAGE WORD COUNT IN TWEETS 
all_sentences = []
lengths = []
for tweet in tweets:
  lengths.append(len(tweet))
  sentence = tweet.split(' ')
  all_sentences.append(sentence)

np.array(lengths).mean()

embedding = Word2Vec(all_sentences,  size=100, min_count=1)  # word_model=gensim.models.Word2Vec(sentences, size=200, min_count=1, window=5)

embdim = embedding.wv['the'].size

words = corpus.split(' ')

# https://keras.io/examples/lstm_text_generation/
# cut the text in semi-redundant sequences of seqlen words
seqlen = 20
step = 5
sentences = []
next_word = []
for i in range(0, len(words) - seqlen, step):
    sentences.append(words[i: i + seqlen])
    next_word.append(words[i + seqlen])

sentences[0][:100]

print('nb sequences:', len(sentences))
print('Vectorization...')
x = np.zeros((len(sentences), seqlen, embdim))
y = np.zeros((len(sentences), embdim))

for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        x[i, t,:] = embedding.wv[word]

for i, next_word in enumerate(next_word):
    y[i, :] = embedding.wv[next_word]

x.shape

# build the model: a single LSTM
print('Build model...')
model = Sequential()
# model.add(CuDNNLSTM(128, return_sequences=True, input_shape=(seqlen, embdim)))
# model.add(Dropout(0.2))
# please use CuDNNLSTM if you're using GPU
model.add(LSTM(250))
model.add(Dropout(0.2))
model.add(Dense(250, activation='relu'))
model.add(Dense(embdim, activation='linear'))

optimizer = Adam(lr=0.01, decay=1e-3)
model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])


def sample(similars, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = [elt[1] for elt in similars]

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    argm = np.argmax(probas)
    return similars[argm][0]

def vec2word(vec, temperature=1.0):
    similars = embedding.most_similar(positive=[vec], topn=10)
    return sample(similars, temperature)

def arr2sent(arr):
    return TreebankWordDetokenizer().detokenize(arr)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(all_sentences))
    sentence = all_sentences[start_index][0:10]
    print(sentence)
    generated = sentence
    
    print('----- Generating with seed: "' + arr2sent(sentence) + '"')
    for i in range(50):
        x_pred = np.zeros((1, seqlen, embdim))
        for t, word in enumerate(sentence):
            x_pred[0, t, :] = embedding.wv[word]

        vec = model.predict(x_pred, verbose=0)[0]
        next_word = vec2word(vec, 0.5)

        sentence.append(next_word)
        sentence = sentence[1:]
        generated.append(next_word)
    print(arr2sent(generated))

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

#batch_size=128
history = model.fit(x, y,
          batch_size=200, 
          epochs=120,
          callbacks=[print_callback])

import matplotlib.pyplot as plt

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

