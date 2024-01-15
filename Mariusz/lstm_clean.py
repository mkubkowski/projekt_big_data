import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from gensim.corpora import Dictionary
import string
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import LSTM
from sklearn.metrics import roc_auc_score

#from tensorflow.keras.datasets import imdb

path_sample = "G:/dane_python/big_data_projekt/sample.json/"
path_data = path_sample + "sample.json"


with open(path_data) as f:
    json_test = json.load(f)


df_pd = pd.DataFrame(json_test)


punc = string.punctuation

def wt(string):
    return [w.lower() for w in word_tokenize(string)
                if w.lower() not in punc]

review_detail_clear = df_pd['review_detail'].apply(wt)
dct = Dictionary(review_detail_clear)


tokens_in_dataset = 1000
document_max_len = 400
batch_size = 32
embedding_dim = 50

# w ponizszym kroku kodujemy tokeny za pomoca slownika, a nastepnie
# liczymy hashe (w uproszczony sposob), by ograniczyc wielkosc slownika do
# embeddingu
review_detail_idx = review_detail_clear.apply(lambda x: [y%tokens_in_dataset 
                                                         for y in dct.doc2idx(x)])

target = np.array(df_pd['spoiler_tag'].copy())

x_train, x_test, y_train, y_test = train_test_split(review_detail_idx,
                                                    target, test_size = 0.3,
                                                    random_state=12)

plt.hist([len(x_train.iloc[i]) for i in range(1000)])

x_train = sequence.pad_sequences(x_train, maxlen=document_max_len, padding='pre', truncating='post')
x_test = sequence.pad_sequences(x_test, maxlen=document_max_len, padding='pre', truncating='post')


model = Sequential()

model.add(Embedding(input_dim=tokens_in_dataset, # liczba unikalnych tokenów
                    output_dim=embedding_dim, # wielkość embeddingu
                    input_length=document_max_len, # długość sekwencji
                    ))
model.add(LSTM(units=32, activation='tanh', return_sequences=False))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

es = EarlyStopping(patience=3, monitor='val_loss')
model.summary()
model.fit(x_train, y_train, batch_size=batch_size,
          callbacks=[es],
          validation_split=0.2,
          epochs=20)

model.evaluate(x_test,y_test)

prob_pred_train = model.predict(x_train)
prob_pred_test = model.predict(x_test)
AUC = {
       'train': roc_auc_score(y_train,prob_pred_train),
       'test': roc_auc_score(y_test,prob_pred_test),
       }

