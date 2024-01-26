
import io
import json
import pandas as pd
import numpy as np
import string
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from keras.layers import LSTM, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping





nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))



df = pd.read_json('C:\\Users\\beata\\Downloads\\Projektowanie_BD\\sample.json')


#PREPROCESSING - cleaning
#1
df['review_detail_clean'] = df['review_detail'].str.lower()

#2
contractions_df = pd.read_csv('C:\\Users\\beata\\Downloads\\contractions.csv')

contractions_df['Contraction'] = contractions_df['Contraction'].str.lower()
contractions_df['Meaning'] = contractions_df['Meaning'].str.lower()
contractions_dict = dict(zip(contractions_df.Contraction, contractions_df.Meaning))

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def expand_contractions(s, contractions_dict=contractions_dict):
  def replace(match):
    return contractions_dict[match.group(0)]
  return contractions_re.sub(replace, s)

df['review_detail_clean1'] = df['review_detail_clean'].apply(expand_contractions)

#3
def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_urls(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def remove_emoticons_and_symbols(text):
    # emotikony
    text_without_emoticons = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+', '', text)
    # symbole
    text_without_symbols = re.sub(r'[^\w\s]', '', text_without_emoticons)
    return text_without_symbols

#df['review_detail_clean2'] = df['review_detail_clean1'].apply(remove_numbers)
df['review_detail_clean3'] = df['review_detail_clean'].apply(remove_urls)
df['review_detail_clean4'] = df['review_detail_clean3'].apply(remove_emoticons_and_symbols)

#4
def remove_punctuation_text(text):
  return text.translate(str.maketrans("", "", string.punctuation))

df['review_detail_clean5'] = df['review_detail_clean4'].apply(remove_punctuation_text)

#5
df['review_detail_clean6'] = df['review_detail_clean5'].apply(word_tokenize)

def remove_stopwords_text(tokens):
  return [word for word in tokens if word not in stop_words]

#not used
#df['review_detail_clean7'] = df['review_detail_clean6'].apply(remove_stopwords_text)
#df.head(3)


#PARAMS
tokens_in_dataset = 5000
document_max_len = 400
batch_size = 32
embedding_dim = 64


#PREPROCESSING - tokenizacja
tokenizer = Tokenizer(num_words=tokens_in_dataset)
tokenizer.fit_on_texts(df['review_detail_clean6'])

sequences = tokenizer.texts_to_sequences(df['review_detail_clean6'])
vocab_size = len(tokenizer.word_index) + 1
max_sequence_length = max(len(seq) for seq in sequences)



#MODEL - input
out = np.array(df['spoiler_tag'])

x_train, x_test, y_train, y_test = train_test_split(sequences,
                                                    out, test_size = 0.2,
                                                    random_state = 42)


x_train = sequence.pad_sequences(x_train, maxlen=max_sequence_length, padding='pre', truncating='post')
x_test = sequence.pad_sequences(x_test, maxlen=max_sequence_length, padding='pre', truncating='post')

#MODEL - create
model = Sequential()
model.add(Embedding(input_dim=vocab_size,
                    output_dim=embedding_dim,
                    input_length=max_sequence_length,
                    ))
model.add(SpatialDropout1D(0.1))
model.add(Bidirectional(LSTM(units=32, activation='tanh', return_sequences=True)))
model.add(Bidirectional(LSTM(units=16, activation='tanh', return_sequences=False)))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

es = EarlyStopping(patience=3, monitor='val_loss')
model.summary()


#MODEL - fit
model.fit(x_train, y_train, batch_size=batch_size,
          callbacks=[es],
          validation_split=0.3,
          epochs=15)


#MODEL - evaluate
#1
model.evaluate(x_test,y_test)

#2
prob_pred_train = model.predict(x_train)
prob_pred_test = model.predict(x_test)
AUC = {
       'train': roc_auc_score(y_train,prob_pred_train),
       'test': roc_auc_score(y_test,prob_pred_test),
       }

print(AUC)


#3
y_pred = model.predict(x_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Krzywa ROC (obszar pod krzywą = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Krzywa ROC dla modelu LSTM')
plt.legend(loc='lower right')
plt.show()

#4

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
f1 = 2 * (precision * recall) / (precision + recall)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, recall[:-1], label='Recall', marker='o', color='blue')
plt.plot(thresholds, precision[:-1], label='Precision', marker='o', color='green')
plt.plot(thresholds, f1[:-1], label='F1', marker='o', color='orange')

plt.xlabel('Threshold')
plt.legend()

plt.grid(axis='y', linestyle='--', alpha=0.2)
plt.show()


f1_scores = [f1_score(y_test, (y_pred > t).astype(int)) for t in thresholds]
best_threshold_idx = f1_scores.index(max(f1_scores))
best_threshold = thresholds[best_threshold_idx]


y_pred_best = (y_pred > best_threshold).astype(int)


cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negatywna", "Pozytywna"], yticklabels=["Negatywna", "Pozytywna"])
plt.xlabel('Przewidziana klasa')
plt.ylabel('Rzeczywista klasa')
plt.title('Macierz pomyłek dla modelu LSTM')
plt.show()


precision = precision_score(y_test, y_pred_best)
recall = recall_score(y_test, y_pred_best)
f1 = f1_score(y_test, y_pred_best)

precision, recall, f1
