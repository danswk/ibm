import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import re
import bz2
import pickle
from tqdm import tqdm

import os
data_dir = './amazon'
print(os.listdir(data_dir))  # files present in directory

def get_labels_and_texts(file):
    labels = []
    texts = []
    for line in bz2.BZ2File(file):
        d = line.decode('utf-8')  # decode 8-bit encodings of source text to ascii
        labels.append(int(d[9]) - 1)  # extract labels, shifts index [1,2] to [0,1]
        texts.append(d[10:])  # append review content
    return np.array(labels), texts

train_labels, train_texts = get_labels_and_texts('./amazon/train.ft.txt.bz2')
test_labels, test_texts = get_labels_and_texts('./amazon/test.ft.txt.bz2')

def normalize_texts(texts):
    normalized_texts = []
    for text in texts:
        no_cap = text.lower()  # change uppercase to lowercase
        no_pun = re.sub(r'[^\w\s]', '', no_cap)  # remove punctuation
        no_non = re.sub(r'[^\x00-\x7F]', '', no_pun)  # remove non-ascii
        no_spa = no_non.strip()  # remove leading/trailing spaces
        normalized_texts.append(no_spa)
    return normalized_texts
        
train_texts = normalize_texts(train_texts)
test_texts = normalize_texts(test_texts)

print(train_texts[:4])  # texts successfully normalized

num_words = 10000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train_texts)

# train_sequences = tokenizer.texts_to_sequences(train_texts)
# train_pickle = pickle.dumps(train_sequences)
# with open('train_pickle.pkl', 'wb') as file:
#     file.write(train_pickle)

# test_sequences = tokenizer.texts_to_sequences(test_texts)
# test_pickle = pickle.dumps(test_sequences)
# with open('test_pickle.pkl', 'wb') as file:
#     file.write(test_pickle)

with open('train_pickle.pkl', 'rb') as file:
    train_pickle = file.read()
train_sequences = pickle.loads(train_pickle)

print(pd.DataFrame(train_sequences[:4]))  # texts successfully tokenized

max_length = max(len(sequence) for sequence in train_sequences)
print(max_length)

embedding_dim = 100

def build_model():
    sequences = layers.Input(shape=(max_length,))
    embedding = layers.Embedding(input_dim=num_words, output_dim=embedding_dim)(sequences)
    
    x = layers.Conv1D(64, 5, activation='relu')(embedding)  # capture higher-level patterns
    x = layers.MaxPool1D(5)(x)  # reduce dimensionality
    x = layers.Conv1D(64, 3, activation='relu')(x)  # capture more fine-grained patterns
    x = layers.MaxPool1D(3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(.5)(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(.5)(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=sequences, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

model = build_model()
print(model.summary())

with open('test_pickle.pkl', 'rb') as file:
    test_pickle = file.read()
test_sequences = pickle.loads(test_pickle)

train_sequences = pad_sequences(train_sequences, maxlen=max_length)
test_sequences = pad_sequences(test_sequences, maxlen=max_length)

model.fit(
    train_sequences, train_labels,
    batch_size=128, epochs=2, verbose=1,
    validation_data=(test_sequences, test_labels)
)

test_labels_pred = model.predict(test_sequences)
print(f'Accuracy score: {np.round(accuracy_score(test_labels, 1 * (test_labels_pred > 0.5)), 4)}')
print(f'F1 score: {np.round(f1_score(test_labels, 1 * (test_labels_pred > 0.5)), 4)}')
print(f'ROC-AUC score: {np.round(roc_auc_score(test_labels, test_labels_pred), 4)}')