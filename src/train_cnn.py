from sklearn.feature_extraction.text import CountVectorizer
from joblib import load, dump

import pandas as pd
import numpy as np

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Conv1D,MaxPooling1D,Flatten
from keras.optimizers import Adam
from keras.models import Sequential, load_model

import json

df = pd.read_csv("../data/train.tsv", delimiter='\t')

text = df["Phrase"].values[:10000]

vectorizer = CountVectorizer()
vectorizer.fit(text)
dump(vectorizer, '../static/count_vectorizer.joblib') 
X = vectorizer.fit_transform(text).toarray()
print("PICKLED!")
X = np.expand_dims(X, axis=2)
print(X.shape)

y = df["Sentiment"].values[:10000]
y = to_categorical(y)


X_train, X_test, y_train, y_test = train_test_split(X, y)

input_dim = len(X_train[0])
model = Sequential()
model.add(Conv1D(5,10, activation="relu", input_shape=(input_dim,1)))
model.add(MaxPooling1D(5))
model.add(Flatten())
model.add(Dense(10, activation="tanh"))
model.add(Dense(10, activation="tanh"))
model.add(Dense(5, activation="relu"))


adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss="mse", metrics=["acc"])
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2)

model.save('../static/feedforward_rt_sent.h5') 

with open("../static/training_hist.json", "w") as fp:
	json.dump(history.history, fp)

del model

