# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:59:35 2022

@author: DELL
"""

#import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
import nltk
nltk.download('stopwords')

words_stopped = stopwords.words('english')


news_dataset = pd.read_csv("train.csv")


news_dataset = news_dataset.fillna("")

news_dataset['content'] = news_dataset['author'] + news_dataset['title']

content_values = news_dataset['content']

x = news_dataset.drop(columns = 'label', axis = 1)
y = news_dataset['label']


port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content




news_dataset['content'] = news_dataset['content'].apply(stemming)



x = news_dataset['content'].values
y = news_dataset['label'].values



vectorizer = TfidfVectorizer()
vectorizer.fit(x)

x = vectorizer.transform(x)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = 2)


model = LogisticRegression()
model.fit(x_train, y_train)


x_train_prediction = model.predict(x_train)
x_train_accuracy = accuracy_score(x_train_prediction, y_train)



x_test_prediction = model.predict(x_test)
x_test_accuracy = accuracy_score(x_test_prediction, y_test)



x_new = x_test[0]
prediction = model.predict(x_new)



 
 
 