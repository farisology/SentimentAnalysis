import sys
import sqlite3
import time
import ssl
import urllib
from urlparse import urljoin
from urlparse import urlparse
import re
from datetime import datetime, timedelta
import string
import zlib
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from bs4 import BeautifulSoup
import nltk
import scipy.sparse as sp
import pandas as pd
from collections import OrderedDict
from datetime import date

tweet = list()
location = list()
twDate = list()
time = list()
lan = ('en',)

# Open the raw data retrieved from the network
conn = sqlite3.connect('starwarsL4.sqlite')
conn.text_factory = str
cur1 = conn.cursor()
for row in cur1.execute('SELECT * FROM Tweets WHERE language =?', lan):
    twee = row[1]
    tweet.append(twee)
    
    loc = row[4]
    location.append(loc)
    
    twD = row[5]
    twDate.append(twD)
    
conn = sqlite3.connect('starwarsL5.sqlite')
conn.text_factory = str
cur2 = conn.cursor()
for row1 in cur2.execute('SELECT * FROM Tweets WHERE language =?', lan):
    twee = row[1]
    tweet.append(twee)
    
    loc = row[4]
    location.append(loc)
    
    twD = row[5]
    twDate.append(twD)
    
conn = sqlite3.connect('starwarsL6.sqlite')
conn.text_factory = str
cur3 = conn.cursor()
for row2 in cur3.execute('SELECT * FROM Tweets WHERE language =?', lan):
    twee = row[1]
    tweet.append(twee)
    
    loc = row[4]
    location.append(loc)
    
    twD = row[5]
    twDate.append(twD)
    
conn = sqlite3.connect('starwarsL7.sqlite')
conn.text_factory = str
cur4 = conn.cursor()
for row3 in cur4.execute('SELECT * FROM Tweets WHERE language =?', lan):
    twee = row[1]
    tweet.append(twee)
    
    loc = row[4]
    location.append(loc)
    
    twD = row[5]
    twDate.append(twD)
    
conn = sqlite3.connect('starwarsL8.sqlite')
conn.text_factory = str
cur5 = conn.cursor()
for row4 in cur5.execute('SELECT * FROM Tweets WHERE language =?', lan):
    twee = row[1]
    tweet.append(twee)
    
    loc = row[4]
    location.append(loc)
    
    twD = row[5]
    twDate.append(twD)
    
for td in twDate:
    dateTime = td.split()
    ti = dateTime[3].split(':')
    tim = ti[0]+ti[1]
    time.append(tim)
    
for i in range(len(time)):
    time[i] = int(time[i])
    
print "done", tweet[0], " - ", location[0], " - ", twDate[0], " - ", time[0]
len(time)

class MyVectorizer(TfidfVectorizer):
    '''
    it is dummy class to workaround sklearn limitation
    '''
    def setIDF(self, idf):
        TfidfVectorizer.idf_ =idf

class SentimentAnalyzer:
    def __init__(self):
        with open('svm.pkl','rb') as f:
            self.clf = pickle.load(f)

        tokenizer = nltk.RegexpTokenizer(r'[a-zA-Z]{2,}')  # extact only words of two characters and more
        self.vectorizer = MyVectorizer(analyzer="word",
                                 # remove HTML tags and convert words to lower case
                                 preprocessor=lambda w: BeautifulSoup(w, 'lxml').get_text().lower(),
                                 tokenizer=tokenizer.tokenize,
                                 stop_words=nltk.corpus.stopwords.words('english'),  # words to be removed
                                 lowercase=False,  # not need it as we already convert them
                                 ngram_range=(1, 2),  # unigram
                                 )
        with open('vectorizer.pkl','rb') as f:
            voc, idf = pickle.load(f)
            self.vectorizer.vocabulary_ = voc
            self.vectorizer.setIDF(idf)
            self.vectorizer._tfidf._idf_diag = sp.spdiags(idf,      #again this to workaround sklearn limitation
                                                     diags=0,
                                                     m=len(idf),
                                                     n=len(idf))

    def analyzes(self, sentences):
        '''
        analyze more than one sentece
        :return: the sentiment 1=positive, 0=negative
        :param sentences: array of sentences
        '''
        return self.clf.predict(self.vectorizer.transform(sentences))

    def analyze(self,sentence):
        '''
                analyze just one sentece
                :return: the sentiment 1=positive, 0=negative
                :param sentence: a string
        '''
        return self.analyzes([sentence])[0]
    
res = SentimentAnalyzer().analyzes(tweet)

CentralDataTW = dict()
CentralDataTW["Tweets"] = tweet
CentralDataTW["Locations"] = location
CentralDataTW["Time&date"] = twDate
CentralDataTW["Sentiments"] = res
CentralDataTW["Time"] = time

df = pd.DataFrame(CentralDataTW)