from __future__ import division
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import nltk
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import numpy as np
import pickle

def loadMovieReviews():
    df = pd.read_csv('labeledTrainData.tsv',header=0, delimiter='\t', quoting=3,escapechar = '\\')
    return df['review'], df['sentiment']

def analyze(clf, vectorizer):
    names = np.asarray(vectorizer.get_feature_names())
    w = np.argsort(clf.coef_.squeeze())

    #the most positives features have big positive values
    print 'Most Positive sentiment words', np.asarray(names)[w[-10:]]
    # the most negatives features have big negative values
    print 'Most Negative sentiment words', np.asarray(names)[w[:10]]
    # Most unusefull words have values around zero
    print 'Most unusefull words', names[np.argsort(np.abs(clf.coef_.squeeze()))[:10]]

def model(revTrain, revTest, sentTrain, sentTest, ngram_range):
    tokenizer = nltk.RegexpTokenizer(r'[a-zA-Z]{2,}')               # extact only words of two characters and more
    vectorizer = TfidfVectorizer(analyzer="word",
                                 # remove HTML tags and convert words to lower case
                                 preprocessor=lambda w: BeautifulSoup(w, 'lxml').get_text().lower(),
                                 tokenizer=tokenizer.tokenize,
                                 stop_words=nltk.corpus.stopwords.words('english'),  # words to be removed
                                 lowercase=False,  # not need it as we already convert them
                                 ngram_range=ngram_range,  # unigram
                                 min_df=2,              #eleminiate words that only apear in one review
                                 max_df=int(80.0 * len(revTrain) / 100), #eleminiate words that apear most of the reviews
                                 )

    matTrain = vectorizer.fit_transform(revTrain)       #convert reviews to matrix if numeric values
    matTest = vectorizer.transform(revTest)             #do the same for test
    clf = LinearSVC()
    clf.fit(matTrain, sentTrain)                        #train the data with linear SVM
    print('Test accuracy {0}'.format(clf.score(matTest, sentTest)))
    analyze(clf, vectorizer)
    with open('svm.pkl', 'wb') as f:
        pickle.dump(clf,f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump((vectorizer.vocabulary_, vectorizer.idf_),f)

def run():
    reviews, sentiments = loadMovieReviews()
    revTrain, revTest, sentTrain, sentTest = train_test_split(reviews, sentiments, test_size=0.15, random_state=0)
    model(revTrain, revTest, sentTrain, sentTest, (1,1)) #uni-gram
    model(revTrain, revTest, sentTrain, sentTest, (1, 2)) #bi-gram


run()