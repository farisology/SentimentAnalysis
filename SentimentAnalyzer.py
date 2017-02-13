import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from bs4 import BeautifulSoup
import nltk
import scipy.sparse as sp

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

    def test(self):
        from sklearn.model_selection import train_test_split
        reviews, sentiments = loadMovieReviews()
        revTrain, revTest, sentTrain, sentTest = train_test_split(reviews, sentiments, test_size=0.15, random_state=0)
        print (self.analyzes(revTest)==sentTest).mean()*100     #if everythis is ok then it should print 90.69

def loadMovieReviews():
    import pandas as pd
    df = pd.read_csv('labeledTrainData.tsv', header=0, delimiter='\t', quoting=3, escapechar='\\')
    return df['review'], df['sentiment']


if __name__ == '__main__':
    import sys
    analyzer = SentimentAnalyzer()
    try:
        while True:
            data = raw_input()
            if analyzer.analyze(data):
                print 'Positive'
            else:
                print 'Negative'
            sys.stdout.flush()
    except Exception as e:
        print e

