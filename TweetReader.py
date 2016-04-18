import codecs

__author__ = 'Akshay'
import re
from nltk.corpus import stopwords
import json
import sys
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation, datasets, linear_model
reload(sys)
sys.setdefaultencoding('utf8')
import nltk
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import csv


def load_csv():
    count = 0
    with open('Tweets.csv', 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        data_tuples = []

        tknzr = TweetTokenizer()
        for row in reader:
            count += 1
            try:

                tup = (row['airline_sentiment'], row['text'])
                print tup
                words = tknzr.tokenize(row['text'])
                print len(words)

                data_tuples.append(tup)
            except:
                continue

    return data_tuples

def main():
   # load_csv()
    data = datasets.load_diabetes()
    for d in data.data:
        print d
        break

    for t in data.target:
        print t
        break


main()
