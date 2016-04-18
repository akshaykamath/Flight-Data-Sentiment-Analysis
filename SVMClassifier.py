import csv
author = 'Akshay'

#########################################################
import sys

reload(sys)
sys.setdefaultencoding('utf8')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier


def load_csv():
    with open('Tweets.csv', 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        count = 1

        reviews = []
        stars = []

        for row in reader:
            try:
                reviews.append(row['text'])
                stars.append(row['airline_sentiment'])
            except:
                continue

    return reviews, stars


def pre_process(data, target):
    count_vectorized = CountVectorizer(binary='false', ngram_range=(0, 1))

    data = count_vectorized.fit_transform(data)
    tfidf_data = TfidfTransformer(use_idf=True, smooth_idf=True).fit_transform(data)
    print "Calculating term frequency."
    return tfidf_data


def learn_model(reviews, stars):
    classifier = OneVsRestClassifier(SVC(C=1, kernel='linear', gamma=1, verbose=False, probability=False))
    # classifier = MultinomialNB()
    print "-" * 60, "\n"

    print "Results with 10-fold cross validation:\n"
    print "-" * 60, "\n"

    predicted = cross_validation.cross_val_predict(classifier, reviews, stars, cv=4, n_jobs=1)
    print 'predicted:', len(predicted)
    print 'total:', len(stars)
    calculate_majority_class(stars)
    print "*" * 20
    print "\t Accuracy Score\t", metrics.accuracy_score(stars, predicted)
    print "*" * 20

    # print "Precision Score\t", metrics.precision_score(stars, predicted)
    # print "Recall Score\t", metrics.recall_score(stars, predicted)
    print "\nClassification Report:\n\n", metrics.classification_report(stars, predicted)

    # calculate_majority_class(target_train)
    # print "\nConfusion Matrix:\n\n", metrics.confusion_matrix(stars, predicted)


def calculate_majority_class(data):
    counter = Counter(data)
    # print counter
    print "Majority Class : " + str((max(counter.values()) * 100) / sum(counter.values())) + "%"


def evaluate_model(target_true, target_predicted):
    print classification_report(target_true, target_predicted)
    print "The accuracy score is {:.2%}".format(accuracy_score(target_true, target_predicted))


def main():
    print ("Loading file and getting reviews..")
    data, target = load_csv()
    print "Get All Data."
    tf_idf = pre_process(data, target)
    learn_model(tf_idf, target)


main()
