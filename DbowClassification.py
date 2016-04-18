
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
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
from nltk.tokenize import TweetTokenizer
import csv
from gensim.models.doc2vec import LabeledSentence
from sklearn.linear_model import LogisticRegression

def load_csv():
    with open('Tweets.csv', 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        count = 1

        reviews = []
        stars = []
        tknzr = TweetTokenizer()
        for row in reader:
            try:
                words=tknzr.tokenize(row['text'])
                label = 'SENT_%s' % count

                #print label
               # TaggedDocument(utils.to_unicode(row['text']).split(), [label])
                # print "label:", label
                #labels = [label]
                #lab_sent = LabeledSentence(words, label)
                #print lab_sent
                #reviews.append(TaggedDocument(utils.to_unicode(row['text']).split(), [label]))
                reviews.append(TaggedDocument(words, [label]))
                stars.append(row['airline_sentiment'])
                count += 1
            except:
                continue

    print "final count:", count
    return reviews, stars


def pre_process(data):
    count_vectorized = CountVectorizer(binary='false', ngram_range=(0, 1))

    data = count_vectorized.fit_transform(data)
    print "data", data[0]
    tfidf_data = TfidfTransformer(use_idf=True, smooth_idf=True).fit_transform(data)
    print "Calculating term frequency."
    return tfidf_data


def learn_model(reviews, stars):
    #classifier = OneVsRestClassifier(SVC(C=1, kernel='linear', gamma=1, verbose=False, probability=False))
    #classifier = MultinomialNB()\
    classifier = SGDClassifier()
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


def doc_to_vecs(data):
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=2)
    model.build_vocab(data)

    for epoch in range(10):
        model.train(data)

    model.save('./dbowtweets.d2v')
   # model = Doc2Vec.load('./dbowtweets.d2v')
    #print model.docvecs['SENT_14640']

    print "*****Models saved and loaded****"
    return data


def load_vecs():
    model = Doc2Vec.load('./dbowtweets.d2v')
    train_arrays = numpy.zeros((14640, 100))
    label = ""
    count = 0

    for i in range(1, 14641):
        label = 'SENT_' + str(i)
        train_arrays[count] = model.docvecs[label]
        count += 1

    return train_arrays


def main():
    print ("Loading file and getting reviews..")
    data, target = load_csv()
    #print "Get All Data."
    #print data[0]
    #data = doc_to_vecs(data)
    #count = 1

    bata = load_vecs()
    #tf_idf = pre_process(bata)
    #tfidf_data = TfidfTransformer(use_idf=True, smooth_idf=True).fit_transform(data)
    learn_model(bata, target)

main()