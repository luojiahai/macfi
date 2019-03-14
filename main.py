# Model-Agnostic Contrastive Interpretation

import numpy as np
from sklearn import svm
import sklearn.model_selection
import sklearn.pipeline
import sklearn.feature_extraction.text
import pprint

import maci_tabular
import maci_text
import interpretation
import utils


def tabular_driver_1():
    X = []
    y = []
    f = open('data/square_100.txt', 'r')
    f.readline()
    for line in f:
        splited = line.strip().split('\t')
        X.append([float(x) for x in splited[1:-1]])
        y.append(int(splited[-1]))

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(X_train, y_train)
    # print(clf.score(X_test, y_test))
    
    finder = maci_tabular.MACITabularFinder(np.array(X_train))
    intr = finder.find_counter_factual_instance(np.array([50,50]), predict_fn=clf.predict_proba)

    print('\nresult: ')
    print('plain instance: ' + str(intr.plain_instance))
    print('counter-factual instance: ' + str(intr.counter_factual_instance))
    print('plain instance predict proba: ' + str(intr.pi_predict_proba))
    print('counter-factual instance predict proba: ' + str(intr.cfi_predict_proba))
    print('distance: ' + str(intr.distance))

def tabular_driver_2():
    dataset = utils.load_dataset('loan')
    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(dataset.train, dataset.labels_train)
    # pprint.pprint(clf.score(dataset.validation, dataset.labels_validation))

    print('\nfeature_names:')
    pprint.pprint(dataset.feature_names)
    # print('\ncategorical_features"')
    # pprint.pprint(dataset.categorical_features)
    print('\ncategorical_names:')
    pprint.pprint(dataset.categorical_names)
    
    finder = maci_tabular.MACITabularFinder(np.array(dataset.train), 
                                            feature_names=dataset.feature_names,
                                            categorical_features=dataset.categorical_features,
                                            categorical_names=dataset.categorical_names)
    intr = finder.find_counter_factual_instance(dataset.validation[0], predict_fn=clf.predict_proba)

    print('\nresult: ')
    print('plain instance: ' + str(intr.plain_instance))
    print('counter-factual instance: ' + str(intr.counter_factual_instance))
    print('plain instance predict proba: ' + str(intr.pi_predict_proba))
    print('counter-factual instance predict proba: ' + str(intr.cfi_predict_proba))
    print('distance: ' + str(intr.distance))
    print('counter-factual description: ')
    for i in range(len(intr.plain_instance)):
        if (intr.plain_instance[i] != intr.counter_factual_instance[i]):
            print('-- from ' + dataset.categorical_names[i][int(intr.plain_instance[i])] + 
                  ' to ' + dataset.categorical_names[i][int(intr.counter_factual_instance[i])])

def text_driver():
    X = []
    y = []
    f = open('data/SMSSpamCollection', 'r', encoding='utf-8')
    for line in f:
        splited = line.strip().split('\t')
        X.append(splited[1])
        if (splited[0] == 'ham'):
            y.append(0)
        elif (splited[0] == 'spam'):
            y.append(1)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

    clf = sklearn.svm.SVC(kernel='linear', C=1.0, probability=True)
    pl = sklearn.pipeline.Pipeline([
        ('vect', sklearn.feature_extraction.text.CountVectorizer(lowercase=False)),
        ('tfidf', sklearn.feature_extraction.text.TfidfTransformer()),
        ('clf', clf),
    ])
    pl.fit(X_train, y_train)
    # print(pl.score(X_test, y_test))

    finder = maci_text.MACITextFinder()
    intr = finder.find_counter_factual_instance(X_test[0], predict_fn=pl.predict_proba)

    print('\nresult: ')
    print('plain instance: ' + str(intr.plain_instance.encode('utf-8')))
    print('counter-factual instance: ' + str(intr.counter_factual_instance.encode('utf-8')))
    print('plain instance predict proba: ' + str(intr.pi_predict_proba))
    print('counter-factual instance predict proba: ' + str(intr.cfi_predict_proba))
    print('distance: ' + str(intr.distance))

def main():
    print("Hello, World!")

    # tabular_driver_1()
    tabular_driver_2()
    # text_driver()


if __name__ == "__main__":
    main()