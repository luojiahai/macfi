# Model-Agnostic Contrastive Interpretation

import numpy as np
from sklearn import svm
import sklearn.model_selection
import sklearn.pipeline
import sklearn.feature_extraction.text
import pprint

import maci_tabular
import interpretation
import utils


def tabular_driver_1():
    # load the dataset
    X = []
    y = []
    f = open('data/square_100.txt', 'r')
    f.readline()
    for line in f:
        splited = line.strip().split('\t')
        X.append([float(x) for x in splited[1:-1]])
        y.append(int(splited[-1]))
    class_names = ['out', 'in']
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

    # fit the model
    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(X_train, y_train)
    # print(clf.score(X_test, y_test))
    
    explainer = maci_tabular.MACITabularExplainer(np.array(X_train))
    intr = explainer.explain(np.array([50,50]), predict_fn=clf.predict_proba)

    print('\nresult: ')
    print('plain instance: ' + str(intr.plain_instance))
    print('plain instance prediction: ' + class_names[clf.predict([intr.plain_instance])[0]])
    print('plain instance predict proba: ' + str(clf.predict_proba([intr.plain_instance])[0]))
    print('counter-factual instance: ' + str(intr.counter_factual_instance))
    print('counter-factual instance prediction: ' + class_names[clf.predict([intr.counter_factual_instance])[0]])
    print('counter-factual instance predict proba: ' + str(clf.predict_proba([intr.counter_factual_instance])[0]))
    print('counter-factual distance: ' + str(intr.counter_factual_distance))
    print('local absolute instance: ' + str(intr.local_absolute_instance))
    print('local absolute instance prediction: ' + class_names[clf.predict([intr.local_absolute_instance])[0]])
    print('local absolute instance predict proba: ' + str(clf.predict_proba([intr.local_absolute_instance])[0]))
    print('local absolute distance: ' + str(intr.local_absolute_distance))

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
    print('\nclass_names:')
    pprint.pprint(dataset.class_names)
    
    explainer = maci_tabular.MACITabularExplainer(np.array(dataset.train), 
                                               feature_names=dataset.feature_names,
                                               categorical_features=dataset.categorical_features,
                                               categorical_names=dataset.categorical_names)
    intr = explainer.explain(dataset.validation[1], predict_fn=clf.predict_proba)

    print('\nresult: ')
    print('plain instance: ' + str(intr.plain_instance))
    print('plain instance prediction: ' + dataset.class_names[clf.predict([intr.plain_instance])[0]])
    print('plain instance predict proba: ' + str(clf.predict_proba([intr.plain_instance])[0]))
    print('counter-factual instance: ' + str(intr.counter_factual_instance))
    print('counter-factual instance prediction: ' + dataset.class_names[clf.predict([intr.counter_factual_instance])[0]])
    print('counter-factual instance predict proba: ' + str(clf.predict_proba([intr.counter_factual_instance])[0]))
    print('counter-factual distance: ' + str(intr.counter_factual_distance))
    print('counter-factual description: ')
    for i in range(len(intr.plain_instance)):
        if (intr.plain_instance[i] != intr.counter_factual_instance[i]):
            print('-- from ' + dataset.categorical_names[i][int(intr.plain_instance[i])] + 
                  ' to ' + dataset.categorical_names[i][int(intr.counter_factual_instance[i])])
    print('local absolute instance: ' + str(intr.local_absolute_instance))
    print('local absolute instance prediction: ' + dataset.class_names[clf.predict([intr.local_absolute_instance])[0]])
    print('local absolute instance predict proba: ' + str(clf.predict_proba([intr.local_absolute_instance])[0]))
    print('local absolute distance: ' + str(intr.local_absolute_distance))

def tabular_driver_3():
    dataset = utils.load_dataset('breast')
    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(dataset.train, dataset.labels_train)
    # pprint.pprint(clf.score(dataset.validation, dataset.labels_validation))

    print('\nfeature_names:')
    pprint.pprint(dataset.feature_names)
    # print('\ncategorical_features"')
    # pprint.pprint(dataset.categorical_features)
    print('\ncategorical_names:')
    pprint.pprint(dataset.categorical_names)
    print('\nclass_names:')
    pprint.pprint(dataset.class_names)
    
    explainer = maci_tabular.MACITabularExplainer(np.array(dataset.train), 
                                            feature_names=dataset.feature_names,
                                            categorical_features=dataset.categorical_features,
                                            categorical_names=dataset.categorical_names)
    intr = explainer.explain(dataset.validation[1], predict_fn=clf.predict_proba)

    print('\nresult: ')
    print('plain instance: ' + str(intr.plain_instance))
    print('plain instance prediction: ' + dataset.class_names[clf.predict([intr.plain_instance])[0]])
    print('plain instance predict proba: ' + str(clf.predict_proba([intr.plain_instance])[0]))
    print('counter-factual instance: ' + str(intr.counter_factual_instance))
    print('counter-factual instance prediction: ' + dataset.class_names[clf.predict([intr.counter_factual_instance])[0]])
    print('counter-factual instance predict proba: ' + str(clf.predict_proba([intr.counter_factual_instance])[0]))
    print('counter-factual distance: ' + str(intr.counter_factual_distance))
    print('counter-factual description: ')
    for i in range(len(intr.plain_instance)):
        if (intr.plain_instance[i] != intr.counter_factual_instance[i]):
            print('-- from ' + dataset.categorical_names[i][int(intr.plain_instance[i])] + 
                  ' to ' + dataset.categorical_names[i][int(intr.counter_factual_instance[i])])
    print('local absolute instance: ' + str(intr.local_absolute_instance))
    print('local absolute instance prediction: ' + dataset.class_names[clf.predict([intr.local_absolute_instance])[0]])
    print('local absolute instance predict proba: ' + str(clf.predict_proba([intr.local_absolute_instance])[0]))
    print('local absolute distance: ' + str(intr.local_absolute_distance))

def main():
    print("Hello, World!")

    tabular_driver_1()
    # tabular_driver_2()
    # tabular_driver_3()


if __name__ == "__main__":
    main()