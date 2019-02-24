# Model-Agnostic Counter-Factual Instance

import numpy as np
from sklearn import svm
import sklearn.model_selection

import macfi_tabular
import instance

def main():
    print("Hello, World!")

    X = []
    y = []
    f = open('data/square_100.txt', 'r')
    f.readline()
    for line in f:
        splited = line.strip().split('\t')
        X.append([float(x) for x in splited[1:-1]])
        y.append(int(splited[-1]))

    clf = svm.SVC(gamma='scale', probability=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    # print(clf.score(X_test, y_test))
    
    finder = macfi_tabular.MACFITabularFinder(X_train)
    inst = finder.find_counter_factual_instance(np.array([50,50]), predict_fn=clf.predict_proba)

    print('plain instance: ' + str(inst.plain_instance))
    print('counter-factual instance: ' + str(inst.counter_factual_instance))
    print('plain instance predict proba: ' + str(inst.pi_predict_proba))
    print('counter-factual instance predict proba: ' + str(inst.cfi_predict_proba))
    print('distance: ' + str(inst.distance))

if __name__ == "__main__":
    main()