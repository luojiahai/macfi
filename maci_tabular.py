import numpy as np
import sklearn
import sklearn.metrics
import sklearn.preprocessing
import sklearn.utils

import interpretation
from exceptions import MACIError


class MACITabularFinder(object):
    def __init__(self,
                 training_data,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 random_state=None):
        self.random_state = sklearn.utils.check_random_state(random_state)
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)

        if categorical_features is None:
            categorical_features = []
        if feature_names is None:
            feature_names = [str(i) for i in range(training_data.shape[1])]

        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)

    def find_counter_factual_instance(self, 
                                      raw_instance,
                                      predict_fn,
                                      num_samples=5000,
                                      distance_metric='euclidean'):
        inverse = self.perturb(raw_instance, num_samples)
        scaled_data = (inverse - self.scaler.mean_) / self.scaler.scale_

        distances = sklearn.metrics.pairwise_distances(
                scaled_data,
                scaled_data[0].reshape(1, -1),
                metric=distance_metric
        ).ravel()

        yss = predict_fn(inverse)

        ipd = sorted([(i, p, d) for i, p, d 
                     in zip(range(len(yss)), yss, distances) 
                     if (p[0] > p[1]) != (yss[0][0] > yss[0][1])], 
                     key=lambda x:x[2])

        if not ipd:
            raise MACIError("Error: no counter-factual instance is found")

        cfi_index = ipd[0][0]
        intr = interpretation.Interpretation(plain_instance=inverse[0], 
                                                  counter_factual_instance=inverse[cfi_index], 
                                                  pi_predict_proba=yss[0],
                                                  cfi_predict_proba=yss[cfi_index],
                                                  distance=distances[cfi_index])

        #debug
        output_file = open('debug/tabular_out.txt', 'w')
        for i, raw, pb, d in zip(range(len(inverse)), inverse, yss, distances):
            output_file.write(str(i) + '\t' + str(['%.4f' % x for x in raw]) + '\t' + str(['%.4f' % x for x in pb]) + '\t' + str(d) + '\n')

        return intr

    def perturb(self,
                raw_instance,
                num_samples):
        data = np.zeros((num_samples, raw_instance.shape[0]))
        data = self.random_state.normal(
                0, 1, num_samples * raw_instance.shape[0]).reshape(
                num_samples, raw_instance.shape[0])

        data = data * self.scaler.scale_ + raw_instance
        # data = data * self.scaler.scale_ + self.scaler.mean_

        categorical_features = self.categorical_features
        first_row = raw_instance

        data[0] = raw_instance.copy()
        inverse = data.copy()
        for column in categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = self.random_state.choice(values, size=num_samples,
                                                      replace=True, p=freqs)
            binary_column = np.array([1 if x == first_row[column]
                                      else 0 for x in inverse_column])
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        inverse[0] = raw_instance

        return inverse