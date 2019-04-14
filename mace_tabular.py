import numpy as np
import sklearn
import sklearn.metrics
import sklearn.preprocessing
import sklearn.utils
import collections

import explanation
from exceptions import MACEError


class MACETabularExplainer(object):
    def __init__(self,
                 training_data,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 random_state=None):
        self.random_state = sklearn.utils.check_random_state(random_state)

        if categorical_features is None:
            categorical_features = []
        if feature_names is None:
            feature_names = [str(i) for i in range(training_data.shape[1])]

        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)

        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)
        self.feature_values = {}
        self.feature_frequencies = {}

        for feature in self.categorical_features:
            column = training_data[:, feature]

            feature_count = collections.Counter(column)
            values, frequencies = map(list, zip(*(feature_count.items())))

            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 float(sum(frequencies)))
            self.scaler.mean_[feature] = 0
            self.scaler.scale_[feature] = 1

    def explain(self, 
                raw_instance,
                predict_fn,
                num_samples=5000,
                distance_metric='euclidean'):
        data, inverse = self._perturb(raw_instance, num_samples)
        scaled_data = (data - self.scaler.mean_) / self.scaler.scale_

        distances = self._distance(data=scaled_data,
                                   instance=scaled_data[0], 
                                   distance_metric=distance_metric)
        yss = predict_fn(inverse)

        counter_factual_list = list(filter(lambda x: (x[1][0] > x[1][1]) != (yss[0][0] > yss[0][1]), 
                                           zip(range(len(inverse)), yss, distances)))
        counter_factual_list = sorted(counter_factual_list, key=lambda x: x[2])
        if not counter_factual_list:
            raise MACEError("MACEError: no counter-factual instance is found")
        counter_factual_instance = inverse[counter_factual_list[0][0]]
        counter_factual_distance = counter_factual_list[0][2]

        local_absolute_list = list(filter(lambda x: (x[1][0] > x[1][1]) == (yss[0][0] > yss[0][1]), 
                                          zip(range(len(inverse)), yss)))
        local_absolute_instance = None
        local_absolute_distance = -1
        for i, _  in local_absolute_list:
            distances_ = self._distance(data=scaled_data, 
                                        instance=scaled_data[i], 
                                        distance_metric=distance_metric)
            counter_factual_list_ = [(j, distances_[j]) for j, _, _ in counter_factual_list]
            counter_factual_list_ = sorted(counter_factual_list_, key=lambda x: x[1])
            counter_factual_instance_ = inverse[i]
            counter_factual_distance_ = counter_factual_list_[0][1]
            if (counter_factual_distance_ > local_absolute_distance):
                local_absolute_distance = counter_factual_distance_
                local_absolute_instance = counter_factual_instance_

        expl = explanation.Explanation(plain_instance=inverse[0], 
                                       counter_factual_instance=counter_factual_instance,
                                       counter_factual_distance=counter_factual_distance,
                                       local_absolute_instance=local_absolute_instance,
                                       local_absolute_distance=local_absolute_distance)

        #debug
        output_file = open('debug/tabular_out.txt', 'w')
        for i, raw, pb, d in zip(range(len(inverse)), inverse, yss, distances):
            output_file.write(str(i) + '\t' + str(['%.4f' % x for x in raw]) + '\t' + str(['%.4f' % x for x in pb]) + '\t' + str(d) + '\n')

        return expl

    def _distance(self,data, instance, distance_metric):
        return sklearn.metrics.pairwise_distances(data, instance.reshape(1,-1),
                                                  metric=distance_metric).ravel()
    
    def _perturb(self, raw_instance, num_samples):
        data = np.zeros((num_samples, raw_instance.shape[0]))
        data = self.random_state.normal(
                   0, 1, num_samples * raw_instance.shape[0]
               ).reshape(num_samples, raw_instance.shape[0])

        data = data * self.scaler.scale_ + raw_instance     # sample around instance x
        # data = data * self.scaler.scale_ + self.scaler.mean_

        categorical_features = self.categorical_features
        first_row = raw_instance

        data[0] = raw_instance.copy()
        inverse = data.copy()
        for column in categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = self.random_state.choice(values, size=num_samples, replace=True, p=freqs)
            binary_column = np.array([1 if x == first_row[column] else 0 for x in inverse_column])
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        inverse[0] = raw_instance

        return data, inverse