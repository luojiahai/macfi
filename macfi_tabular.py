import numpy as np
import sklearn
import sklearn.metrics
import sklearn.preprocessing
import sklearn.utils

import instance
from exceptions import MACFIError

class MACFITabularFinder(object):
    def __init__(self,
                 training_data,
                 random_state=None):
        self.random_state = sklearn.utils.check_random_state(random_state)
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)

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
            raise MACFIError("Error: no counter-factual instance is found")

        cfi_index = ipd[0][0]
        inst = instance.MACFIInstance(plain_instance=inverse[0], 
                                      counter_factual_instance=inverse[cfi_index], 
                                      pi_predict_proba=yss[0],
                                      cfi_predict_proba=yss[cfi_index],
                                      distance=distances[cfi_index])

        #debug
        output_file = open('debug/tabular_out.txt', 'w')
        for i, raw, pb, d in zip(range(len(inverse)), inverse, yss, distances):
            output_file.write(str(i) + '\t' + str(['%.4f' % x for x in raw]) + '\t' + str(['%.4f' % x for x in pb]) + '\t' + str(d) + '\n')

        return inst

    def perturb(self,
                raw_instance,
                num_samples):
        inverse = np.zeros((num_samples, raw_instance.shape[0]))
        inverse = self.random_state.normal(
                0, 1, num_samples * raw_instance.shape[0]).reshape(
                num_samples, raw_instance.shape[0])

        inverse = inverse * self.scaler.scale_ + raw_instance
        # inverse = inverse * self.scaler.scale_ + self.scaler.mean_
        
        inverse[0] = raw_instance.copy()
        return inverse