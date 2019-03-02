class Interpretation(object):
    def __init__(self, 
                 plain_instance,
                 counter_factual_instance,
                 pi_predict_proba,
                 cfi_predict_proba,
                 distance):
        self.plain_instance = plain_instance
        self.counter_factual_instance = counter_factual_instance
        self.pi_predict_proba = pi_predict_proba
        self.cfi_predict_proba = cfi_predict_proba
        self.distance = distance