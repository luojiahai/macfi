class Instance(object):
    def __init__(self,
                 raw_instance):
        self.raw_instance = raw_instance
        self.predict_proba = None
        self.counter_factual_instance = None

    def find_counter_factual_instance(self):
        return None