class Interpretation(object):
    def __init__(self, 
                 plain_instance,
                 counter_factual_instance,
                 distance):
        self.plain_instance = plain_instance
        self.counter_factual_instance = counter_factual_instance
        self.distance = distance