class Interpretation(object):
    def __init__(self, 
                 plain_instance,
                 counter_factual_instance=None,
                 counter_factual_distance=None,
                 local_absolute_instance=None,
                 local_absolute_distance=None):
        self.plain_instance = plain_instance
        self.counter_factual_instance = counter_factual_instance
        self.counter_factual_distance = counter_factual_distance
        self.local_absolute_instance = local_absolute_instance
        self.local_absolute_distance = local_absolute_distance