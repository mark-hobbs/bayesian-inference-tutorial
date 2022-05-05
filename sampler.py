
import numpy as np


class Sampler():

    def __init__(self):
        pass


class MetropolisHastings(Sampler):

    def __init__(self, model):
        self.model = model

    def sample(self):
        x_p = self.model.draw_proposal()
        pi_x_i = self.model.calculate_posterior(x_i)
        pi_x_p = self.model.calculate_posterior(x_p)
        self.accept_or_reject()

    def accept_or_reject(self):
        alpha = self.calculate_acceptance_ratio()
        u = self.generate_uniform_random_number()
        # Accept if u < alpha
        # Reject if u > alpha

    def calculate_acceptance_ratio(self):
        return min(1, pi_x_p, pi_x_i)

    def generate_uniform_random_number():
        return np.random.uniform(low=0.0, high=1.0)
