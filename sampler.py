

class Sampler():

    def __init__(self):
        pass


class MetropolisHastings(Sampler):

    def __init__(self):
        pass

    def sample(self):
        # Draw candidate from the proposal distribution (q)
        # Accept or reject candidate
        pass

    def accept_or_reject(self):
        alpha = self.calculate_acceptance_ratio()
        u = self.generate_uniform_random_number()
        # Accept if u < alpha
        # Reject if u > alpha

    def calculate_acceptance_ratio(self):
        return min(1, pdf_candidate_x, pdf_current_x)

    def generate_uniform_random_number():
        return numpy.random.uniform(low=0.0, high=1.0)
