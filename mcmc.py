
"""
https://num.pyro.ai/en/stable/mcmc.html
"""


class MCMC():

    def __init__(self, num_warmup=3000, num_samples=10000):

        self.num_warmup = num_warmup
        self.num_samples = num_samples

    def run(self):
        """
        for sample in range(num_samples):
        """
        pass

    def prior(self):
        pass

    def posterior(self):
        pass

    def proposal_distribution(self):
        """
        Proposal distribution (q)
        """
        pass

    def sampler(self):
        pass
