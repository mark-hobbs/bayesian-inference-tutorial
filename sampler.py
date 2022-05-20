
import numpy as np


class Sampler():

    def __init__(self):
        pass


class MetropolisHastings(Sampler):
    """
    Metropolis-Hastings sampler class

    Attributes
    ----------
    model : MaterialModel
        Material model class (linear elastic,
                              linear elastic-perfectly plastic,
                              linear elastic-linear hardening,
                              linear elastic-nonlinear hardening)

    data : ndarray
        Experimental stress-strain data (observations)

    n_samples : int
        Number of samples (default = 10,000)

    burn : int
        Burn-in (default = 3,000). Discard the first 3,000 samples. Burn-in is
        intended to give the Markov Chain time to reach its equilibrium
        distribution.

    Methods
    -------
    
    Notes
    -----
    """

    def __init__(self, model, data, n_samples=1E4, burn=3E3):
        self.model = model
        self.data = data
        self.n_samples = int(n_samples)
        self.burn = int(burn)

    def sample(self, x_0):
        """
        Sampling loop

        Parameters
        ----------
        x_0 : ndarray
            Initial sample

        Returns
        -------

        """
        x_i = x_0.copy()
        for i in range(1, self.n_samples):
            x_i = self.sample_step(x_i)

    def sample_step(self, x_i):
        """
        Draw a new sample

        Parameters
        ----------
        x_i : ndarray
            Current sample

        x_p : ndarray
            New sample (x_p) is proposed by drawing from a proposal
            distribution

        Returns
        -------
        x_i : ndarray
            Current sample

        """
        x_p = self.draw_proposal(x_i)
        pi_x_i = self.calculate_posterior(x_i)
        pi_x_p = self.calculate_posterior(x_p)
        self.accept_or_reject()

    def accept_or_reject(self):
        """
        Accept of reject a new candidate

        Parameters
        ----------

        Returns
        -------

        """
        alpha = self.calculate_acceptance_ratio()
        u = self.generate_uniform_random_number()
        # Accept if u < alpha
        # Reject if u > alpha

    def calculate_acceptance_ratio(self):
        return min(1, pi_x_p, pi_x_i)

    def generate_uniform_random_number():
        return np.random.uniform(low=0.0, high=1.0)

    def calculate_posterior(self):
        """
        Calculate the posterior

        Parameters
        ----------
        x_i : ndarray
            Current sample

        model : MaterialModel class
            Material model class

        Returns
        -------

        """
        return self.model.posterior(self.data)

    def draw_proposal(self, x_i):
        """
        Draw x (candidate) from proposal distribution

        Parameters
        ----------
        x_i : ndarray
            Current sample
        
        model : MaterialModel class
            Material model class

        Returns
        -------
        x_p : ndarray
            Proposed sample (candidate sample)

        """
        return x_i * (self.model.proposal_distribution()
                      * np.transpose(
                          np.random.normal(size=(1, self.model.n_p))))
