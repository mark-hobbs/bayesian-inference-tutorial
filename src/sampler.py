
import numpy as np
from tqdm import tqdm


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

    n_chains : int
        Number of chains (default = 1). This value should be equal to or less
        than the number of available threads.

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
        x_hist: ndarray
            Parameter chain

        pdf_hist : ndarray
            Probability density of every sample

        """
        x_i = x_0.copy()
        x_hist = np.zeros([self.n_samples, self.model.n_p])
        pdf_hist = np.zeros(self.n_samples)

        for i in tqdm(range(self.n_samples)):
            x_i, pdf = self.sample_step(x_i)
            x_hist[i, :] = np.transpose(x_i)
            pdf_hist[i] = pdf

        return x_hist, pdf_hist

    def sample_step(self, x_i):
        """
        Draw a new sample

        Parameters
        ----------
        x_i : ndarray
            Current sample

        Returns
        -------
        x_i : ndarray
            Current sample

        """
        x_p = self.draw_proposal(x_i)
        pi_x_i = self.calculate_posterior(x_i)
        pi_x_p = self.calculate_posterior(x_p)
        return self.accept_or_reject(x_i, x_p, pi_x_i, pi_x_p)

    def accept_or_reject(self, x_i, x_p, pi_x_i, pi_x_p):
        """
        Accept of reject a new candidate

        Parameters
        ----------
        x_i : ndarray
            Current sample

        x_p : ndarray
            New sample (x_p) is proposed by drawing from a proposal
            distribution

        Returns
        -------

        """
        alpha = self.calculate_acceptance_ratio(pi_x_p, pi_x_i)
        u = self.generate_uniform_random_number()
        if u <= alpha:  # Accept proposal
            return x_p, pi_x_p
        if u > alpha:  # Reject proposal
            return x_i, pi_x_i

    def calculate_acceptance_ratio(self, pi_x_p, pi_x_i):
        return min(1, pi_x_p / pi_x_i)

    @staticmethod
    def generate_uniform_random_number():
        return np.random.uniform(low=0.0, high=1.0)

    def calculate_posterior(self, x_i):
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
        return self.model.posterior(self.data[0], self.data[1], x_i)

    def draw_proposal(self, x_i):
        """
        Draw x (candidate) from proposal distribution q

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
        return x_i + (self.model.compute_gamma()
                      * np.transpose(
                          np.random.normal(size=(1, self.model.n_p))))

    def draw_proposal_adaptive(self):
        """
        Draw x (candidate) from adaptive proposal distribution

        Parameters
        ----------
        K : ndarray
            All previous samples are stored in matrix K of size n_k x n_p,
            where n_k is... and n_p is the number of unknown parameters.

        Returns
        -------

        Notes
        -----
        Eq. (58) in Rappel et al., (2018)

        TODO: move to AdaptiveMetropolisHastings class

        """
        pass

    def calculate_mean(self, x_hist):
        """
        Calculate the mean of the posterior distribution

        Parameters
        ----------
        x_hist : ndarray

        Returns
        -------
        x_mean : ndarray

        TODO: move to Sampler or Utilities class

        """
        x_hist_burned = x_hist[self.burn:]
        return np.mean(x_hist_burned, 0)

    def calculate_covariance(self, x_hist):
        """
        Calculate the covariance (correlation) between the model parameters

        Parameters
        ----------
        x_hist : ndarray

        Returns
        -------
        x_cov : ndarray

        """
        x_hist_burned = x_hist[self.burn:]
        return np.cov(np.transpose(x_hist_burned))

    def calculate_95_percent_credible_region(self):
        pass
