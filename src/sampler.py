import numpy as np
from tqdm import tqdm


class Sampler:

    def __init__(self):
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
        x_hist_burned = x_hist[self.burn :]
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
        x_hist_burned = x_hist[self.burn :]
        return np.cov(np.transpose(x_hist_burned))

    def calculate_MAP_point(self, x_hist, pdf_hist):
        """
        Calculate the maximum a posteriori probability (MAP) point - the point
        at which the posterior distribution is (globally) maximum

        Parameters
        ----------
        x_hist : ndarray

        pdf_hist : ndarray

        Returns
        -------
        MAP_point : ndarray

        """
        x_hist_burned = x_hist[self.burn :]
        pdf_hist_burned = pdf_hist[self.burn :]
        return x_hist_burned[np.argmax(pdf_hist_burned), :]

    def plot_chain_histogram(self):
        """
        Histogram of the Markov chain
        """
        pass

    def plot_chain_density(self):
        """
        Density plot of the Markov chain
        """
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

    def __init__(self, model, data, n_samples=1e4, burn=3e3, step_size=2.38):
        self.model = model
        self.data = data
        self.n_samples = int(n_samples)
        self.burn = int(burn)
        self.step_size = step_size

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

        accept_rate: ndarray
            The optimal acceptance rate is 0.25. If the acceptance rate is too
            high or too low it will slow down the convergence of the Markov
            chain.

        """
        x_i = x_0.copy()
        x_hist = np.zeros([self.n_samples, self.model.n_p])
        pdf_hist = np.zeros(self.n_samples)
        accept_rate = np.zeros(self.n_samples)
        n_accept = 0

        for i in tqdm(range(self.n_samples)):
            x_i, pdf, n_accept = self.sample_step(x_i, n_accept)
            x_hist[i, :] = np.transpose(x_i)
            pdf_hist[i] = pdf
            accept_rate[i] = n_accept / (i + 1)

        return x_hist, pdf_hist, accept_rate

    def sample_step(self, x_i, n_accept):
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
        return self.accept_or_reject(x_i, x_p, pi_x_i, pi_x_p, n_accept)

    def accept_or_reject(self, x_i, x_p, pi_x_i, pi_x_p, n_accept):
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
            n_accept += 1
            return x_p, pi_x_p, n_accept
        if u > alpha:  # Reject proposal
            return x_i, pi_x_i, n_accept

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
        return x_i + (self.compute_gamma() * np.random.normal(size=(self.model.n_p, 1)))

    def compute_gamma(self):
        """
        Compute gamma, a parameter that determines the width of the proposal
        distribution and must be tuned to obtain an efficient and converging
        algorithm

        Parameters
        ----------
        step_size : ndarray


        n_p : int
            Number of unknown parameters

        Returns
        -------
        gamma : float
            Parameter that determines the width of the proposal distribution
            and must be tuned to obtain an efficient and converging algorithm

        TODO: rename compute_gamma()?
        TODO: should this move to the Sampler class?

        """
        return self.step_size / np.sqrt(self.model.n_p)

    def calculate_95_percent_credible_region(self):
        pass


class AdaptiveMetropolisHastings(Sampler):
    """
    Adaptive Metropolis-Hastings sampler class - the adaptive proposal method
    updates the width of the proposal distribution, using the existing
    knowledge of the posterior. The existing knowledge is based on the previous
    samples.

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

    update_freq: int
        Proposal distribution update frequency (default = 1000). The
        frequency at which the proposal distribution is updated.

    n_K : ndarray
        Number of samples / update frequency

    Methods
    -------

    Notes
    -----
    """

    def __init__(
        self, model, data, n_samples=1e4, burn=3e3, update_freq=1e3, step_size=2.38
    ):
        self.model = model
        self.data = data
        self.n_samples = int(n_samples)
        self.burn = int(burn)
        self.update_freq = int(update_freq)
        self.n_K = int(self.n_samples / self.update_freq)
        self.step_size = step_size

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

        accept_rate: ndarray
            The optimal acceptance rate is 0.25. If the acceptance rate is too
            high or too low it will slow down the convergence of the Markov
            chain.

        """
        x_i = x_0.copy()
        x_hist = np.zeros([self.n_samples, self.model.n_p])
        pdf_hist = np.zeros(self.n_samples)
        accept_rate = np.zeros(self.n_samples)
        K = np.zeros([self.n_K, self.model.n_p])
        K_tilde = self.calculate_K_tilde(K)
        n_accept = 0
        counter = 0

        for i in tqdm(range(self.n_samples)):

            if i % self.update_freq == 0:
                K[counter, :] = np.transpose(x_i)
                K_tilde = self.calculate_K_tilde(K)
                counter += 1

            x_i, pdf, n_accept = self.sample_step(x_i, K_tilde, n_accept)
            x_hist[i, :] = np.transpose(x_i)
            pdf_hist[i] = pdf
            accept_rate[i] = n_accept / (i + 1)

        return x_hist, pdf_hist, accept_rate

    def sample_step(self, x_i, K_tilde, n_accept):
        """
        Draw a new sample using the Metropolis-Hastings algorithm with the
        adaptive proposal technique

        Parameters
        ----------
        x_i : ndarray
            Current sample

        Returns
        -------
        x_i : ndarray
            Current sample

        """
        x_p = self.draw_proposal(x_i, K_tilde)
        pi_x_i = self.calculate_posterior(x_i)
        pi_x_p = self.calculate_posterior(x_p)
        return self.accept_or_reject(x_i, x_p, pi_x_i, pi_x_p, n_accept)

    def accept_or_reject(self, x_i, x_p, pi_x_i, pi_x_p, n_accept):
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
            n_accept += 1
            return x_p, pi_x_p, n_accept
        if u > alpha:  # Reject proposal
            return x_i, pi_x_i, n_accept

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

    def draw_proposal(self, x_i, K_tilde):
        """
        Draw x (candidate) from adaptive proposal distribution q

        Parameters
        ----------
        x_i : ndarray
            Current sample

        model : MaterialModel class
            Material model class

        K_tilde : ndarray
            Write description... n_K x n_p

        Returns
        -------
        x_p : ndarray
            Proposed sample (candidate sample)

        Notes
        -----
        Eq. (58) in Rappel et al., (2018)

        """
        return x_i + (
            (self.compute_gamma() / np.sqrt(self.n_K - 1))
            * np.matmul(np.transpose(K_tilde), np.random.normal(size=(self.n_K, 1)))
        )

    def compute_gamma(self):
        """
        Compute gamma, a parameter that determines the width of the proposal
        distribution and must be tuned to obtain an efficient and converging
        algorithm

        Parameters
        ----------
        step_size : ndarray


        n_p : int
            Number of unknown parameters

        Returns
        -------
        gamma : float
            Parameter that determines the width of the proposal distribution
            and must be tuned to obtain an efficient and converging algorithm

        TODO: rename compute_gamma()?
        TODO: should this move to the Sampler class?

        """
        return self.step_size / np.sqrt(self.model.n_p)

    def calculate_K_tilde(self, K):
        """
        Calculate the centred matrix

        Parameters
        ----------
        K : ndarray
            Sample chain - all previous samples are stored in matrix K of size
            n_k x n_p, where n_k is... and n_p is the number of unknown
            parameters. Each row represents one sampled point.

        K_mean : ndarray
            History (evolution) of mean value of all previous samples
            (n_samples x n_p)

        Returns
        -------
        K_tilde : ndarray
            The centred matrix n_K x n_p (K_tilde = K - E[K])
        """
        return K - np.mean(K, 0)
