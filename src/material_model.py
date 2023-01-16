import numpy as np
import matplotlib.pyplot as plt


class MaterialModel():

    def __init__(self):
        pass

    def stress_strain_response(self, strain):
        """
        Plot the stress-strain response
        """
        stress = np.zeros(np.size(strain))
        for i in range(len(strain)):
            stress[i] = self.calculate_stress(strain[i])

        plt.plot(strain, stress, color='C1', label="True model")
        plt.title("Stress-strain graph")
        plt.xlabel("Strain $\epsilon$")
        plt.ylabel("Stress $\sigma$")
        plt.legend()

    def generate_synthetic_data(self, strain, n_data_points, seed=None):
        """
        The noise in the stress measurements is a normal distribution with a
        zero mean and a standard deviation of s_noise
        """
        np.random.seed(seed)
        strain_data = np.random.choice(strain, n_data_points)
        stress_data = np.zeros(n_data_points)

        for i in range(len(stress_data)):
            stress_data[i] = (self.calculate_stress(strain_data[i])
                              + (self.s_noise * np.random.normal()))

        return strain_data, stress_data

    def set_priors(self, x_prior, cov_matrix_prior):
        """
        It is considered bad practice to set attributes outside of the
        __init__ method
        """
        self.x_prior = x_prior
        self.cov_matrix_prior = cov_matrix_prior

    def likelihood(self):
        pass

    def prior(self, x_i):
        """
        Prior distribution

        Parameters
        ----------
        x_i : ndarray
            Candidate vector [E, stress_y]

        x_prior : ndarray
            Prior vector [E, stress_y]
            TODO: is this variable a prior? or just an initial guess?

        cov_matrix : ndarray
            Covariance matrix

        Returns
        -------

        """
        inv_cov_matrix = np.linalg.inv(self.cov_matrix_prior)
        numerator = np.matmul(np.transpose(x_i - self.x_prior),
                              np.matmul(inv_cov_matrix, x_i - self.x_prior))
        return np.exp(-numerator / 2)

    def posterior(self):
        pass


class LinearElasticity(MaterialModel):
    """
    Linear Elastic material model class

    Attributes
    ----------
    n_p : int
        Number of unknown parameters

    E : float
        Young's modulus (exact value that we wish to infer from the noisy
        experimental observations)

    s_noise : float
        Noise in the stress observations (determined via calibration of the
        testing machine). Normal distribution with a zero mean and a standard
        deviation of s_noise.
            Methods
    -------

    Notes
    -----
    """

    def __init__(self, E, s_noise):
        self.n_p = 1
        self.E = E

        self.s_noise = s_noise

    def calculate_stress(self, strain, E=None):
        E = E or self.E
        return E * strain 


class LinearElasticityPerfectPlasticity(MaterialModel):
    """
    Linear Elasticity-Perfect Plasticity material model class

    Attributes
    ----------
    n_p : int
        Number of unknown parameters

    E : float
        Young's modulus (exact value that we wish to infer from the noisy
        experimental observations)

    stress_y : float
        Yield stress (exact value that we wish to infer from the noisy
        experimental observations)

    s_noise : float
        Noise in the stress observations (determined via calibration of the
        testing machine). Normal distribution with a zero mean and a standard
        deviation of s_noise.

    Methods
    -------

    Notes
    -----
    """

    def __init__(self, E, stress_y, s_noise):
        self.n_p = 2
        self.E = E
        self.stress_y = stress_y

        self.s_noise = s_noise
        self.x_prior = None
        self.cov_matrix_prior = None

    def calculate_stress(self, strain, E=None, stress_y=None):
        E = E or self.E
        stress_y = stress_y or self.stress_y

        strain_y = stress_y / E

        if strain <= strain_y:
            return E * strain
        elif strain > strain_y:
            return stress_y

    def likelihood(self, strain, stress, E_candidate, stress_y_candidate):
        """
        Likelihood function for a single stress measurement

        Parameters
        ----------
        s_noise : float
            Noise in the stress measurement (determined experimentally)

        strain : float
            Experimentally measured strain

        stress : float
            Experimentally measured stress

        E_candidate : float
            Young's modulus candidate

        stress_y_candidate : float
            Yield stress candidate

        Returns
        -------
        likelihood : float
            Likelihood for a single stress measurement

        """
        alpha = 1 / (self.s_noise * np.sqrt(2 * np.pi))
        beta = stress - self.calculate_stress(strain,
                                              E=E_candidate,
                                              stress_y=stress_y_candidate)
        return alpha * np.exp(- (beta ** 2) / (2 * self.s_noise ** 2))

    def likelihood_(self, strain_data, stress_data,
                    E_candidate, stress_y_candidate):
        """
        Likelihood function for full data set

        Parameters
        ----------
        s_noise : float
            Noise in the stress measurement (determined experimentally)

        strain_data : ndarray
            Experimentally measured strain data

        stress_data : ndarray
            Experimental measured stress data

        E_candidate : float
            Young's modulus candidate

        stress_y_candidate : float
            Yield stress candidate

        Returns
        -------
        likelihood : float
            Likelihood

        """
        alpha = 1 / (self.s_noise * np.sqrt(2 * np.pi))
        beta = 0
        for i in range(len(stress_data)):
            beta += (stress_data[i]
                     - self.calculate_stress(strain_data[i],
                                             E=E_candidate,
                                             stress_y=stress_y_candidate,
                                             )) ** 2
        return alpha * np.exp(- beta / (2 * self.s_noise ** 2))

    def log_likelihood(self, strain, stress, E_candidate, stress_y_candidate):
        """
        Log-likelihood function for a single stress measurement

        Parameters
        ----------
        s_noise : float
            Noise in the stress measurement (determined experimentally)

        strain : float
            Experimentally measured strain

        stress : float
            Experimentally measured stress

        E_candidate : float
            Young's modulus candidate

        stress_y_candidate : float
            Yield stress candidate

        Returns
        -------
        log_likelihood : float
            Log-likelihood for a single stress measurement

        Notes
        -----
        TODO: check the formula is correct
        """
        return np.log(self.likelihood(strain, stress, E_candidate,
                                      stress_y_candidate))

    def log_prior(self, x_i):
        """
        Log-prior distribution

        Parameters
        ----------
        x_i : ndarray
            Candidate vector [E, stress_y]

        x_prior : ndarray
            Prior vector [E, stress_y]
            TODO: is this variable a prior? or just an initial guess?

        cov_matrix : ndarray
            Covariance matrix

        Returns
        -------

        """
        return np.log(self.prior(x_i))

    def posterior(self, strain_data, stress_data, x_i):
        """
        Calculate the posterior

        Parameters
        ----------
        strain_data : ndarray
            Experimentally measured strain data

        stress_data : ndarray
            Experimental measured stress data

        x_i : ndarray
            Candidate vector [E, stress_y]

        Returns
        -------

        """
        alpha = []
        for i in range(len(stress_data)):
            alpha.append(self.likelihood(strain_data[i], stress_data[i],
                                         x_i[0], x_i[1]))
        return self.prior(x_i) * np.prod(alpha)

    def posterior_(self, strain_data, stress_data, x_i):
        """
        Calculate the posterior

        Parameters
        ----------
        strain_data : ndarray
            Experimentally measured strain data

        stress_data : ndarray
            Experimental measured stress data

        x_i : ndarray
            Candidate vector [E, stress_y]

        Returns
        -------

        """
        return self.prior(x_i) * self.likelihood_(strain_data, stress_data,
                                                  x_i[0], x_i[1])

    def log_posterior(self, strain_data, stress_data, x_i):
        """
        Calculate the log-posterior

        Parameters
        ----------
        strain_data : ndarray
            Experimentally measured strain data

        stress_data : ndarray
            Experimental measured stress data

        x_i : ndarray
            Candidate vector [E, stress_y]

        Returns
        -------

        """
        alpha = []
        for i in range(len(stress_data)):
            alpha.append(self.log_likelihood(strain_data[i], stress_data[i],
                                             x_i[0], x_i[1]))
        return self.log_prior(x_i) * np.prod(alpha)

    def calculate_PPD(self, strain, x_hist, burn=3000):
        """
        Calculate the posterior predictive distribution (PPD)

        TODO: move method to a more suitable class?
        """

        for sample in x_hist[burn:]:
            stress = np.zeros(np.size(strain))

            for i in range(len(strain)):
                stress[i] = self.calculate_stress(strain[i], E=sample[0],
                                                  stress_y=sample[1])

            plt.plot(strain, stress, color='dimgray', alpha=.005)
            plt.title("Stress-strain graph")
            plt.xlabel("Strain $\epsilon$")
            plt.ylabel("Stress $\sigma$")

        strain_data, stress_data = self.generate_synthetic_data(strain, 100)
        plt.scatter(strain_data, stress_data)


class LinearElasticityLinearHardening(MaterialModel):
    """
    Linear Elasticity-Linear Hardening material model class

    Attributes
    ----------
    n_p : int
        Number of unknown parameters

    E : float
        Young's modulus (exact value that we wish to infer from the noisy
        experimental observations)

    stress_y : float
        Yield stress (exact value that we wish to infer from the noisy
        experimental observations)

    H : float
        Plastic modulus (exact value that we wish to infer from the noise
        experimental observations)

    s_noise : float
        Noise in the stress observations (determined via calibration of the
        testing machine). Normal distribution with a zero mean and a standard
        deviation of s_noise.

    Methods
    -------

    Notes
    -----
    """

    def __init__(self, E, stress_y, H, s_noise):
        self.n_p = 3
        self.E = E
        self.stress_y = stress_y
        self.H = H

        self.s_noise = s_noise
        self.x_prior = None
        self.cov_matrix_prior = None

    def calculate_stress(self, strain, E=None, stress_y=None, H=None):
        E = E or self.E
        stress_y = stress_y or self.stress_y
        H = H or self.H

        strain_y = stress_y / E

        if strain <= strain_y:
            return E * strain
        elif strain > strain_y:
            strain_p = strain - strain_y
            return stress_y + (H * strain_p)

    def likelihood(self, strain, stress, E_candidate, stress_y_candidate,
                   H_candidate):
        """
        Likelihood function for a single stress measurement

        Parameters
        ----------
        s_noise : float
            Noise in the stress measurement (determined experimentally)

        strain : float
            Experimentally measured strain

        stress : float
            Experimentally measured stress

        E_candidate : float
            Young's modulus candidate

        stress_y_candidate : float
            Yield stress candidate

        H_candidate : float
            Plastic modulus candidate

        Returns
        -------
        likelihood : float
            Likelihood for a single stress measurement

        # TODO: Is it possible to move this and the prior function to the
        parent class?

        """
        alpha = 1 / (self.s_noise * np.sqrt(2 * np.pi))
        beta = stress - self.calculate_stress(strain,
                                              E=E_candidate,
                                              stress_y=stress_y_candidate,
                                              H=H_candidate)
        return alpha * np.exp(- (beta ** 2) / (2 * self.s_noise ** 2))

    def posterior(self, strain_data, stress_data, x_i):
        """
        Calculate the posterior

        Parameters
        ----------
        strain_data : ndarray
            Experimentally measured strain data

        stress_data : ndarray
            Experimental measured stress data

        x_i : ndarray
            Candidate vector [E, stress_y]

        Returns
        -------

        """
        alpha = []
        for i in range(len(stress_data)):
            alpha.append(self.likelihood(strain_data[i], stress_data[i],
                                         x_i[0], x_i[1], x_i[2]))
        return self.prior(x_i) * np.prod(alpha)


class LinearElasticityNonlinearHardening(MaterialModel):
    pass
