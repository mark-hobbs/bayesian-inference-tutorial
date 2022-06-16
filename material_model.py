import numpy as np
import matplotlib.pyplot as plt


class MaterialModel():

    def __init__(self):
        pass

    def calculate_stress(self):
        pass

    def stress_strain_response(self):
        pass

    def generate_synthetic_data(self):
        pass

    def noise_model(self):
        """
        Additive noise model
        """
        pass


class LinearElasticity(MaterialModel):
    pass


class LinearElasticityPerfectPlasticity():
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

    def calculate_stress(self, E, stress_y, strain):

        strain_y = stress_y / E

        if strain < strain_y:
            return E * strain
        elif strain > strain_y:
            return stress_y

    def stress_strain_response(self, strain):
        """
        Plot the stress-strain response
        """
        stress = np.zeros(np.size(strain))
        for i in range(len(strain)):
            stress[i] = self.calculate_stress(self.E, self.stress_y,
                                              strain[i])

        plt.plot(strain, stress)
        plt.title("Stress-strain graph")
        plt.xlabel("Strain $\epsilon$")
        plt.ylabel("Stress $\sigma$")

    def generate_synthetic_data(self, strain, n_data_points):
        """
        The noise in the stress measurements is a normal distribution with a
        zero mean and a standard deviation of s_noise
        """
        strain_data = np.random.choice(strain, n_data_points)
        stress_data = np.zeros(n_data_points)

        for i in range(len(stress_data)):
            stress_data[i] = (self.calculate_stress(self.E, self.stress_y,
                              strain_data[i])
                              + (self.s_noise * np.random.normal()))

        return strain_data, stress_data

    def set_priors(self, x_prior, cov_matrix_prior):
        """
        It is considered bad practice to set attributes outside of the
        __init__ method
        """
        self.x_prior = x_prior
        self.cov_matrix_prior = cov_matrix_prior

    def likelihood_single_measurement(self, strain, stress,
                                      E_candidate, stress_y_candidate):
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
        beta = stress - self.calculate_stress(E_candidate,
                                              stress_y_candidate,
                                              strain)
        return alpha * np.exp(- (beta ** 2) / (2 * self.s_noise ** 2))

    def likelihood(self, strain_data, stress_data,
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
                     - self.calculate_stress(E_candidate,
                                             stress_y_candidate,
                                             strain_data[i])) ** 2
        return alpha * np.exp(- beta / (2 * self.s_noise ** 2))

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
        return self.prior(x_i) * self.likelihood(strain_data, stress_data,
                                                 x_i[0], x_i[1])

    # def posterior(self, strain_data, stress_data, x_i):
    #     """
    #     Calculate the posterior

    #     Parameters
    #     ----------
    #     strain_data : ndarray
    #         Experimentally measured strain data

    #     stress_data : ndarray
    #         Experimental measured stress data

    #     x_i : ndarray
    #         Candidate vector [E, stress_y]

    #     Returns
    #     -------

    #     """
    #     alpha = 0
    #     for i in range(len(stress_data)):
    #         alpha += self.likelihood_single_measurement(strain_data[i],
    #                                                     stress_data[i],
    #                                                     x_i[0], x_i[1])

    #     return self.prior(x_i) * alpha

    def proposal_distribution(self):
        """
        Proposal distribution (q)

        Parameters
        ----------
        n_p : int
            Number of unknown parameters

        Returns
        -------
        gamma : float
            Parameter that determines the width of the proposal distribution
            and must be tuned to obtain an efficient and converging algorithm

        """
        return [[5], [0.1]] / np.sqrt(self.n_p)


class LinearElasticityLinearHardening(MaterialModel):
    pass


class LinearElasticityNonlinearHardening(MaterialModel):
    pass
