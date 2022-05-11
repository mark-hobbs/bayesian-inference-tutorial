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

    def __init__(self, E, stress_y):
        self.E = E
        self.stress_y = stress_y

    def calculate_stress(self, E, stress_y, strain):

        strain_y = stress_y / E

        if strain < strain_y:
            return E * strain
        elif strain > strain_y:
            return stress_y

    def stress_strain_response(self, strain):

        stress = np.zeros(np.size(strain))
        for i in range(len(strain)):
            stress[i] = self.calculate_stress(self.E, self.stress_y,
                                              strain[i])

        plt.plot(strain, stress)
        plt.title("Stress-strain graph")
        plt.xlabel("Strain $\epsilon$")
        plt.ylabel("Stress $\sigma$")

    def generate_synthetic_data(self, strain, s_noise, n_data_points):
        """
        The noise in the stress measurements is a normal distribution with a
        zero mean and a standard deviation of s_noise
        """
        strain_data = np.random.choice(strain, n_data_points)
        stress_data = np.zeros(n_data_points)

        for i in range(len(stress_data)):
            stress_data[i] = (self.calculate_stress(self.E, self.stress_y,
                              strain_data[i]) + (s_noise * np.random.normal()))

        return strain_data, stress_data

    def likelihood(self, s_noise, strain, stress,
                   candidate_E, candidate_stress_y):
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

        candidate_E : float
            Young's modulus candidate

        candidate_stress_y : float
            Yield stress candidate

        Returns
        -------

        """
        return ((1 / (s_noise * np.sqrt(2 * np.pi))) * np.exp(-((stress
                - self.calculate_stress(candidate_E, candidate_stress_y,
                                        strain)) ** 2) / (2 * s_noise ** 2)))

    def prior(self, cov_matrix, candidate_x, prior_x):
        """
        Prior distribution
        
        Parameters
        ----------
        cov_matrix : ndarray
            Covariance matrix

        candidate_x : ndarray
            Candidate vector [E, stress_y]

        prior_x : ndarray
            Prior vector [E, stress_y]
            TODO: is this variable a prior? or just an initial guess?

        Returns
        -------
        
        """
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        numerator = np.matmul(np.transpose(candidate_x - prior_x),
                              np.matmul(inv_cov_matrix, candidate_x - prior_x))
        return np.exp(-numerator / 2)

    def posterior(self, strain_data, stress_data, s_noise,
                  cov_matrix, candidate_x, prior_x):
        """
        Calculate the posterior

        Parameters
        ----------
        strain_data : ndarray
            Experimentally measured strain data
        
        stress_data : ndarray
            Experimental measured stress data

        cov_matrix : ndarray
            TODO: attribute of the sampler?

        candidate : ndarray
            TODO: attribute of the sampler?

        Returns
        -------

        """
        total = 0
        for i in range(len(stress_data)):
            total += self.likelihood(s_noise,
                                     strain_data[i-1], stress_data[i-1],
                                     candidate_x[0], candidate_x[1])
        return self.prior(cov_matrix, candidate_x, prior_x) * total

    def proposal_distribution(self):
        """
        Proposal distribution (q)
        """
        pass


class LinearElasticityLinearHardening(MaterialModel):
    pass


class LinearElasticityNonlinearHardening(MaterialModel):
    pass
