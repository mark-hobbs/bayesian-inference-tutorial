import numpy as np
import matplotlib.pyplot as plt


class MaterialModel():

    def __init__(self):
        pass

    def calculate_stress(self):
        pass

    def plot_stress_strain(self):
        pass


class LinearElasticity(MaterialModel):
    pass


class LinearElasticityPerfectPlasticity():

    def __init__(self, E, stress_y):
        self.E = E
        self.stress_y = stress_y

    def calculate_stress(self, strain):

        stress = []
        strain_y = self.stress_y / self.E

        for i in range(len(strain)):
            if strain[i] < strain_y:
                tmp = self.E * strain[i]
            elif strain[i] > strain_y:
                tmp = self.stress_y
            stress.append(tmp)

        return stress

    def stress_strain_graph(self, strain):
        stress = self.calculate_stress(strain)
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
        stress_data = (self.calculate_stress(strain_data)
                       + (s_noise * np.random.normal(size=n_data_points)))
        return strain_data, stress_data


class LinearElasticityLinearHardening(MaterialModel):
    pass


class LinearElasticityNonlinearHardening(MaterialModel):
    pass
