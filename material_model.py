

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
            if strain < strain_y:
                tmp = self.E * strain
            elif strain > strain_y:
                tmp = self.stress_y
            stress.append(tmp)

        return stress

    def plot_stress_strain():
        pass


class LinearElasticityLinearHardening(MaterialModel):
    pass


class LinearElasticityNonlinearHardening(MaterialModel):
    pass
