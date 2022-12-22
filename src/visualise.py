import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Times New Roman"]})
plt.rcParams["font.family"] = "Times New Roman"

class Visualise():
    """
    Data visualisation class

    TODO: take a look at ArviZ
    """

    def __init__(self, chain):
        self.chain = chain

    def plot_chain_history(self):
        pass