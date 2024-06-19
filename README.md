# A tutorial on Bayesian inference to identify material parameters in solid mechanics

> THIS IS A WORK IN PROGRESS :hourglass:

This repository contains Jupyter Notebooks to recreate the examples published in the following paper:

- Rappel, H., Beex, L. A., Hale, J. S., Noels, L., & Bordas, S. P. A. (2020). A tutorial on Bayesian inference to identify material parameters in solid mechanics. *Archives of Computational Methods in Engineering*, 27(2), 361-385. [https://doi.org/10.1007/s11831-018-09311-x](https://doi.org/10.1007/s11831-018-09311-x)

## Paper abstract

The aim of this contribution is to explain in a straightforward manner how Bayesian inference can be used to identify material parameters of material models for solids. Bayesian approaches have already been used for this purpose, but most of the literature is not necessarily easy to understand for those new to the field. The reason for this is that most literature focuses either on complex statistical and machine learning concepts and/or on relatively complex mechanical models. In order to introduce the approach as gently as possible, we only focus on stress–strain measurements coming from uniaxial tensile tests and we only treat elastic and elastoplastic material models. Furthermore, the stress–strain measurements are created artificially in order to allow a one-to-one comparison between the true parameter values and the identified parameter distributions.

## Example problems

| Example | Details |
| ------- | ----------- |
| [Linear Elasticity](examples/01-linear-elasticity.ipynb) <br><hr> <a href="https://colab.research.google.com/github/mark-hobbs/bayesian-inference-tutorial/blob/main/examples/01-linear-elasticity.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  | - Use grid search and the standard Metropolis-Hastings algorithm to infer model parameters. <br> - Implement `Prior`, `Likelihood`, `Posterior` and `Model` classes. <br> - Apply gradient descent with momentum to determine the best-fitting model parameters and demonstrate the limitations of point estimates. <br> - Implement a `GradientDescent` class.|
| [Linear Elasticity-Perfect Plasticity](examples/02-linear-elasticity-perfect-plasticity.ipynb) <br><hr> <a href="https://colab.research.google.com/github/mark-hobbs/bayesian-inference-tutorial/blob/main/examples/02-linear-elasticity-perfect-plasticity.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  | - Use the Adaptive Metropolis-Hastings algorithm and analyse sampler performance. <br> - Implement `Sampler` and `Proposal` classes. <br> - Sample complex posterior distributions to deepen understanding.|
| [Linear Elasticity-Linear Hardening](examples/03-linear-elasticity-linear-hardening.ipynb) | - |
| Linear Elasticity-Nonlinear Hardening | - |

There are plans to add two additional examples that explain more advanced concepts:

1) Noise in both stress and strain
2) [Model uncertainty](model-uncertainty.ipynb)

## Getting started

Using [`Pipenv`](https://pipenv.pypa.io/en/latest/):

```shell
$ git clone https://github.com/mark-hobbs/bayesian-inference-tutorial.git
$ cd bayesian-inference-tutorial
$ pipenv install
$ pipenv shell
$ jupyter lab
```

## Dependencies

- NumPy
- SciPy
- Matplotlib
- JupyterLab
- tqdm

**Development dependencies**

- Black

## Further reading

The reader might also be interested in the following paper:

- Rappel, H., Beex, L. A., Noels, L., & Bordas, S. P. A. (2019). Identifying elastoplastic parameters with Bayes’ theorem considering output error, input error and model uncertainty. *Probabilistic Engineering Mechanics*, 55, 28-41. [https://doi.org/10.1016/j.probengmech.2018.08.004](https://doi.org/10.1016/j.probengmech.2018.08.004)

Additional resources that the reader might find useful are listed below

- Hogg, D. W., Bovy, J., & Lang, D. (2010). Data analysis recipes: Fitting a model to data. arXiv preprint [arXiv:1008.4686](https://doi.org/10.48550/arXiv.1008.4686).

- Hogg, D. W., & Foreman-Mackey, D. (2018). Data analysis recipes: Using markov chain monte carlo. *The Astrophysical Journal Supplement Series*, 236(1), 11. [doi.org/10.3847/1538-4365/aab76e](https://doi.org/10.3847/1538-4365/aab76e)

## Contact

If you spot any mistakes then please raise an issue or if you would prefer you can contact me using the following email address:

mhobbs@turing.ac.uk 


